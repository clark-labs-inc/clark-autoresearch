//! The optimization loop body.
//!
//! This is the architectural core that turns the crate from a state model into an
//! execution-agnostic optimizer. [`optimize`] runs the GEPA loop —
//! propose → minibatch-evaluate → accept → full-evaluate → Pareto-update — but
//! every provider call, sandbox, and evaluator stays behind the
//! [`ResearchAdapter`] and [`Proposer`] traits. the crate owns the loop body and
//! the state; the host owns execution.
//!
//! Compare GEPA's `gepa.core.engine.GEPAEngine.run`, which bundles the
//! reflection LM and the candidate pool inside the engine. the crate keeps the
//! same loop shape (parent selection, minibatch acceptance, full-valset
//! evaluation, Pareto frontier update via [`ExperimentGraph::evaluated_pool`])
//! while leaving execution to the host — and adds the pieces GEPA lacks:
//! mode-aware parent selection, ledger-backed reflection, and gate-gated
//! `Validate` acceptance.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::acceptance::AcceptanceCriterion;
use crate::adapter::{Candidate, EvaluationBatch, GateRunner, ResearchAdapter};
use crate::cache::{EvaluationCache, cached_evaluate};
use crate::graph::{ExperimentGraph, ExperimentId, ExperimentNode, Hypothesis, ResearchMode};
use crate::ids::stable_unit;
use crate::lm_loop::{EvidenceConfidence, ResearchLedger};
use crate::opportunity::ResearchBias;
use crate::outcome::{Metric, Objectives, OutcomeStatus, TrialOutcome};
use crate::pareto::{FrontierType, non_dominated_set, pareto_front};
use crate::proposer::Proposer;
use crate::stop::{LoopSnapshot, StopCondition};

/// How many ledger items the reflection dossier surfaces each iteration.
const REFLECTION_HISTORY_ITEMS: usize = 16;

/// A recorded rejection, with the iteration and the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectionRecord {
    pub iteration: u32,
    pub reason: String,
}

/// Mutable state the optimization loop owns.
///
/// Wraps the [`ExperimentGraph`] (lineage + outcomes), the optimization
/// objectives, the acceptance policy, the explore/exploit/validate bias, the
/// research ledger (for reflection memory), and a map from node id to the
/// [`Candidate`] that produced it.
pub struct OptimizationState {
    pub graph: ExperimentGraph,
    pub metric: Metric,
    pub objectives: Objectives,
    pub frontier_type: FrontierType,
    pub acceptance: AcceptanceCriterion,
    pub bias: ResearchBias,
    pub snapshot: LoopSnapshot,
    pub best_score: Option<f64>,
    /// Node id -> candidate. The seed candidate is stored under the root id.
    pub candidates: BTreeMap<ExperimentId, Candidate>,
    /// The research ledger, rendered into the proposer's reflection prompt so
    /// reflection reads the whole history, not just the last minibatch.
    pub ledger: ResearchLedger,
    /// Rejections recorded during the loop, with reasons (score or gate).
    pub rejections: Vec<RejectionRecord>,
}

impl OptimizationState {
    /// Create a new state with a seed candidate and a single primary metric.
    pub fn new(metric: Metric, seed_candidate: Candidate) -> Self {
        let mut candidates = BTreeMap::new();
        candidates.insert("root".to_string(), seed_candidate);
        let objectives = Objectives::single(metric.clone());
        let ledger = ResearchLedger::new("optimize", vec![metric.name.clone()]);
        Self {
            graph: ExperimentGraph::new("optimize"),
            metric,
            objectives,
            frontier_type: FrontierType::Instance,
            acceptance: AcceptanceCriterion::StrictImprovement,
            bias: ResearchBias::default(),
            snapshot: LoopSnapshot::default(),
            best_score: None,
            candidates,
            ledger,
            rejections: Vec::new(),
        }
    }

    pub fn with_objectives(mut self, objectives: Objectives) -> Self {
        self.objectives = objectives;
        self
    }

    pub fn with_frontier_type(mut self, frontier_type: FrontierType) -> Self {
        self.frontier_type = frontier_type;
        self
    }

    pub fn with_acceptance(mut self, acceptance: AcceptanceCriterion) -> Self {
        self.acceptance = acceptance;
        self
    }

    pub fn with_bias(mut self, bias: ResearchBias) -> Self {
        self.bias = bias;
        self
    }

    fn record_rejection(&mut self, reason: impl Into<String>) {
        self.rejections.push(RejectionRecord {
            iteration: self.snapshot.iteration,
            reason: reason.into(),
        });
    }

    fn candidate_of(&self, id: &str) -> Candidate {
        self.candidates
            .get(id)
            .cloned()
            .unwrap_or_else(|| self.candidates["root"].clone())
    }

    /// Select the parent to mutate (mode-aware) and report the chosen mode.
    ///
    /// Samples a mode from [`ResearchBias`] (explore / exploit / validate),
    /// then selects a parent from the Pareto non-dominated set accordingly:
    /// - **Explore**: softmax-sample a diverse specialist (lower win-count
    ///   preferred) to map new surface.
    /// - **Exploit**: argmax aggregate score — refine the strongest candidate.
    /// - **Validate**: the strictest-gated node (most inherited gates), so
    ///   gate-gated acceptance targets the most-constrained lineage.
    ///
    /// When nothing is evaluated yet, the seed (under the root id) is returned
    /// in `Explore` mode.
    pub fn select_parent_mode_aware(&self, seed: u64) -> (ExperimentId, Candidate, ResearchMode) {
        let pool = self.graph.evaluated_pool();
        if pool.is_empty() {
            // Sample the mode from the bias even before anything is evaluated,
            // so a validation-only bias enforces gates on the very first proposal.
            let mode = sample_mode(&self.bias, seed, "parent_mode", self.snapshot.iteration);
            return ("root".to_string(), self.candidates["root"].clone(), mode);
        }

        let mode = sample_mode(&self.bias, seed, "parent_mode", self.snapshot.iteration);
        let nd = non_dominated_set(&pool, self.frontier_type, &self.objectives);
        let selectable: Vec<&ExperimentNode> = if nd.is_empty() { pool.clone() } else { nd };
        let front = pareto_front(&pool, self.frontier_type, &self.objectives);

        let chosen = match mode {
            ResearchMode::Exploit => argmax_score(&selectable, &self.metric),
            ResearchMode::Explore => {
                explore_sample(&selectable, &front, seed, self.snapshot.iteration)
            }
            ResearchMode::Validate => strictest_gated(&selectable, &self.metric, |id| {
                self.graph.effective_gates(id).map(|g| g.len()).unwrap_or(0)
            }),
        };
        let chosen = chosen.unwrap_or_else(|| argmax_score(&selectable, &self.metric).unwrap());
        (chosen.id.clone(), self.candidate_of(&chosen.id), mode)
    }

    /// Current-best (exploit) parent selection — kept for callers that want
    /// deterministic argmax without the mode-aware sampling.
    pub fn select_parent(&self) -> (ExperimentId, Candidate) {
        let pool = self.graph.evaluated_pool();
        match argmax_score(&pool, &self.metric) {
            Some(node) => (node.id.clone(), self.candidate_of(&node.id)),
            None => ("root".to_string(), self.candidates["root"].clone()),
        }
    }
}

/// Run the optimization loop until `stop` fires.
///
/// Each iteration: select a parent (mode-aware), sample a minibatch, evaluate
/// the parent (capturing traces), build a reflective dataset, propose a child
/// with the rendered ledger dossier as reflection history, evaluate the child
/// on the minibatch, apply the acceptance criterion, and — in `Validate` mode —
/// enforce inherited gates via `gate_runner` (gate-gated acceptance). If
/// accepted, run a full validation evaluation and commit the child to the
/// graph (which updates the Pareto frontier via the evaluated pool) and record
/// it in the ledger.
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn optimize<A, P>(
    adapter: &A,
    proposer: &P,
    state: &mut OptimizationState,
    stop: &StopCondition,
    cache: &mut EvaluationCache,
    gate_runner: Option<&dyn GateRunner>,
    trainset: &[serde_json::Value],
    valset: &[serde_json::Value],
    minibatch_size: usize,
    seed: u64,
) -> Result<()>
where
    A: ResearchAdapter,
    P: Proposer,
{
    if valset.is_empty() {
        return Ok(());
    }

    while !stop.should_stop(&state.snapshot) {
        let (parent_id, parent_candidate, mode) = state.select_parent_mode_aware(seed);
        let components: Vec<String> = parent_candidate.keys().cloned().collect();

        let (minibatch, minibatch_ids) =
            sample_minibatch(trainset, minibatch_size, state.snapshot.iteration);

        // Evaluate the parent on the minibatch, capturing traces for reflection.
        let before = cached_evaluate(
            adapter,
            cache,
            &minibatch,
            &minibatch_ids,
            &parent_candidate,
            true,
        )?;
        state.snapshot.metric_calls += before.num_metric_calls;

        // Render the ledger dossier as reflection history (the crate's unique lever:
        // reflection reads the whole history, not just the last minibatch).
        let history = state.ledger.render_dossier(REFLECTION_HISTORY_ITEMS);

        // Build the reflective dataset and propose a child.
        let reflective =
            adapter.make_reflective_dataset(&parent_candidate, &minibatch, &before, &components)?;
        let child =
            proposer.propose(&parent_candidate, &reflective, &components, Some(&history))?;

        // Evaluate the child on the same minibatch (no traces needed).
        let after = cached_evaluate(adapter, cache, &minibatch, &minibatch_ids, &child, false)?;
        state.snapshot.metric_calls += after.num_metric_calls;

        // Acceptance gate on the minibatch (GEPA sums subsample scores).
        let before_sum: f64 = before.scores.iter().sum();
        let after_sum: f64 = after.scores.iter().sum();
        let verdict = state
            .acceptance
            .should_accept(after_sum, Some(before_sum), &state.metric);
        if !verdict.accepted {
            state.record_rejection(verdict.reason);
            state.snapshot.iteration += 1;
            state.snapshot.iterations_since_improvement += 1;
            continue;
        }

        // Gate-gated acceptance in Validate mode (the crate's unique lever; GEPA
        // has no gate concept). The proposal must pass inherited gates IN
        // ADDITION to improving the score.
        if mode == ResearchMode::Validate {
            if let Some(runner) = gate_runner {
                let gates = state.graph.effective_gates(&parent_id).unwrap_or_default();
                let mut gate_failed = None;
                for gate in &gates {
                    let outcome = runner.run_gate(gate, &child)?;
                    if !outcome.passed {
                        gate_failed = Some(format!(
                            "gate '{}' failed: {}",
                            gate.name,
                            if outcome.output_snippet.is_empty() {
                                "no output"
                            } else {
                                outcome.output_snippet.as_str()
                            }
                        ));
                        break;
                    }
                }
                if let Some(reason) = gate_failed {
                    state.record_rejection(reason);
                    state.snapshot.iteration += 1;
                    state.snapshot.iterations_since_improvement += 1;
                    continue;
                }
            }
        }

        // Full validation evaluation of the accepted child.
        let val_ids: Vec<String> = (0..valset.len()).map(|i| format!("val_{i}")).collect();
        let full = cached_evaluate(adapter, cache, valset, &val_ids, &child, false)?;
        state.snapshot.metric_calls += full.num_metric_calls;

        // Commit the child to the graph (updates the Pareto frontier via the
        // evaluated pool).
        let hypothesis = Hypothesis::new(format!(
            "proposed iter {} ({:?})",
            state.snapshot.iteration + 1,
            mode
        ))
        .with_mode(mode);
        let node_id = state.graph.allocate_child(&parent_id, hypothesis)?;
        state.candidates.insert(node_id.clone(), child);
        let outcome = build_outcome(&full, &val_ids);
        let child_score = outcome.score;
        state.graph.record_outcome(&node_id, outcome)?;
        state
            .graph
            .commit(&node_id, format!("iter {}", state.snapshot.iteration + 1))?;

        // Record the accepted candidate in the ledger so future reflection
        // iterations can reuse it.
        state.ledger.record_observation(
            &state.metric.name,
            "loop",
            format!("accepted {node_id} score={child_score:.4} mode={mode:?}"),
            format!("parent={parent_id}"),
            EvidenceConfidence::High,
        );

        // Track the best score for the NoImprovement stop condition.
        if state
            .best_score
            .is_none_or(|best| state.metric.direction.is_better(child_score, best))
        {
            state.best_score = Some(child_score);
            state.snapshot.iterations_since_improvement = 0;
        } else {
            state.snapshot.iterations_since_improvement += 1;
        }
        state.snapshot.iteration += 1;
    }
    Ok(())
}

/// Sample a research mode from the bias weights, deterministically per
/// (seed, iteration).
fn sample_mode(bias: &ResearchBias, seed: u64, label: &str, iteration: u32) -> ResearchMode {
    let total = bias.explore_weight + bias.exploit_weight + bias.validation_weight;
    if total <= 0.0 {
        return ResearchMode::Exploit;
    }
    let label = format!("{label}_{iteration}");
    let r = stable_unit(seed, &label) * total;
    if r < bias.explore_weight {
        ResearchMode::Explore
    } else if r < bias.explore_weight + bias.exploit_weight {
        ResearchMode::Exploit
    } else {
        ResearchMode::Validate
    }
}

/// argmax by directional aggregate score, tie-break by id for determinism.
fn argmax_score<'a>(pool: &'a [&'a ExperimentNode], metric: &Metric) -> Option<&'a ExperimentNode> {
    pool.iter()
        .max_by(|a, b| {
            let a_score = metric
                .direction
                .directional_score(a.score.unwrap_or(f64::NEG_INFINITY));
            let b_score = metric
                .direction
                .directional_score(b.score.unwrap_or(f64::NEG_INFINITY));
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.id.cmp(&a.id))
        })
        .copied()
}

/// Explore selection: softmax-sample a diverse specialist. Lower win-count
/// (fewer frontier keys won) gets higher weight, so the loop maps new surface
/// instead of re-refining the incumbent.
fn explore_sample<'a>(
    pool: &'a [&'a ExperimentNode],
    front: &crate::pareto::ParetoFront,
    seed: u64,
    iteration: u32,
) -> Option<&'a ExperimentNode> {
    if pool.is_empty() {
        return None;
    }
    let max_wins = pool
        .iter()
        .map(|n| front.win_count(&n.id))
        .max()
        .unwrap_or(0)
        .max(1) as f64;
    // Weight = (max_wins - wins + 1): specialists (low wins) weigh more.
    let weights: Vec<f64> = pool
        .iter()
        .map(|n| (max_wins - front.win_count(&n.id) as f64 + 1.0).max(0.001))
        .collect();
    let total: f64 = weights.iter().sum();
    let r = stable_unit(seed, &format!("explore_{iteration}")) * total;
    let mut acc = 0.0;
    for (node, w) in pool.iter().zip(weights.iter()) {
        acc += *w;
        if r <= acc {
            return Some(node);
        }
    }
    pool.last().copied()
}

/// Validate selection: the strictest-gated node (most inherited gates),
/// tie-broken by score. Gates come from the host via the closure so this stays
/// pure over the graph.
fn strictest_gated<'a, F>(
    pool: &'a [&'a ExperimentNode],
    metric: &Metric,
    gate_count: F,
) -> Option<&'a ExperimentNode>
where
    F: Fn(&str) -> usize,
{
    pool.iter()
        .max_by(|a, b| {
            let ag = gate_count(&a.id);
            let bg = gate_count(&b.id);
            let a_score = metric
                .direction
                .directional_score(a.score.unwrap_or(f64::NEG_INFINITY));
            let b_score = metric
                .direction
                .directional_score(b.score.unwrap_or(f64::NEG_INFINITY));
            ag.cmp(&bg)
                .then_with(|| {
                    a_score
                        .partial_cmp(&b_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.id.cmp(&a.id))
        })
        .copied()
}

/// Build a [`TrialOutcome`] from a full validation batch, populating the
/// per-instance subscores and aggregate objective scores that feed the Pareto
/// frontier.
fn build_outcome(full: &EvaluationBatch, val_ids: &[String]) -> TrialOutcome {
    let score = if full.scores.is_empty() {
        0.0
    } else {
        full.scores.iter().sum::<f64>() / full.scores.len() as f64
    };
    let val_subscores: BTreeMap<String, f64> = val_ids
        .iter()
        .zip(full.scores.iter())
        .map(|(id, &s)| (id.clone(), s))
        .collect();

    let objective_scores: BTreeMap<String, f64> = match &full.objective_scores {
        Some(per_example) => {
            let mut totals: BTreeMap<String, (f64, usize)> = BTreeMap::new();
            for obj_map in per_example {
                for (name, &val) in obj_map {
                    let entry = totals.entry(name.clone()).or_insert((0.0, 0));
                    entry.0 += val;
                    entry.1 += 1;
                }
            }
            totals
                .into_iter()
                .map(|(name, (sum, count))| (name, sum / count.max(1) as f64))
                .collect()
        }
        None => BTreeMap::new(),
    };

    let mut outcome = TrialOutcome::passed(score, format!("full eval mean={score:.4}"));
    outcome.status = if score.is_finite() {
        OutcomeStatus::Passed
    } else {
        OutcomeStatus::Inconclusive
    };
    outcome.val_subscores = val_subscores;
    outcome.objective_scores = objective_scores;
    outcome
}

/// Sample a minibatch from the trainset, deterministically rotating by
/// iteration. Returns the inputs and their stable trainset-index ids (so the
/// cache keys on the actual example, not the position).
fn sample_minibatch(
    trainset: &[serde_json::Value],
    size: usize,
    iteration: u32,
) -> (Vec<serde_json::Value>, Vec<String>) {
    if trainset.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let size = size.max(1).min(trainset.len());
    let start = (iteration as usize * size) % trainset.len();
    let mut inputs = Vec::with_capacity(size);
    let mut ids = Vec::with_capacity(size);
    for i in 0..size {
        let idx = (start + i) % trainset.len();
        inputs.push(trainset[idx].clone());
        ids.push(format!("train_{idx}"));
    }
    (inputs, ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ResearchAdapter;
    use crate::policy::{Gate, GateOutcome, GatePhase};
    use crate::proposer::Proposer;
    use crate::stop::StopCondition;

    /// Adapter that scores a candidate by parsing its "score" component.
    struct ScoreAdapter;
    impl ResearchAdapter for ScoreAdapter {
        fn evaluate(
            &self,
            batch: &[serde_json::Value],
            candidate: &Candidate,
            _capture_traces: bool,
        ) -> Result<EvaluationBatch> {
            let base: f64 = candidate
                .get("score")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            Ok(EvaluationBatch {
                scores: batch.iter().map(|_| base).collect(),
                outputs: batch.to_vec(),
                trajectories: None,
                objective_scores: None,
                num_metric_calls: batch.len() as u32,
            })
        }
    }

    /// Proposer that increments the parent's score by 0.1 and bumps the text
    /// version, so every proposal strictly improves.
    struct IncrementingProposer;
    impl Proposer for IncrementingProposer {
        fn propose(
            &self,
            parent: &Candidate,
            _dataset: &crate::adapter::ReflectiveDataset,
            _components: &[String],
            _history: Option<&str>,
        ) -> Result<Candidate> {
            let mut child = parent.clone();
            let parent_score: f64 = parent
                .get("score")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            if let Some(score) = child.get_mut("score") {
                *score = format!("{:.1}", parent_score + 0.1);
            }
            let text = parent.get("text").cloned().unwrap_or_else(|| "v0".into());
            let version: u32 = text.trim_start_matches('v').parse().unwrap_or(0);
            if let Some(text) = child.get_mut("text") {
                *text = format!("v{}", version + 1);
            }
            Ok(child)
        }
    }

    /// An exploit-only bias so the Phase 3 reproduction stays deterministic
    /// (parent = current best → child = parent + 0.1).
    fn exploit_bias() -> ResearchBias {
        ResearchBias {
            explore_weight: 0.0,
            exploit_weight: 1.0,
            validation_weight: 0.0,
            require_in_scope: false,
        }
    }

    #[test]
    fn loop_accepts_strict_improvements_and_stops_on_budget() {
        // Reproduction for Phase 3: a mock adapter/proposer drives a 3-iteration
        // loop, each proposal strictly improves and is accepted, and the loop
        // stops on MaxMetricCalls. Reverting the acceptance gate (accept
        // everything) or the stop check (never stop) breaks the assertions.
        let adapter = ScoreAdapter;
        let proposer = IncrementingProposer;
        let metric = Metric::maximize("accuracy");
        let seed: Candidate = [
            ("text".to_string(), "v0".to_string()),
            ("score".to_string(), "0.0".to_string()),
        ]
        .into_iter()
        .collect();
        let mut state = OptimizationState::new(metric, seed).with_bias(exploit_bias());

        // 4 metric calls per iteration (1 before + 1 after + 2 full eval),
        // so 3 iterations consume 12 calls; stop fires at the 4th iteration.
        let stop = StopCondition::MaxMetricCalls { max: 12 };
        let mut cache = EvaluationCache::new();
        let trainset = vec![
            serde_json::json!("q0"),
            serde_json::json!("q1"),
            serde_json::json!("q2"),
        ];
        let valset = vec![serde_json::json!("v0"), serde_json::json!("v1")];

        optimize(
            &adapter, &proposer, &mut state, &stop, &mut cache, None, &trainset, &valset, 1, 0,
        )
        .unwrap();

        // Exactly 3 accepted iterations, strictly improving scores.
        let committed: Vec<f64> = state
            .graph
            .nodes
            .values()
            .filter(|n| n.commit.is_some())
            .filter_map(|n| n.score)
            .collect();
        assert_eq!(committed.len(), 3, "exactly 3 candidates committed");
        assert_eq!(committed, vec![0.1, 0.2, 0.3]);
        assert_eq!(state.snapshot.iteration, 3);
        assert!(state.snapshot.metric_calls <= 12);
        assert_eq!(state.best_score, Some(0.3));
        // The ledger absorbed the accepted candidates as observations.
        assert!(!state.ledger.observations.is_empty());
    }

    #[test]
    fn loop_rejects_non_improving_proposal() {
        // A proposer that does not change the score: strict improvement rejects
        // every proposal, so nothing is committed and the loop stops on
        // MaxIterations with zero commits.
        struct IdentityProposer;
        impl Proposer for IdentityProposer {
            fn propose(
                &self,
                parent: &Candidate,
                _dataset: &crate::adapter::ReflectiveDataset,
                _components: &[String],
                _history: Option<&str>,
            ) -> Result<Candidate> {
                Ok(parent.clone())
            }
        }
        let adapter = ScoreAdapter;
        let proposer = IdentityProposer;
        let metric = Metric::maximize("accuracy");
        let seed: Candidate = [
            ("text".to_string(), "v0".to_string()),
            ("score".to_string(), "0.5".to_string()),
        ]
        .into_iter()
        .collect();
        let mut state = OptimizationState::new(metric, seed).with_bias(exploit_bias());
        let stop = StopCondition::MaxIterations { max: 5 };
        let mut cache = EvaluationCache::new();
        let trainset = vec![serde_json::json!("q0")];
        let valset = vec![serde_json::json!("v0")];

        optimize(
            &adapter, &proposer, &mut state, &stop, &mut cache, None, &trainset, &valset, 1, 0,
        )
        .unwrap();

        let committed = state
            .graph
            .nodes
            .values()
            .filter(|n| n.commit.is_some())
            .count();
        assert_eq!(committed, 0, "non-improving proposals must be rejected");
        assert_eq!(state.snapshot.iteration, 5);
        // Rejections were recorded with reasons.
        assert!(!state.rejections.is_empty());
        assert!(state.rejections[0].reason.contains("strict improvement"));
    }

    /// A gate runner that always fails, simulating a failing `cargo test`.
    struct FailingGateRunner;
    impl GateRunner for FailingGateRunner {
        fn run_gate(&self, gate: &Gate, _candidate: &Candidate) -> Result<GateOutcome> {
            Ok(GateOutcome {
                name: gate.name.clone(),
                phase: gate.phase,
                passed: false,
                exit_code: Some(1),
                output_snippet: "cargo test failed: 2 tests failed".to_string(),
            })
        }
    }

    #[test]
    fn validate_rejects_score_improving_proposal_that_fails_a_gate() {
        // Reproduction for Phase 4: a Validate-mode proposal that improves the
        // score but fails an inherited gate is rejected with a gate reason.
        // Reverting the gate-gated acceptance (Validate = score-only) makes
        // this test fail — the candidate would be committed.
        let adapter = ScoreAdapter;
        let proposer = IncrementingProposer;
        let metric = Metric::maximize("accuracy");
        let seed: Candidate = [
            ("text".to_string(), "v0".to_string()),
            ("score".to_string(), "0.0".to_string()),
        ]
        .into_iter()
        .collect();
        let mut state = OptimizationState::new(metric, seed).with_bias(ResearchBias {
            explore_weight: 0.0,
            exploit_weight: 0.0,
            validation_weight: 1.0,
            require_in_scope: false,
        });
        // Attach an inherited gate to the root so Validate runs it.
        state
            .graph
            .add_gate("root", Gate::new("test", "cargo test"))
            .unwrap();

        let stop = StopCondition::MaxIterations { max: 3 };
        let mut cache = EvaluationCache::new();
        let runner = FailingGateRunner;
        let trainset = vec![serde_json::json!("q0")];
        let valset = vec![serde_json::json!("v0")];

        optimize(
            &adapter,
            &proposer,
            &mut state,
            &stop,
            &mut cache,
            Some(&runner),
            &trainset,
            &valset,
            1,
            0,
        )
        .unwrap();

        let committed = state
            .graph
            .nodes
            .values()
            .filter(|n| n.commit.is_some())
            .count();
        assert_eq!(
            committed, 0,
            "score-improving but gate-failing proposals must be rejected in Validate mode"
        );
        // The rejection reason must cite the gate, not the score.
        let gate_rejection = state
            .rejections
            .iter()
            .find(|r| r.reason.contains("gate 'test' failed"));
        assert!(
            gate_rejection.is_some(),
            "expected a gate-failure rejection, got: {:?}",
            state.rejections
        );
    }

    #[test]
    fn validate_accepts_when_gate_passes() {
        // Counterpart: when the gate passes, the score-improving Validate
        // proposal IS committed (gate-gated acceptance is additive, not a
        // blanket block).
        struct PassingGateRunner;
        impl GateRunner for PassingGateRunner {
            fn run_gate(&self, gate: &Gate, _candidate: &Candidate) -> Result<GateOutcome> {
                Ok(GateOutcome {
                    name: gate.name.clone(),
                    phase: GatePhase::Post,
                    passed: true,
                    exit_code: Some(0),
                    output_snippet: String::new(),
                })
            }
        }
        let adapter = ScoreAdapter;
        let proposer = IncrementingProposer;
        let metric = Metric::maximize("accuracy");
        let seed: Candidate = [
            ("text".to_string(), "v0".to_string()),
            ("score".to_string(), "0.0".to_string()),
        ]
        .into_iter()
        .collect();
        let mut state = OptimizationState::new(metric, seed).with_bias(ResearchBias {
            explore_weight: 0.0,
            exploit_weight: 0.0,
            validation_weight: 1.0,
            require_in_scope: false,
        });
        state
            .graph
            .add_gate("root", Gate::new("test", "cargo test"))
            .unwrap();

        let stop = StopCondition::MaxIterations { max: 1 };
        let mut cache = EvaluationCache::new();
        let runner = PassingGateRunner;
        let trainset = vec![serde_json::json!("q0")];
        let valset = vec![serde_json::json!("v0")];

        optimize(
            &adapter,
            &proposer,
            &mut state,
            &stop,
            &mut cache,
            Some(&runner),
            &trainset,
            &valset,
            1,
            0,
        )
        .unwrap();

        let committed = state
            .graph
            .nodes
            .values()
            .filter(|n| n.commit.is_some())
            .count();
        assert_eq!(
            committed, 1,
            "passing gate + score improvement must be accepted"
        );
    }
}
