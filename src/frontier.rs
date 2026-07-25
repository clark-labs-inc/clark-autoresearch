use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph::{ExperimentGraph, ExperimentId, ResearchMode};
use crate::ids::stable_unit;
use crate::outcome::{Metric, MetricDirection, Objectives};
use crate::pareto::{FrontierType, non_dominated_set, pareto_front};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrontierStrategy {
    ArgMax,
    TopK {
        k: usize,
    },
    EpsilonGreedy {
        epsilon: f64,
        seed: u64,
    },
    Softmax {
        temperature: f64,
        k: usize,
        seed: u64,
    },
    ParetoPerTask {
        k: usize,
        task_floor: f64,
    },
    /// Pareto-efficient selection over the evaluated pool.
    ///
    /// Unlike `ParetoPerTask` (a per-task win-count heuristic over frontier
    /// tips only), this computes the true non-dominated set over every
    /// evaluated node — including discarded specialists — mirroring GEPA's
    /// `_get_pareto_front_mapping`. Candidates are ranked by the number of
    /// frontier keys they win, then by aggregate score.
    Pareto {
        frontier_type: FrontierType,
        #[serde(default)]
        objectives: Vec<Metric>,
    },
}

impl Default for FrontierStrategy {
    fn default() -> Self {
        Self::TopK { k: 5 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub id: ExperimentId,
    pub rank: usize,
    pub score: Option<f64>,
    pub mode: ResearchMode,
    pub reason: String,
    /// When acceptance is enforced, a candidate that does not improve its
    /// parent carries the rejection reason here (and `None` when it passes or
    /// acceptance is not applied). Mirrors GEPA's reject-with-reason logging.
    #[serde(default)]
    pub reject_reason: Option<String>,
}

pub fn rank_frontier(
    graph: &ExperimentGraph,
    metric: &Metric,
    strategy: &FrontierStrategy,
) -> Vec<RankedCandidate> {
    let nodes = graph.frontier_nodes();
    if nodes.is_empty() {
        return Vec::new();
    }

    match strategy {
        FrontierStrategy::ArgMax => top_k(nodes, metric, 1, "argmax"),
        FrontierStrategy::TopK { k } => top_k(nodes, metric, *k, "top_k"),
        FrontierStrategy::EpsilonGreedy { epsilon, seed } => {
            let epsilon = epsilon.clamp(0.0, 1.0);
            if stable_unit(*seed, "epsilon_greedy") < epsilon {
                let chosen = nodes
                    .into_iter()
                    .max_by(|a, b| {
                        stable_unit(*seed, &a.id)
                            .partial_cmp(&stable_unit(*seed, &b.id))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                vec![candidate(chosen, 1, "epsilon_greedy_explore")]
            } else {
                top_k(nodes, metric, 1, "epsilon_greedy_exploit")
            }
        }
        FrontierStrategy::Softmax {
            temperature,
            k,
            seed,
        } => softmax(nodes, metric, *temperature, *k, *seed),
        FrontierStrategy::ParetoPerTask { k, task_floor } => {
            pareto_per_task(nodes, metric, *k, *task_floor)
        }
        FrontierStrategy::Pareto {
            frontier_type,
            objectives,
        } => pareto_select(graph, metric, *frontier_type, objectives),
    }
}

/// Pareto-efficient ranking over the evaluated pool.
///
/// Builds the objectives (falling back to the single metric when none are
/// supplied), computes the non-dominated set, and ranks it by the number of
/// frontier keys each candidate wins (descending) then by aggregate score.
/// A node that wins one example/objective stays ranked even if its aggregate
/// is lower — the property scalar frontier selection loses.
fn pareto_select(
    graph: &ExperimentGraph,
    metric: &Metric,
    frontier_type: FrontierType,
    objectives: &[Metric],
) -> Vec<RankedCandidate> {
    let pool: Vec<&crate::graph::ExperimentNode> = graph.evaluated_pool();
    if pool.is_empty() {
        return Vec::new();
    }
    let objectives = if objectives.is_empty() {
        Objectives::single(metric.clone())
    } else {
        Objectives::new(objectives.to_vec())
    };
    let front = pareto_front(&pool, frontier_type, &objectives);
    let nd = non_dominated_set(&pool, frontier_type, &objectives);
    if nd.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<&crate::graph::ExperimentNode> = nd;
    ranked.sort_by(|a, b| {
        front
            .win_count(&b.id)
            .cmp(&front.win_count(&a.id))
            .then_with(|| compare_nodes(a, b, metric))
    });
    ranked
        .into_iter()
        .enumerate()
        .map(|(idx, node)| {
            let wins = front.win_count(&node.id);
            let mut c = candidate(node, idx + 1, "pareto");
            c.reason = format!("pareto:{}:wins={wins}", frontier_label(frontier_type));
            c
        })
        .collect()
}

fn frontier_label(frontier_type: FrontierType) -> &'static str {
    match frontier_type {
        FrontierType::Instance => "instance",
        FrontierType::Objective => "objective",
        FrontierType::Hybrid => "hybrid",
        FrontierType::Cartesian => "cartesian",
    }
}

fn top_k(
    mut nodes: Vec<&crate::graph::ExperimentNode>,
    metric: &Metric,
    k: usize,
    reason: &str,
) -> Vec<RankedCandidate> {
    nodes.sort_by(|a, b| compare_nodes(a, b, metric));
    nodes
        .into_iter()
        .take(k.max(1))
        .enumerate()
        .map(|(idx, node)| candidate(node, idx + 1, reason))
        .collect()
}

fn softmax(
    nodes: Vec<&crate::graph::ExperimentNode>,
    metric: &Metric,
    temperature: f64,
    k: usize,
    seed: u64,
) -> Vec<RankedCandidate> {
    let temperature = temperature.clamp(0.001, 100.0);
    let mut scored = nodes
        .into_iter()
        .map(|node| {
            let score = directional_or_zero(node.score, metric.direction);
            (node, score)
        })
        .collect::<Vec<_>>();
    let max_score = scored
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);

    scored.sort_by(|(a, a_score), (b, b_score)| {
        let a_weight = ((*a_score - max_score) / temperature)
            .exp()
            .max(f64::MIN_POSITIVE);
        let b_weight = ((*b_score - max_score) / temperature)
            .exp()
            .max(f64::MIN_POSITIVE);
        let a_key = -stable_unit(seed, &a.id).ln() / a_weight;
        let b_key = -stable_unit(seed, &b.id).ln() / b_weight;
        a_key
            .partial_cmp(&b_key)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    scored
        .into_iter()
        .take(k.max(1))
        .enumerate()
        .map(|(idx, (node, _))| candidate(node, idx + 1, "softmax"))
        .collect()
}

fn pareto_per_task(
    nodes: Vec<&crate::graph::ExperimentNode>,
    metric: &Metric,
    k: usize,
    task_floor: f64,
) -> Vec<RankedCandidate> {
    let mut task_ids: Option<BTreeSet<String>> = None;
    let mut task_scores: BTreeMap<String, BTreeMap<String, (f64, MetricDirection)>> =
        BTreeMap::new();

    for node in &nodes {
        let Some(outcome) = &node.outcome else {
            continue;
        };
        let ids = outcome
            .task_scores
            .iter()
            .map(|score| score.task_id.clone())
            .collect::<BTreeSet<_>>();
        if ids.is_empty() {
            continue;
        }
        task_ids = Some(match task_ids {
            Some(existing) => existing.intersection(&ids).cloned().collect(),
            None => ids,
        });
        let by_task = task_scores.entry(node.id.clone()).or_default();
        for score in &outcome.task_scores {
            by_task.insert(
                score.task_id.clone(),
                (score.score, score.direction.unwrap_or(metric.direction)),
            );
        }
    }

    let Some(common_tasks) = task_ids else {
        return top_k(nodes, metric, k, "pareto_fallback_top_k");
    };
    if common_tasks.is_empty() {
        return top_k(nodes, metric, k, "pareto_fallback_top_k");
    }

    let mut win_counts: BTreeMap<String, usize> = BTreeMap::new();
    for task_id in common_tasks {
        let mut best: Option<f64> = None;
        let mut winners = Vec::new();
        for node in &nodes {
            let Some((score, direction)) = task_scores
                .get(&node.id)
                .and_then(|scores| scores.get(&task_id))
                .copied()
            else {
                continue;
            };
            if matches!(direction, MetricDirection::Maximize) && score <= task_floor {
                continue;
            }
            let directional = direction.directional_score(score);
            if best.is_none_or(|incumbent| directional > incumbent) {
                best = Some(directional);
                winners.clear();
                winners.push(node.id.clone());
            } else if best == Some(directional) {
                winners.push(node.id.clone());
            }
        }
        for winner in winners {
            *win_counts.entry(winner).or_default() += 1;
        }
    }

    if win_counts.is_empty() {
        return top_k(nodes, metric, k, "pareto_fallback_top_k");
    }

    let mut pareto_nodes = nodes
        .into_iter()
        .filter(|node| win_counts.contains_key(&node.id))
        .collect::<Vec<_>>();
    pareto_nodes.sort_by(|a, b| {
        win_counts
            .get(&b.id)
            .cmp(&win_counts.get(&a.id))
            .then_with(|| compare_nodes(a, b, metric))
    });

    pareto_nodes
        .into_iter()
        .take(k.max(1))
        .enumerate()
        .map(|(idx, node)| {
            let wins = win_counts.get(&node.id).copied().unwrap_or_default();
            let mut c = candidate(node, idx + 1, "pareto_per_task");
            c.reason = format!("pareto_per_task:wins={wins}");
            c
        })
        .collect()
}

fn compare_nodes(
    a: &crate::graph::ExperimentNode,
    b: &crate::graph::ExperimentNode,
    metric: &Metric,
) -> std::cmp::Ordering {
    let a_score = directional_or_neg_inf(a.score, metric.direction);
    let b_score = directional_or_neg_inf(b.score, metric.direction);
    b_score
        .partial_cmp(&a_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.id.cmp(&b.id))
}

fn directional_or_neg_inf(score: Option<f64>, direction: MetricDirection) -> f64 {
    score
        .map(|score| direction.directional_score(score))
        .unwrap_or(f64::NEG_INFINITY)
}

fn directional_or_zero(score: Option<f64>, direction: MetricDirection) -> f64 {
    score
        .map(|score| direction.directional_score(score))
        .unwrap_or(0.0)
}

fn candidate(
    node: &crate::graph::ExperimentNode,
    rank: usize,
    reason: impl Into<String>,
) -> RankedCandidate {
    RankedCandidate {
        id: node.id.clone(),
        rank,
        score: node.score,
        mode: node.mode(),
        reason: reason.into(),
        reject_reason: None,
    }
}

/// Enforce an [`AcceptanceCriterion`] over a ranked frontier in place.
///
/// For each ranked candidate that has a scored parent, this judges whether the
/// candidate improved on its parent and records the verdict in
/// [`RankedCandidate::reject_reason`]. Candidates that pass (or have no scored
/// parent) keep `reject_reason = None`; rejected ones carry the reason. This
/// turns the previously-unused [`ExperimentGraph::outcome_improves_parent`]
/// into an enforced, explainable gate reachable from the frontier path.
///
/// Rejected candidates are moved to the end of the list (preserving their
/// relative order) so accepted improvements stay at the top. Nothing is
/// removed: the caller sees both what was accepted and what was rejected and
/// why, exactly like GEPA's reject-with-reason logging.
pub fn enforce_acceptance(
    graph: &ExperimentGraph,
    metric: &Metric,
    criterion: crate::acceptance::AcceptanceCriterion,
    ranked: &mut [RankedCandidate],
) {
    for candidate in ranked.iter_mut() {
        let Ok(verdict) = graph.acceptance_verdict(&candidate.id, metric, criterion) else {
            candidate.reject_reason = None;
            continue;
        };
        candidate.reject_reason = if verdict.accepted {
            None
        } else {
            Some(verdict.reason)
        };
    }
    // Stable partition: accepted (reject_reason None) first, rejected last.
    ranked.sort_by_key(|candidate| candidate.reject_reason.is_some());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ExperimentGraph, Hypothesis};
    use crate::outcome::{Metric, MetricDirection, TaskScore, TrialOutcome};
    use crate::pareto::FrontierType;

    #[test]
    fn pareto_strategy_ranks_discarded_specialist() {
        // Reproduction for Phase 1: a discarded node that wins one example
        // must appear in the Pareto frontier ranking even though its aggregate
        // is lower and it is not a frontier tip. Reverting to scalar/ tip-only
        // selection drops it, failing this test.
        let metric = Metric::maximize("accuracy");
        let mut graph = ExperimentGraph::new("run");

        let generalist = graph
            .allocate_child("root", Hypothesis::new("generalist"))
            .unwrap();
        graph
            .record_outcome(
                &generalist,
                TrialOutcome::passed(0.8, "").with_val_subscores(
                    [("ex_0".to_string(), 0.6), ("ex_1".to_string(), 0.6)]
                        .into_iter()
                        .collect(),
                ),
            )
            .unwrap();
        graph.commit(&generalist, "g").unwrap();

        let specialist = graph
            .allocate_child("root", Hypothesis::new("specialist"))
            .unwrap();
        graph
            .record_outcome(
                &specialist,
                TrialOutcome::passed(0.7, "").with_val_subscores(
                    [("ex_0".to_string(), 0.4), ("ex_1".to_string(), 1.0)]
                        .into_iter()
                        .collect(),
                ),
            )
            .unwrap();
        graph.discard(&specialist, "lower aggregate").unwrap();

        let ranked = rank_frontier(
            &graph,
            &metric,
            &FrontierStrategy::Pareto {
                frontier_type: FrontierType::Instance,
                objectives: Vec::new(),
            },
        );
        let ids: Vec<&str> = ranked.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&specialist.as_str()));
        assert!(ids.contains(&generalist.as_str()));
        // generalist wins ex_0 and ties ex_1? ex_1: generalist 0.6 vs specialist 1.0
        // -> generalist wins ex_0 (0.6 vs 0.4), specialist wins ex_1 (1.0 vs 0.6).
        assert!(
            ranked
                .iter()
                .any(|c| c.reason.starts_with("pareto:instance"))
        );
    }

    #[test]
    fn top_k_respects_minimize_metric() {
        let metric = Metric::minimize("loss");
        let mut graph = ExperimentGraph::new("run");
        let a = graph.allocate_child("root", Hypothesis::new("a")).unwrap();
        graph
            .record_outcome(&a, TrialOutcome::passed(10.0, ""))
            .unwrap();
        graph.commit(&a, "a").unwrap();
        let b = graph.allocate_child("root", Hypothesis::new("b")).unwrap();
        graph
            .record_outcome(&b, TrialOutcome::passed(8.0, ""))
            .unwrap();
        graph.commit(&b, "b").unwrap();

        let ranked = rank_frontier(&graph, &metric, &FrontierStrategy::ArgMax);
        assert_eq!(ranked[0].id, b);
    }

    #[test]
    fn pareto_preserves_task_specialist() {
        let metric = Metric::maximize("aggregate");
        let mut graph = ExperimentGraph::new("run");
        let a = graph
            .allocate_child("root", Hypothesis::new("aggregate"))
            .unwrap();
        graph
            .record_outcome(
                &a,
                TrialOutcome::passed(0.8, "").with_task_scores(vec![
                    TaskScore::with_direction("task_a", 1.0, MetricDirection::Maximize),
                    TaskScore::with_direction("task_b", 0.2, MetricDirection::Maximize),
                ]),
            )
            .unwrap();
        graph.commit(&a, "a").unwrap();
        let b = graph
            .allocate_child("root", Hypothesis::new("specialist"))
            .unwrap();
        graph
            .record_outcome(
                &b,
                TrialOutcome::passed(0.7, "").with_task_scores(vec![
                    TaskScore::with_direction("task_a", 0.5, MetricDirection::Maximize),
                    TaskScore::with_direction("task_b", 1.0, MetricDirection::Maximize),
                ]),
            )
            .unwrap();
        graph.commit(&b, "b").unwrap();

        let ranked = rank_frontier(
            &graph,
            &metric,
            &FrontierStrategy::ParetoPerTask {
                k: 2,
                task_floor: 0.0,
            },
        );
        let ids = ranked.into_iter().map(|c| c.id).collect::<Vec<_>>();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
    }
}
