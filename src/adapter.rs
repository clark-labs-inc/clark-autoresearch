//! Execution-agnostic adapter boundary.
//!
//! This is the architectural move that turns clark from a state model into an
//! execution-agnostic optimizer loop, without bundling execution. The host
//! (a Rust agent runtime, a Tauri/Rust desktop, or a thin CLI adapter) owns
//! the provider, sandbox, and evaluator; clark owns the loop body and the
//! state. Provider calls, code editing, sandboxing, and network tools stay
//! behind [`ResearchAdapter`] and never enter clark's core.
//!
//! This mirrors GEPA's `GEPAAdapter` (`gepa.core.adapter`): a single
//! integration point with `evaluate` (run the candidate on a batch and return
//! scores + traces) and `make_reflective_dataset` (turn traces into the small
//! per-component dataset the proposer reads). GEPA bundles the LM and the
//! sandbox inside its engines (`gepa.oa`); clark pushes both back to the host
//! so the same state model stays portable and publishable.

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::policy::{Gate, GateOutcome};

/// A candidate: a mapping from named component to its text.
///
/// GEPA optimizes "any text parameter" — prompts, code snippets, agent
/// architectures, configs — as a `dict[str, str]`. clark keeps the same
/// opaque representation; the adapter and proposer interpret it, clark does
/// not.
pub type Candidate = BTreeMap<String, String>;

/// Per-example multi-objective scores (objective name -> score).
pub type ObjectiveScores = BTreeMap<String, f64>;

/// The result of evaluating a candidate on a batch of examples.
///
/// Mirrors GEPA's `EvaluationBatch`. `scores` are per-example and higher is
/// better (the adapter is responsible for calibration); `trajectories` are
/// opaque to clark and consumed by [`ResearchAdapter::make_reflective_dataset`];
/// `num_metric_calls` feeds the loop's budget tracking.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvaluationBatch {
    /// Per-example scalar scores (higher = better), length == batch length.
    pub scores: Vec<f64>,
    /// Raw per-example outputs, opaque to clark.
    #[serde(default)]
    pub outputs: Vec<serde_json::Value>,
    /// Optional per-example execution trajectories used for reflection.
    /// `None` when `capture_traces` was false.
    #[serde(default)]
    pub trajectories: Option<Vec<serde_json::Value>>,
    /// Optional per-example multi-objective scores.
    #[serde(default)]
    pub objective_scores: Option<Vec<ObjectiveScores>>,
    /// How many metric (evaluation) calls this batch consumed.
    pub num_metric_calls: u32,
}

/// One entry in a reflective dataset: the per-example context the proposer
/// reads to mutate a component's text.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReflectiveEntry {
    #[serde(default)]
    pub input: serde_json::Value,
    #[serde(default)]
    pub output: serde_json::Value,
    pub score: f64,
    #[serde(default)]
    pub trajectory: Option<serde_json::Value>,
}

/// A reflective dataset: per component, the entries the proposer reflects on.
/// Mirrors GEPA's `make_reflective_dataset` output.
pub type ReflectiveDataset = BTreeMap<String, Vec<ReflectiveEntry>>;

/// Host-implemented execution boundary.
///
/// Implementors own the provider, sandbox, and evaluator. clark calls only
/// these methods; it never makes a network call, runs code, or touches a model
/// directly.
pub trait ResearchAdapter {
    /// Evaluate `candidate` on `batch`, returning per-example scores and
    /// optional traces.
    ///
    /// - `capture_traces`: when true, populate `EvaluationBatch.trajectories`
    ///   so [`Self::make_reflective_dataset`] can extract reflection context.
    ///   When false, traces may be `None` to save time.
    ///
    /// Error handling mirrors GEPA's contract: never raise for individual
    /// example failures — return a valid `EvaluationBatch` with per-example
    /// failure scores (e.g. 0.0) and record the error in the trajectory.
    fn evaluate(
        &self,
        batch: &[serde_json::Value],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch>;

    /// Build a reflective dataset from a batch for the components to update.
    ///
    /// The default implementation pairs each example's input/output/score/
    /// trajectory for every requested component — sufficient when the whole
    /// candidate is reflected on as one. Adapters with per-component traces
    /// override this to extract component-specific context.
    fn make_reflective_dataset(
        &self,
        _candidate: &Candidate,
        inputs: &[serde_json::Value],
        batch: &EvaluationBatch,
        components: &[String],
    ) -> Result<ReflectiveDataset> {
        let mut dataset = ReflectiveDataset::new();
        for component in components {
            let entries = batch
                .scores
                .iter()
                .enumerate()
                .map(|(i, &score)| ReflectiveEntry {
                    input: inputs.get(i).cloned().unwrap_or_default(),
                    output: batch.outputs.get(i).cloned().unwrap_or_default(),
                    score,
                    trajectory: batch
                        .trajectories
                        .as_ref()
                        .and_then(|trajectories| trajectories.get(i).cloned()),
                })
                .collect();
            dataset.insert(component.clone(), entries);
        }
        Ok(dataset)
    }
}

/// Host-implemented gate runner.
///
/// clark's gates are command strings (`cargo test`, smoke checks, safety
/// checks) — running them is execution, which stays in the host. The
/// [`OptimizationState`](crate::loop_opt::OptimizationState) `Validate` mode
/// uses this to enforce inherited gates *in addition to* score improvement
/// (gate-gated acceptance), a concept GEPA has no equivalent of. When no gate
/// runner is supplied, `Validate` mode falls back to score-only acceptance.
pub trait GateRunner {
    /// Run `gate` against the materialized `candidate` and return the outcome.
    fn run_gate(&self, gate: &Gate, candidate: &Candidate) -> Result<GateOutcome>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny in-process adapter: scores a candidate by reading its "score"
    /// component. Used by the loop test in `src/loop.rs` and here to verify the
    /// default reflective dataset builder.
    pub(crate) struct ScoreComponentAdapter;

    impl ResearchAdapter for ScoreComponentAdapter {
        fn evaluate(
            &self,
            batch: &[serde_json::Value],
            candidate: &Candidate,
            capture_traces: bool,
        ) -> Result<EvaluationBatch> {
            let base: f64 = candidate
                .get("score")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let scores = batch.iter().map(|_| base).collect::<Vec<_>>();
            Ok(EvaluationBatch {
                scores,
                outputs: batch.to_vec(),
                trajectories: if capture_traces {
                    Some(batch.to_vec())
                } else {
                    None
                },
                objective_scores: None,
                num_metric_calls: batch.len() as u32,
            })
        }
    }

    #[test]
    fn default_reflective_dataset_pairs_every_example() {
        let adapter = ScoreComponentAdapter;
        let candidate: Candidate = [("prompt".to_string(), "be brief".to_string())]
            .into_iter()
            .collect();
        let inputs = vec![serde_json::json!("q1"), serde_json::json!("q2")];
        let batch = adapter.evaluate(&inputs, &candidate, true).unwrap();
        let dataset = adapter
            .make_reflective_dataset(&candidate, &inputs, &batch, &["prompt".to_string()])
            .unwrap();
        let entries = dataset.get("prompt").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].input, serde_json::json!("q1"));
        assert!(entries[0].trajectory.is_some());
    }
}
