//! A complete execution-agnostic optimization loop with a mock adapter/proposer.
//!
//! This mirrors GEPA's loop shape (propose → minibatch-eval → accept →
//! full-eval → Pareto-update) but every provider call stays behind the
//! [`ResearchAdapter`] and [`Proposer`] traits. Here both are fakes: the
//! adapter scores a candidate by parsing its "score" component, and the
//! proposer increments it. Swap them for a real HTTP-eval adapter and an
//! LM-backed `ReflectiveMutation` to optimize a real system.

use std::collections::BTreeMap;

use anyhow::Result;
use clark_autoresearch::{
    Candidate, EvaluationBatch, EvaluationCache, Gate, OptimizationState, Proposer,
    ReflectiveDataset, ResearchAdapter, StopCondition, optimize,
};

/// A fake adapter: scores a candidate by parsing its "score" component.
struct ScoreAdapter;
impl ResearchAdapter for ScoreAdapter {
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
        Ok(EvaluationBatch {
            scores: batch.iter().map(|_| base).collect(),
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

/// A fake proposer: bumps the parent's score by 0.1 and the text version.
struct IncrementingProposer;
impl Proposer for IncrementingProposer {
    fn propose(
        &self,
        parent: &Candidate,
        _dataset: &ReflectiveDataset,
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

fn main() -> Result<()> {
    let metric = clark_autoresearch::Metric::maximize("accuracy");
    let seed: Candidate = BTreeMap::from([
        ("text".to_string(), "v0".to_string()),
        ("score".to_string(), "0.0".to_string()),
    ]);
    let mut state =
        OptimizationState::new(metric, seed).with_bias(clark_autoresearch::ResearchBias {
            explore_weight: 0.0,
            exploit_weight: 1.0,
            validation_weight: 0.0,
            require_in_scope: false,
        });
    state
        .graph
        .add_gate("root", Gate::new("test", "cargo test"))?;

    let adapter = ScoreAdapter;
    let proposer = IncrementingProposer;
    let stop = StopCondition::MaxIterations { max: 5 };
    let mut cache = EvaluationCache::new();

    let trainset = vec![serde_json::json!("q0"), serde_json::json!("q1")];
    let valset = vec![serde_json::json!("v0"), serde_json::json!("v1")];

    optimize(
        &adapter, &proposer, &mut state, &stop, &mut cache, None, &trainset, &valset, 1, 0,
    )?;

    println!("best score: {:?}", state.best_score);
    println!("metric calls: {}", state.snapshot.metric_calls);
    println!("cache entries: {} (hits={})", cache.len(), cache.hits());
    for node in state.graph.nodes.values() {
        if node.commit.is_some() {
            println!(
                "  committed {} score={:?} hypothesis={:?}",
                node.id, node.score, node.hypothesis.statement
            );
        }
    }
    Ok(())
}
