//! Pareto frontier over the evaluated pool.
//!
//! This module ports GEPA's per-instance / per-objective Pareto tracking
//! (`gepa.core.state._get_pareto_front_mapping`) into the crate's state model. The
//! key property — and the reason it is more sample-efficient than scalar
//! frontier selection — is that a candidate which wins even one validation
//! example (or one objective) stays on the frontier and remains selectable as
//! a parent, instead of being discarded behind a lower aggregate score.
//!
//! Four frontier types are supported, matching GEPA:
//!
//! - `Instance`: per validation example. A candidate is on the front for an
//!   example if its `val_subscores[example]` is the best (or tied best) among
//!   the pool, under the metric's direction.
//! - `Objective`: per objective. A candidate is on the front for an objective
//!   if its `objective_scores[objective]` is the best (or tied best).
//! - `Hybrid`: the union of `Instance` and `Objective` fronts.
//! - `Cartesian`: per (example, objective) pair, using `objective_subscores`.
//!   Falls back to `Hybrid` when no per-pair data is present.
//!
//! The non-dominated set is the union of every candidate that appears in at
//! least one front entry.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph::ExperimentNode;
use crate::outcome::Objectives;

/// Strategy for tracking which candidates form the Pareto frontier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierType {
    /// Per validation example: keeps candidates that win a single example.
    #[default]
    Instance,
    /// Per objective: keeps candidates that win a single objective.
    Objective,
    /// Combined instance and objective fronts.
    Hybrid,
    /// Per (example, objective) pair.
    Cartesian,
}

/// A key identifying one axis of the Pareto frontier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrontierKey {
    /// A validation example id.
    Instance(String),
    /// An objective name.
    Objective(String),
    /// A (validation example, objective) pair.
    Cartesian { instance: String, objective: String },
}

/// The Pareto frontier: for each frontier key, the node ids that are best (or
/// tied best) on that key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParetoFront {
    pub mapping: BTreeMap<FrontierKey, Vec<String>>,
}

impl ParetoFront {
    /// The flat non-dominated set: every node id that wins at least one
    /// frontier key.
    pub fn non_dominated_ids(&self) -> BTreeSet<String> {
        self.mapping.values().flatten().cloned().collect()
    }

    /// The number of frontier keys a node wins (for ranking parents).
    pub fn win_count(&self, id: &str) -> usize {
        self.mapping
            .values()
            .filter(|winners| winners.iter().any(|w| w == id))
            .count()
    }
}

/// Compute the Pareto frontier over the evaluated pool.
///
/// `pool` is the set of evaluated nodes (use [`ExperimentGraph::evaluated_pool`]).
/// `objectives` supplies per-objective directions; the primary objective's
/// direction is the fallback for any unrecognised objective or instance axis.
pub fn pareto_front<'a>(
    pool: &'a [&'a ExperimentNode],
    frontier_type: FrontierType,
    objectives: &Objectives,
) -> ParetoFront {
    match frontier_type {
        FrontierType::Instance => instance_front(pool, objectives),
        FrontierType::Objective => objective_front(pool, objectives),
        FrontierType::Hybrid => {
            let mut front = instance_front(pool, objectives);
            for (key, winners) in objective_front(pool, objectives).mapping {
                front.mapping.insert(key, winners);
            }
            front
        }
        FrontierType::Cartesian => cartesian_front(pool, objectives),
    }
}

/// The non-dominated set: node ids that win at least one frontier key.
pub fn non_dominated_set<'a>(
    pool: &'a [&'a ExperimentNode],
    frontier_type: FrontierType,
    objectives: &Objectives,
) -> Vec<&'a ExperimentNode> {
    let front = pareto_front(pool, frontier_type, objectives);
    let ids = front.non_dominated_ids();
    pool.iter()
        .filter(|node| ids.contains(&node.id))
        .copied()
        .collect()
}

fn instance_front(pool: &[&ExperimentNode], objectives: &Objectives) -> ParetoFront {
    let direction = objectives.primary().direction;
    // best directional score per example id, then the node ids achieving it.
    let mut best: BTreeMap<String, (f64, Vec<String>)> = BTreeMap::new();
    for node in pool {
        let Some(outcome) = &node.outcome else {
            continue;
        };
        for (example_id, &score) in &outcome.val_subscores {
            let directional = direction.directional_score(score);
            match best.get_mut(example_id) {
                Some((best_score, winners)) => {
                    if directional > *best_score {
                        *best_score = directional;
                        winners.clear();
                        winners.push(node.id.clone());
                    } else if directional == *best_score {
                        winners.push(node.id.clone());
                    }
                }
                None => {
                    best.insert(example_id.clone(), (directional, vec![node.id.clone()]));
                }
            }
        }
    }
    let mapping = best
        .into_iter()
        .map(|(example_id, (_, winners))| (FrontierKey::Instance(example_id), winners))
        .collect();
    ParetoFront { mapping }
}

fn objective_front(pool: &[&ExperimentNode], objectives: &Objectives) -> ParetoFront {
    let mut best: BTreeMap<String, (f64, Vec<String>)> = BTreeMap::new();
    for node in pool {
        let Some(outcome) = &node.outcome else {
            continue;
        };
        for (objective, &score) in &outcome.objective_scores {
            let direction = objectives.direction_for_or_primary(objective);
            let directional = direction.directional_score(score);
            match best.get_mut(objective) {
                Some((best_score, winners)) => {
                    if directional > *best_score {
                        *best_score = directional;
                        winners.clear();
                        winners.push(node.id.clone());
                    } else if directional == *best_score {
                        winners.push(node.id.clone());
                    }
                }
                None => {
                    best.insert(objective.clone(), (directional, vec![node.id.clone()]));
                }
            }
        }
    }
    let mapping = best
        .into_iter()
        .map(|(objective, (_, winners))| (FrontierKey::Objective(objective), winners))
        .collect();
    ParetoFront { mapping }
}

fn cartesian_front(pool: &[&ExperimentNode], objectives: &Objectives) -> ParetoFront {
    let mut best: BTreeMap<(String, String), (f64, Vec<String>)> = BTreeMap::new();
    for node in pool {
        let Some(outcome) = &node.outcome else {
            continue;
        };
        for (example_id, per_objective) in &outcome.objective_subscores {
            for (objective, &score) in per_objective {
                let direction = objectives.direction_for_or_primary(objective);
                let directional = direction.directional_score(score);
                match best.get_mut(&(example_id.clone(), objective.clone())) {
                    Some((best_score, winners)) => {
                        if directional > *best_score {
                            *best_score = directional;
                            winners.clear();
                            winners.push(node.id.clone());
                        } else if directional == *best_score {
                            winners.push(node.id.clone());
                        }
                    }
                    None => {
                        best.insert(
                            (example_id.clone(), objective.clone()),
                            (directional, vec![node.id.clone()]),
                        );
                    }
                }
            }
        }
    }
    // Cartesian falls back to Hybrid when no per-pair data is present.
    if best.is_empty() {
        let mut front = instance_front(pool, objectives);
        for (key, winners) in objective_front(pool, objectives).mapping {
            front.mapping.insert(key, winners);
        }
        return front;
    }
    let mapping = best
        .into_iter()
        .map(|((instance, objective), (_, winners))| {
            (
                FrontierKey::Cartesian {
                    instance,
                    objective,
                },
                winners,
            )
        })
        .collect();
    ParetoFront { mapping }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ExperimentGraph, Hypothesis};
    use crate::outcome::{Metric, TrialOutcome};

    /// A node that wins one example stays non-dominated even if its aggregate
    /// is lower — the core property GEPA's instance frontier provides and that
    /// scalar frontier selection loses. Reverting the Pareto change (e.g.
    /// ranking by aggregate only) makes this test fail.
    #[test]
    fn instance_frontier_keeps_single_example_specialist() {
        let metric = Metric::maximize("accuracy");
        let objectives = Objectives::single(metric.clone());
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

        let pool: Vec<&_> = graph.evaluated_pool();
        let nd = non_dominated_set(&pool, FrontierType::Instance, &objectives);
        let ids: Vec<&str> = nd.iter().map(|n| n.id.as_str()).collect();

        assert!(
            ids.contains(&specialist.as_str()),
            "specialist must stay non-dominated"
        );
        assert!(
            ids.contains(&generalist.as_str()),
            "generalist must stay non-dominated"
        );
    }

    #[test]
    fn objective_frontier_keeps_single_objective_winner() {
        let objectives =
            Objectives::new(vec![Metric::maximize("accuracy"), Metric::minimize("cost")]);
        let mut graph = ExperimentGraph::new("run");

        let a = graph
            .allocate_child("root", Hypothesis::new("accurate"))
            .unwrap();
        graph
            .record_outcome(
                &a,
                TrialOutcome::passed(0.9, "").with_objective_scores(
                    [("accuracy".to_string(), 0.9), ("cost".to_string(), 5.0)]
                        .into_iter()
                        .collect(),
                ),
            )
            .unwrap();
        graph.commit(&a, "a").unwrap();

        let b = graph
            .allocate_child("root", Hypothesis::new("cheap"))
            .unwrap();
        graph
            .record_outcome(
                &b,
                TrialOutcome::passed(0.5, "").with_objective_scores(
                    [("accuracy".to_string(), 0.5), ("cost".to_string(), 1.0)]
                        .into_iter()
                        .collect(),
                ),
            )
            .unwrap();
        graph.discard(&b, "lower aggregate").unwrap();

        let pool: Vec<&_> = graph.evaluated_pool();
        let nd = non_dominated_set(&pool, FrontierType::Objective, &objectives);
        let ids: Vec<&str> = nd.iter().map(|n| n.id.as_str()).collect();

        assert!(ids.contains(&a.as_str()), "a wins accuracy");
        assert!(ids.contains(&b.as_str()), "b wins cost (minimize)");
    }

    #[test]
    fn hybrid_frontier_combines_instance_and_objective() {
        let objectives = Objectives::new(vec![Metric::maximize("accuracy")]);
        let mut graph = ExperimentGraph::new("run");

        let a = graph.allocate_child("root", Hypothesis::new("a")).unwrap();
        graph
            .record_outcome(
                &a,
                TrialOutcome::passed(0.7, "")
                    .with_val_subscores([("ex_0".to_string(), 1.0)].into_iter().collect()),
            )
            .unwrap();
        graph.commit(&a, "a").unwrap();

        let b = graph.allocate_child("root", Hypothesis::new("b")).unwrap();
        graph
            .record_outcome(
                &b,
                TrialOutcome::passed(0.6, "")
                    .with_objective_scores([("accuracy".to_string(), 0.99)].into_iter().collect()),
            )
            .unwrap();
        graph.discard(&b, "lower aggregate").unwrap();

        let pool: Vec<&_> = graph.evaluated_pool();
        let front = pareto_front(&pool, FrontierType::Hybrid, &objectives);
        let ids = front.non_dominated_ids();
        assert!(ids.contains(&a), "a wins instance ex_0");
        assert!(ids.contains(&b), "b wins objective accuracy");
    }

    #[test]
    fn cartesian_frontier_uses_per_pair_data() {
        let objectives = Objectives::new(vec![Metric::maximize("accuracy")]);
        let mut graph = ExperimentGraph::new("run");

        let a = graph.allocate_child("root", Hypothesis::new("a")).unwrap();
        let mut a_pairs: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        a_pairs.insert(
            "ex_0".to_string(),
            [("accuracy".to_string(), 1.0)].into_iter().collect(),
        );
        graph
            .record_outcome(&a, TrialOutcome::passed(0.9, ""))
            .unwrap();
        graph.commit(&a, "a").unwrap();

        // b wins (ex_1, accuracy) which a never scored
        let b = graph.allocate_child("root", Hypothesis::new("b")).unwrap();
        let mut b_pairs: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        b_pairs.insert(
            "ex_1".to_string(),
            [("accuracy".to_string(), 1.0)].into_iter().collect(),
        );
        graph
            .record_outcome(&b, TrialOutcome::passed(0.4, ""))
            .unwrap();
        graph.discard(&b, "lower aggregate").unwrap();

        // Attach per-pair data via outcome mutation.
        let a_node = graph.node_mut(&a).unwrap();
        if let Some(outcome) = &mut a_node.outcome {
            outcome.objective_subscores = a_pairs;
        }
        let b_node = graph.node_mut(&b).unwrap();
        if let Some(outcome) = &mut b_node.outcome {
            outcome.objective_subscores = b_pairs;
        }

        let pool: Vec<&_> = graph.evaluated_pool();
        let front = pareto_front(&pool, FrontierType::Cartesian, &objectives);
        let ids = front.non_dominated_ids();
        assert!(ids.contains(&a), "a wins (ex_0, accuracy)");
        assert!(ids.contains(&b), "b wins (ex_1, accuracy)");
    }

    #[test]
    fn win_count_ranks_broader_winners_higher() {
        let metric = Metric::maximize("accuracy");
        let objectives = Objectives::single(metric);
        let mut graph = ExperimentGraph::new("run");

        let a = graph.allocate_child("root", Hypothesis::new("a")).unwrap();
        graph
            .record_outcome(
                &a,
                TrialOutcome::passed(0.9, "").with_val_subscores(
                    [("ex_0".to_string(), 1.0), ("ex_1".to_string(), 1.0)]
                        .into_iter()
                        .collect(),
                ),
            )
            .unwrap();
        graph.commit(&a, "a").unwrap();

        let b = graph.allocate_child("root", Hypothesis::new("b")).unwrap();
        graph
            .record_outcome(
                &b,
                TrialOutcome::passed(0.5, "")
                    .with_val_subscores([("ex_0".to_string(), 0.2)].into_iter().collect()),
            )
            .unwrap();
        graph.discard(&b, "lower").unwrap();

        let pool: Vec<&_> = graph.evaluated_pool();
        let front = pareto_front(&pool, FrontierType::Instance, &objectives);
        assert!(front.win_count(&a) >= 2);
        assert_eq!(front.win_count(&b), 0);
    }

    #[test]
    fn minimize_direction_respected_on_instance_frontier() {
        let metric = Metric::minimize("latency_ms");
        let objectives = Objectives::single(metric);
        let mut graph = ExperimentGraph::new("run");

        let a = graph.allocate_child("root", Hypothesis::new("a")).unwrap();
        graph
            .record_outcome(
                &a,
                TrialOutcome::passed(10.0, "")
                    .with_val_subscores([("ex_0".to_string(), 100.0)].into_iter().collect()),
            )
            .unwrap();
        graph.commit(&a, "a").unwrap();

        let b = graph.allocate_child("root", Hypothesis::new("b")).unwrap();
        graph
            .record_outcome(
                &b,
                TrialOutcome::passed(20.0, "")
                    .with_val_subscores([("ex_0".to_string(), 5.0)].into_iter().collect()),
            )
            .unwrap();
        graph.discard(&b, "higher aggregate latency").unwrap();

        let pool: Vec<&_> = graph.evaluated_pool();
        let front = pareto_front(&pool, FrontierType::Instance, &objectives);
        // b wins ex_0 because lower latency is better, despite higher aggregate.
        assert!(front.non_dominated_ids().contains(&b));
    }
}
