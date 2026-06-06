use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph::{ExperimentGraph, ExperimentId, ResearchMode};
use crate::ids::stable_unit;
use crate::outcome::{Metric, MetricDirection};

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ExperimentGraph, Hypothesis};
    use crate::outcome::{Metric, MetricDirection, TaskScore, TrialOutcome};

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
