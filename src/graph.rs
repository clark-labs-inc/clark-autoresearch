use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::outcome::{Metric, TrialOutcome};
use crate::policy::Gate;

pub type ExperimentId = String;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchMode {
    #[default]
    Explore,
    Exploit,
    Validate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Root,
    Pending,
    Active,
    Evaluated,
    Committed,
    Discarded,
    Failed,
    Pruned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub statement: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub mode: ResearchMode,
}

impl Hypothesis {
    pub fn new(statement: impl Into<String>) -> Self {
        Self {
            statement: statement.into(),
            target: None,
            rationale: String::new(),
            mode: ResearchMode::Explore,
        }
    }

    pub fn with_mode(mut self, mode: ResearchMode) -> Self {
        self.mode = mode;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentNode {
    pub id: ExperimentId,
    #[serde(default)]
    pub parent: Option<ExperimentId>,
    #[serde(default)]
    pub children: Vec<ExperimentId>,
    pub status: ExperimentStatus,
    pub hypothesis: Hypothesis,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub outcome: Option<TrialOutcome>,
    #[serde(default)]
    pub gates: Vec<Gate>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub worktree: Option<String>,
    #[serde(default)]
    pub commit: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub pruned_reason: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl ExperimentNode {
    pub fn mode(&self) -> ResearchMode {
        self.hypothesis.mode
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentGraph {
    pub run_id: String,
    pub root: ExperimentId,
    pub next_id: u64,
    pub nodes: BTreeMap<ExperimentId, ExperimentNode>,
    #[serde(default)]
    pub workspace_notes: Vec<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("unknown experiment node: {0}")]
    UnknownNode(String),
    #[error("cannot allocate a child from terminal node {id} ({status:?})")]
    TerminalParent {
        id: ExperimentId,
        status: ExperimentStatus,
    },
    #[error("node has no parent: {0}")]
    MissingParent(ExperimentId),
}

impl ExperimentGraph {
    pub fn new(run_id: impl Into<String>) -> Self {
        let now = now_millis();
        let root = "root".to_string();
        let mut nodes = BTreeMap::new();
        nodes.insert(
            root.clone(),
            ExperimentNode {
                id: root.clone(),
                parent: None,
                children: Vec::new(),
                status: ExperimentStatus::Root,
                hypothesis: Hypothesis::new("synthetic root"),
                score: None,
                outcome: None,
                gates: Vec::new(),
                branch: None,
                worktree: None,
                commit: None,
                notes: Vec::new(),
                pruned_reason: None,
                created_at_ms: now,
                updated_at_ms: now,
            },
        );
        Self {
            run_id: run_id.into(),
            root,
            next_id: 0,
            nodes,
            workspace_notes: Vec::new(),
        }
    }

    pub fn node(&self, id: &str) -> Result<&ExperimentNode, GraphError> {
        self.nodes
            .get(id)
            .ok_or_else(|| GraphError::UnknownNode(id.to_string()))
    }

    pub fn node_mut(&mut self, id: &str) -> Result<&mut ExperimentNode, GraphError> {
        self.nodes
            .get_mut(id)
            .ok_or_else(|| GraphError::UnknownNode(id.to_string()))
    }

    pub fn allocate_child(
        &mut self,
        parent_id: &str,
        hypothesis: Hypothesis,
    ) -> Result<ExperimentId, GraphError> {
        let parent_status = self.node(parent_id)?.status;
        if matches!(
            parent_status,
            ExperimentStatus::Discarded | ExperimentStatus::Failed | ExperimentStatus::Pruned
        ) {
            return Err(GraphError::TerminalParent {
                id: parent_id.to_string(),
                status: parent_status,
            });
        }

        let id = format!("exp_{:04}", self.next_id);
        self.next_id += 1;
        let now = now_millis();
        let node = ExperimentNode {
            id: id.clone(),
            parent: Some(parent_id.to_string()),
            children: Vec::new(),
            status: ExperimentStatus::Pending,
            hypothesis,
            score: None,
            outcome: None,
            gates: Vec::new(),
            branch: Some(format!("autoresearch/{}/{}", self.run_id, id)),
            worktree: None,
            commit: None,
            notes: Vec::new(),
            pruned_reason: None,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.nodes.insert(id.clone(), node);
        self.node_mut(parent_id)?.children.push(id.clone());
        Ok(id)
    }

    pub fn set_status(&mut self, id: &str, status: ExperimentStatus) -> Result<(), GraphError> {
        let node = self.node_mut(id)?;
        node.status = status;
        node.updated_at_ms = now_millis();
        Ok(())
    }

    pub fn record_outcome(&mut self, id: &str, outcome: TrialOutcome) -> Result<(), GraphError> {
        let node = self.node_mut(id)?;
        node.score = Some(outcome.score);
        node.outcome = Some(outcome);
        node.status = ExperimentStatus::Evaluated;
        node.updated_at_ms = now_millis();
        Ok(())
    }

    pub fn commit(&mut self, id: &str, commit: impl Into<String>) -> Result<(), GraphError> {
        let node = self.node_mut(id)?;
        node.commit = Some(commit.into());
        node.status = ExperimentStatus::Committed;
        node.updated_at_ms = now_millis();
        Ok(())
    }

    pub fn discard(&mut self, id: &str, reason: impl Into<String>) -> Result<(), GraphError> {
        let node = self.node_mut(id)?;
        node.status = ExperimentStatus::Discarded;
        node.pruned_reason = Some(reason.into());
        node.updated_at_ms = now_millis();
        Ok(())
    }

    pub fn add_gate(&mut self, id: &str, gate: Gate) -> Result<(), GraphError> {
        let node = self.node_mut(id)?;
        node.gates.push(gate);
        node.updated_at_ms = now_millis();
        Ok(())
    }

    pub fn effective_gates(&self, id: &str) -> Result<Vec<Gate>, GraphError> {
        let mut gates = Vec::new();
        let mut seen = BTreeSet::new();
        for node in self.path_to_root(id)? {
            for gate in &node.gates {
                if seen.insert(gate.name.clone()) {
                    gates.push(gate.clone());
                }
            }
        }
        Ok(gates)
    }

    pub fn path_to_root(&self, id: &str) -> Result<Vec<&ExperimentNode>, GraphError> {
        let mut path = Vec::new();
        let mut cursor = id.to_string();
        loop {
            let node = self.node(&cursor)?;
            path.push(node);
            let Some(parent) = &node.parent else {
                break;
            };
            cursor = parent.clone();
        }
        path.reverse();
        Ok(path)
    }

    pub fn outcome_improves_parent(&self, id: &str, metric: &Metric) -> Result<bool, GraphError> {
        let node = self.node(id)?;
        let Some(candidate) = node.score else {
            return Ok(false);
        };
        let Some(parent_id) = &node.parent else {
            return Err(GraphError::MissingParent(id.to_string()));
        };
        let parent_score = self.node(parent_id)?.score;
        Ok(match parent_score {
            Some(parent) => metric.direction.is_better_or_equal(candidate, parent),
            None => true,
        })
    }

    pub fn frontier_nodes(&self) -> Vec<&ExperimentNode> {
        let mut out = Vec::new();
        for node in self.nodes.values() {
            if !matches!(
                node.status,
                ExperimentStatus::Root | ExperimentStatus::Committed
            ) {
                continue;
            }
            let has_live_child = node.children.iter().any(|child_id| {
                self.nodes
                    .get(child_id)
                    .map(|child| {
                        matches!(
                            child.status,
                            ExperimentStatus::Active | ExperimentStatus::Committed
                        )
                    })
                    .unwrap_or(false)
            });
            if !has_live_child {
                out.push(node);
            }
        }
        out
    }

    pub fn best_committed(&self, metric: &Metric) -> Option<&ExperimentNode> {
        self.nodes
            .values()
            .filter(|node| {
                matches!(node.status, ExperimentStatus::Committed) && node.score.is_some()
            })
            .max_by(|a, b| {
                let a_score = metric.direction.directional_score(a.score.unwrap());
                let b_score = metric.direction.directional_score(b.score.unwrap());
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.id.cmp(&a.id))
            })
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{Metric, TrialOutcome};
    use crate::policy::Gate;

    #[test]
    fn allocates_child_and_records_parent_lineage() {
        let mut graph = ExperimentGraph::new("run_0000");
        let child = graph
            .allocate_child("root", Hypothesis::new("try focused endpoint validation"))
            .unwrap();

        assert_eq!(child, "exp_0000");
        assert_eq!(graph.node("root").unwrap().children, vec!["exp_0000"]);
        assert_eq!(
            graph.node("exp_0000").unwrap().parent.as_deref(),
            Some("root")
        );
    }

    #[test]
    fn inherited_gates_are_deduplicated_by_name() {
        let mut graph = ExperimentGraph::new("run_0000");
        graph
            .add_gate("root", Gate::new("scope", "cargo test scope"))
            .unwrap();
        let child = graph
            .allocate_child("root", Hypothesis::new("probe auth bypass"))
            .unwrap();
        graph
            .add_gate(&child, Gate::new("scope", "cargo test stricter_scope"))
            .unwrap();
        graph
            .add_gate(&child, Gate::new("smoke", "cargo test"))
            .unwrap();

        let gates = graph.effective_gates(&child).unwrap();
        assert_eq!(
            gates.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(),
            vec!["scope", "smoke"]
        );
        assert_eq!(gates[0].command, "cargo test scope");
    }

    #[test]
    fn lower_metric_can_improve_parent() {
        let metric = Metric::minimize("latency_ms");
        let mut graph = ExperimentGraph::new("run_0000");
        let parent = graph
            .allocate_child("root", Hypothesis::new("baseline"))
            .unwrap();
        graph
            .record_outcome(&parent, TrialOutcome::passed(10.0, "baseline"))
            .unwrap();
        graph.commit(&parent, "abc").unwrap();

        let child = graph
            .allocate_child(&parent, Hypothesis::new("remove wasted work"))
            .unwrap();
        graph
            .record_outcome(&child, TrialOutcome::passed(8.0, "faster"))
            .unwrap();

        assert!(graph.outcome_improves_parent(&child, &metric).unwrap());
    }
}
