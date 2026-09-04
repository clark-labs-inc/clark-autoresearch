use serde::{Deserialize, Serialize};

use crate::frontier::FrontierStrategy;
use crate::outcome::Metric;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatePhase {
    Pre,
    #[default]
    Post,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gate {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub phase: GatePhase,
}

impl Gate {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            phase: GatePhase::Post,
        }
    }

    pub fn pre(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            phase: GatePhase::Pre,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateOutcome {
    pub name: String,
    pub phase: GatePhase,
    pub passed: bool,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub output_snippet: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResearchPolicy {
    pub metric: Metric,
    pub frontier_strategy: FrontierStrategy,
    #[serde(default)]
    pub gates: Vec<Gate>,
    pub max_attempts: u32,
}

impl ResearchPolicy {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            frontier_strategy: FrontierStrategy::default(),
            gates: Vec::new(),
            max_attempts: 3,
        }
    }

    pub fn with_gate(mut self, gate: Gate) -> Self {
        self.gates.push(gate);
        self
    }
}
