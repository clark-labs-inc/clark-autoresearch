use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Maximize,
    Minimize,
}

impl MetricDirection {
    pub fn is_better(self, candidate: f64, incumbent: f64) -> bool {
        match self {
            Self::Maximize => candidate > incumbent,
            Self::Minimize => candidate < incumbent,
        }
    }

    pub fn is_better_or_equal(self, candidate: f64, incumbent: f64) -> bool {
        match self {
            Self::Maximize => candidate >= incumbent,
            Self::Minimize => candidate <= incumbent,
        }
    }

    pub fn directional_score(self, score: f64) -> f64 {
        match self {
            Self::Maximize => score,
            Self::Minimize => -score,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub direction: MetricDirection,
}

impl Metric {
    pub fn maximize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: MetricDirection::Maximize,
        }
    }

    pub fn minimize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: MetricDirection::Minimize,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskScore {
    pub task_id: String,
    pub score: f64,
    #[serde(default)]
    pub direction: Option<MetricDirection>,
}

impl TaskScore {
    pub fn new(task_id: impl Into<String>, score: f64) -> Self {
        Self {
            task_id: task_id.into(),
            score,
            direction: None,
        }
    }

    pub fn with_direction(
        task_id: impl Into<String>,
        score: f64,
        direction: MetricDirection,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            score,
            direction: Some(direction),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialOutcome {
    pub score: f64,
    pub status: OutcomeStatus,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub task_scores: Vec<TaskScore>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl TrialOutcome {
    pub fn passed(score: f64, summary: impl Into<String>) -> Self {
        Self {
            score,
            status: OutcomeStatus::Passed,
            summary: summary.into(),
            task_scores: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn failed(score: f64, summary: impl Into<String>) -> Self {
        Self {
            score,
            status: OutcomeStatus::Failed,
            summary: summary.into(),
            task_scores: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_task_scores(mut self, task_scores: Vec<TaskScore>) -> Self {
        self.task_scores = task_scores;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_direction_compares_scores() {
        assert!(MetricDirection::Maximize.is_better(2.0, 1.0));
        assert!(MetricDirection::Minimize.is_better(1.0, 2.0));
        assert_eq!(MetricDirection::Minimize.directional_score(2.0), -2.0);
    }
}
