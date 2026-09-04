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
    /// Per-objective aggregate scores (objective name -> score).
    ///
    /// Enables true multi-objective Pareto frontiers (Objective / Hybrid /
    /// Cartesian frontier types). Empty for back-compat with single-metric
    /// outcomes; the scalar `score` remains the primary aggregate.
    #[serde(default)]
    pub objective_scores: BTreeMap<String, f64>,
    /// Per-example validation subscores (example id -> score).
    ///
    /// Enables the per-instance Pareto frontier (GEPA's instance frontier
    /// type), which keeps candidates that win a single example selectable as
    /// parents even when their aggregate score is lower.
    #[serde(default)]
    pub val_subscores: BTreeMap<String, f64>,
    /// Per-example, per-objective scores (example id -> objective -> score).
    ///
    /// Enables the Cartesian frontier type (per (example, objective) pair).
    /// Empty by default; Cartesian falls back to Hybrid when this is absent.
    #[serde(default)]
    pub objective_subscores: BTreeMap<String, BTreeMap<String, f64>>,
}

impl TrialOutcome {
    pub fn passed(score: f64, summary: impl Into<String>) -> Self {
        Self {
            score,
            status: OutcomeStatus::Passed,
            summary: summary.into(),
            task_scores: Vec::new(),
            metadata: BTreeMap::new(),
            objective_scores: BTreeMap::new(),
            val_subscores: BTreeMap::new(),
            objective_subscores: BTreeMap::new(),
        }
    }

    pub fn failed(score: f64, summary: impl Into<String>) -> Self {
        Self {
            score,
            status: OutcomeStatus::Failed,
            summary: summary.into(),
            task_scores: Vec::new(),
            metadata: BTreeMap::new(),
            objective_scores: BTreeMap::new(),
            val_subscores: BTreeMap::new(),
            objective_subscores: BTreeMap::new(),
        }
    }

    pub fn with_task_scores(mut self, task_scores: Vec<TaskScore>) -> Self {
        self.task_scores = task_scores;
        self
    }

    pub fn with_objective_scores(mut self, objective_scores: BTreeMap<String, f64>) -> Self {
        self.objective_scores = objective_scores;
        self
    }

    pub fn with_val_subscores(mut self, val_subscores: BTreeMap<String, f64>) -> Self {
        self.val_subscores = val_subscores;
        self
    }
}

/// One or more named objectives with per-objective optimization directions.
///
/// A single `Metric` is the degenerate case (one objective). Multiple
/// objectives enable true multi-objective Pareto frontiers: a candidate that
/// is best on one objective stays selectable as a parent even if it is worse
/// on another.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objectives {
    pub metrics: Vec<Metric>,
}

impl Objectives {
    pub fn single(metric: Metric) -> Self {
        Self {
            metrics: vec![metric],
        }
    }

    pub fn new(metrics: Vec<Metric>) -> Self {
        Self { metrics }
    }

    /// The primary (first) objective, used for aggregate ranking and as the
    /// fallback direction when an objective name has no explicit direction.
    pub fn primary(&self) -> &Metric {
        self.metrics
            .first()
            .expect("Objectives must contain at least one metric")
    }

    /// Look up the optimization direction for a named objective.
    pub fn direction_for(&self, name: &str) -> Option<MetricDirection> {
        self.metrics
            .iter()
            .find(|metric| metric.name == name)
            .map(|metric| metric.direction)
    }

    /// Resolve the direction for a named objective, falling back to the
    /// primary objective's direction when the name is unknown.
    pub fn direction_for_or_primary(&self, name: &str) -> MetricDirection {
        self.direction_for(name).unwrap_or(self.primary().direction)
    }
}

impl From<Metric> for Objectives {
    fn from(metric: Metric) -> Self {
        Self::single(metric)
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
