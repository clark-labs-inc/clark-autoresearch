//! Enforceable stop conditions for the optimization loop.
//!
//! Replaces the descriptive [`crate::lm_loop::WorkBudget`] (which only
//! *describes* a budget but is never enforced) with an actual contract the
//! loop body can consult. Ports GEPA's `gepa.utils.stop_condition` stoppers
//! (`MaxMetricCallsStopper`, `NoImprovementStopper`, `FileStopper`,
//! `CompositeStopper`) into clark's state model so a host-driven loop can stop
//! on a real budget instead of running until the host kills it.

use std::path::PathBuf;

/// A snapshot of loop progress that a [`StopCondition`] inspects.
#[derive(Clone, Debug, Default)]
pub struct LoopSnapshot {
    /// How many loop iterations have completed.
    pub iteration: u32,
    /// How many metric (evaluation) calls have been made so far.
    pub metric_calls: u32,
    /// Iterations since the last accepted improvement (best score update).
    pub iterations_since_improvement: u32,
}

/// An enforceable stop condition for the optimization loop.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StopCondition {
    /// Stop after a maximum number of metric (evaluation) calls.
    MaxMetricCalls { max: u32 },
    /// Stop after a maximum number of loop iterations.
    MaxIterations { max: u32 },
    /// Stop when no improvement has been accepted for `window` iterations.
    NoImprovement { window: u32 },
    /// Stop when the given file exists (graceful out-of-band stop).
    ///
    /// Mirrors GEPA's `FileStopper`: a host or operator touches the file to
    /// halt the loop between iterations.
    FileStopper { path: PathBuf },
    /// A conjunction of stop conditions: stop as soon as *any* member fires.
    Composite { conditions: Vec<StopCondition> },
}

impl StopCondition {
    /// Returns `true` when the loop should stop, given the current snapshot.
    pub fn should_stop(&self, snapshot: &LoopSnapshot) -> bool {
        match self {
            StopCondition::MaxMetricCalls { max } => snapshot.metric_calls >= *max,
            StopCondition::MaxIterations { max } => snapshot.iteration >= *max,
            StopCondition::NoImprovement { window } => {
                snapshot.iterations_since_improvement >= *window
            }
            StopCondition::FileStopper { path } => path.exists(),
            StopCondition::Composite { conditions } => conditions
                .iter()
                .any(|condition| condition.should_stop(snapshot)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_metric_calls_stops_at_budget() {
        let stop = StopCondition::MaxMetricCalls { max: 10 };
        assert!(!stop.should_stop(&LoopSnapshot {
            metric_calls: 9,
            ..Default::default()
        }));
        assert!(stop.should_stop(&LoopSnapshot {
            metric_calls: 10,
            ..Default::default()
        }));
    }

    #[test]
    fn max_iterations_stops_at_budget() {
        let stop = StopCondition::MaxIterations { max: 3 };
        assert!(!stop.should_stop(&LoopSnapshot {
            iteration: 2,
            ..Default::default()
        }));
        assert!(stop.should_stop(&LoopSnapshot {
            iteration: 3,
            ..Default::default()
        }));
    }

    #[test]
    fn no_improvement_stops_after_window() {
        let stop = StopCondition::NoImprovement { window: 5 };
        assert!(!stop.should_stop(&LoopSnapshot {
            iterations_since_improvement: 4,
            ..Default::default()
        }));
        assert!(stop.should_stop(&LoopSnapshot {
            iterations_since_improvement: 5,
            ..Default::default()
        }));
    }

    #[test]
    fn file_stopper_fires_when_file_exists() {
        let path = std::env::temp_dir().join("clark_autoresearch_stop_test_marker");
        let _ = std::fs::remove_file(&path);
        let stop = StopCondition::FileStopper { path: path.clone() };
        assert!(!stop.should_stop(&LoopSnapshot::default()));
        std::fs::write(&path, b"stop").unwrap();
        assert!(stop.should_stop(&LoopSnapshot::default()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn composite_stops_if_any_member_fires() {
        let stop = StopCondition::Composite {
            conditions: vec![
                StopCondition::MaxIterations { max: 100 },
                StopCondition::NoImprovement { window: 3 },
            ],
        };
        assert!(!stop.should_stop(&LoopSnapshot {
            iteration: 1,
            iterations_since_improvement: 2,
            ..Default::default()
        }));
        assert!(stop.should_stop(&LoopSnapshot {
            iteration: 1,
            iterations_since_improvement: 3,
            ..Default::default()
        }));
    }

    #[test]
    fn composite_stops_on_max_metric_calls_member() {
        // Reproduction tie-in for Phase 3: the loop stops on MaxMetricCalls.
        let stop = StopCondition::Composite {
            conditions: vec![
                StopCondition::MaxMetricCalls { max: 5 },
                StopCondition::NoImprovement { window: 100 },
            ],
        };
        assert!(stop.should_stop(&LoopSnapshot {
            metric_calls: 5,
            ..Default::default()
        }));
    }
}
