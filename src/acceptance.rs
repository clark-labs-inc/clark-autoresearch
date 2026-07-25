//! Acceptance criteria for proposed candidates.
//!
//! Ports GEPA's `gepa.strategies.acceptance` (~30 lines) into clark's state
//! model. An acceptance criterion decides whether a proposed candidate should
//! be kept, based on whether its score improves on its parent's. The verdict
//! carries a human-readable reason so rejections are explainable — GEPA's
//! `StrictImprovementAcceptance` is the default; `ImprovementOrEqualAcceptance`
//! allows lateral moves that explore the solution space without regressing.
//!
//! GEPA applies the criterion in its engine loop; clark exposes it as a
//! standalone, host-callable policy so the loop body (see `src/loop.rs`) and
//! the frontier ranking path can enforce it without bundling execution.

use serde::{Deserialize, Serialize};

use crate::outcome::Metric;

/// Policy for accepting a proposed candidate against its parent's score.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceCriterion {
    /// Accept only if the candidate is strictly better than the parent.
    ///
    /// This is GEPA's default. Equal-score proposals are rejected so the loop
    /// does not churn on lateral moves.
    #[default]
    StrictImprovement,
    /// Accept if the candidate is at least as good as the parent.
    ///
    /// Useful when lateral moves that do not regress the score are wanted to
    /// explore different regions of the solution space.
    ImprovementOrEqual,
}

/// The verdict returned by [`AcceptanceCriterion::should_accept`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptanceVerdict {
    pub accepted: bool,
    pub reason: String,
}

impl AcceptanceCriterion {
    /// Judge whether a candidate's score improves on its parent's.
    ///
    /// - `candidate_score`: the proposed candidate's score.
    /// - `parent_score`: the parent's score, or `None` when the parent has no
    ///   recorded outcome (e.g. a child of the synthetic root). A parentless
    ///   candidate is always accepted — there is nothing to improve on.
    /// - `metric`: supplies the optimization direction (maximize/minimize).
    ///
    /// Returns a verdict with a reason. When rejected, the reason explains
    /// why (e.g. equal score under strict improvement).
    pub fn should_accept(
        self,
        candidate_score: f64,
        parent_score: Option<f64>,
        metric: &Metric,
    ) -> AcceptanceVerdict {
        let Some(parent) = parent_score else {
            return AcceptanceVerdict {
                accepted: true,
                reason: "no parent score to improve on".to_string(),
            };
        };
        match self {
            AcceptanceCriterion::StrictImprovement => {
                if metric.direction.is_better(candidate_score, parent) {
                    AcceptanceVerdict {
                        accepted: true,
                        reason: format!(
                            "strict improvement: {candidate_score} > {parent} ({})",
                            metric.name
                        ),
                    }
                } else {
                    AcceptanceVerdict {
                        accepted: false,
                        reason: format!(
                            "strict improvement required: candidate {candidate_score} is not better than parent {parent} ({}, {:?})",
                            metric.name, metric.direction
                        ),
                    }
                }
            }
            AcceptanceCriterion::ImprovementOrEqual => {
                if metric.direction.is_better_or_equal(candidate_score, parent) {
                    AcceptanceVerdict {
                        accepted: true,
                        reason: format!(
                            "improvement-or-equal: {candidate_score} >= {parent} ({})",
                            metric.name
                        ),
                    }
                } else {
                    AcceptanceVerdict {
                        accepted: false,
                        reason: format!(
                            "improvement-or-equal required: candidate {candidate_score} is worse than parent {parent} ({}, {:?})",
                            metric.name, metric.direction
                        ),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_improvement_rejects_equal_score_with_reason() {
        // Reproduction for Phase 2: a strict-improvement criterion must reject
        // an equal-score proposal and report the reason. Reverting the
        // criterion to improvement-or-equal (or dropping the reject path) makes
        // this test fail.
        let metric = Metric::maximize("accuracy");
        let verdict =
            AcceptanceCriterion::StrictImprovement.should_accept(0.82, Some(0.82), &metric);
        assert!(!verdict.accepted);
        assert!(
            verdict.reason.contains("strict improvement required"),
            "reason should explain the rejection: {}",
            verdict.reason
        );
    }

    #[test]
    fn strict_improvement_accepts_higher_score() {
        let metric = Metric::maximize("accuracy");
        let verdict =
            AcceptanceCriterion::StrictImprovement.should_accept(0.9, Some(0.82), &metric);
        assert!(verdict.accepted);
    }

    #[test]
    fn improvement_or_equal_accepts_equal_score() {
        let metric = Metric::maximize("accuracy");
        let verdict =
            AcceptanceCriterion::ImprovementOrEqual.should_accept(0.82, Some(0.82), &metric);
        assert!(verdict.accepted);
    }

    #[test]
    fn improvement_or_equal_rejects_lower_score() {
        let metric = Metric::maximize("accuracy");
        let verdict =
            AcceptanceCriterion::ImprovementOrEqual.should_accept(0.7, Some(0.82), &metric);
        assert!(!verdict.accepted);
        assert!(verdict.reason.contains("worse"));
    }

    #[test]
    fn minimize_direction_respected() {
        let metric = Metric::minimize("latency_ms");
        // Lower is better: 8.0 vs 10.0 should be accepted under strict.
        let verdict =
            AcceptanceCriterion::StrictImprovement.should_accept(8.0, Some(10.0), &metric);
        assert!(verdict.accepted);
        // Equal should be rejected under strict.
        let verdict =
            AcceptanceCriterion::StrictImprovement.should_accept(10.0, Some(10.0), &metric);
        assert!(!verdict.accepted);
    }

    #[test]
    fn parentless_candidate_is_accepted() {
        let metric = Metric::maximize("accuracy");
        let verdict = AcceptanceCriterion::StrictImprovement.should_accept(0.5, None, &metric);
        assert!(verdict.accepted);
        assert!(verdict.reason.contains("no parent"));
    }
}
