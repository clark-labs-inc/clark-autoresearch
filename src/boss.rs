use crate::graph::ResearchMode;
use crate::opportunity;
use serde::{Deserialize, Serialize};

pub type BossSurfaceKind = opportunity::SurfaceKind;
pub type BossOpportunity = opportunity::ResearchOpportunity;
pub type BossResearchBias = opportunity::ResearchBias;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BossDispatchClass {
    SurfaceExplorer,
    VulnerabilityProber,
    ExploitValidator,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BossDispatchHint {
    pub node_id: String,
    pub surface: BossSurfaceKind,
    pub mode: ResearchMode,
    pub dispatch_class: BossDispatchClass,
    pub score: f64,
    pub focus: String,
    pub rationale: String,
}

pub fn rank_boss_opportunities(
    opportunities: &[BossOpportunity],
    bias: &BossResearchBias,
) -> Vec<BossDispatchHint> {
    opportunity::rank_opportunities(opportunities, bias)
        .into_iter()
        .map(|hint| BossDispatchHint {
            node_id: hint.node_id,
            surface: hint.surface,
            mode: hint.mode,
            dispatch_class: boss_dispatch_class(hint.dispatch_class),
            score: hint.score,
            focus: hint.focus,
            rationale: hint.rationale,
        })
        .collect()
}

pub fn dispatch_class_for(opportunity: &BossOpportunity, mode: ResearchMode) -> BossDispatchClass {
    boss_dispatch_class(opportunity::dispatch_class_for(opportunity, mode))
}

fn boss_dispatch_class(dispatch_class: opportunity::DispatchClass) -> BossDispatchClass {
    match dispatch_class {
        opportunity::DispatchClass::SurfaceExplorer => BossDispatchClass::SurfaceExplorer,
        opportunity::DispatchClass::HypothesisProber => BossDispatchClass::VulnerabilityProber,
        opportunity::DispatchClass::ProofValidator => BossDispatchClass::ExploitValidator,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_filter_drops_unscoped_opportunities() {
        let bias = BossResearchBias::default();
        let ranked = rank_boss_opportunities(
            &[
                BossOpportunity {
                    node_id: "outside".into(),
                    surface: BossSurfaceKind::Hypothesis,
                    priority: 1.0,
                    novelty: 1.0,
                    confidence: 1.0,
                    impact: 1.0,
                    in_scope: false,
                    requires_validation: true,
                },
                BossOpportunity {
                    node_id: "inside".into(),
                    surface: BossSurfaceKind::Target,
                    priority: 0.2,
                    novelty: 1.0,
                    confidence: 0.2,
                    impact: 0.2,
                    in_scope: true,
                    requires_validation: false,
                },
            ],
            &bias,
        );

        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].node_id, "inside");
    }

    #[test]
    fn validation_bias_ranks_proof_work_first() {
        let ranked = rank_boss_opportunities(
            &[
                BossOpportunity {
                    node_id: "map".into(),
                    surface: BossSurfaceKind::Target,
                    priority: 0.9,
                    novelty: 1.0,
                    confidence: 0.2,
                    impact: 0.3,
                    in_scope: true,
                    requires_validation: false,
                },
                BossOpportunity {
                    node_id: "prove".into(),
                    surface: BossSurfaceKind::Hypothesis,
                    priority: 0.8,
                    novelty: 0.2,
                    confidence: 0.8,
                    impact: 0.9,
                    in_scope: true,
                    requires_validation: true,
                },
            ],
            &BossResearchBias::default(),
        );

        assert_eq!(ranked[0].node_id, "prove");
        assert_eq!(ranked[0].mode, ResearchMode::Validate);
    }
}
