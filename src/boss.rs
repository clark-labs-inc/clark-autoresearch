use serde::{Deserialize, Serialize};

use crate::graph::ResearchMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BossSurfaceKind {
    Target,
    Endpoint,
    Finding,
    Hypothesis,
    Evidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BossDispatchClass {
    SurfaceExplorer,
    VulnerabilityProber,
    ExploitValidator,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BossOpportunity {
    pub node_id: String,
    pub surface: BossSurfaceKind,
    /// Urgency from 0.0 to 1.0. Higher means BOSS should spend attention sooner.
    pub priority: f64,
    /// How much new map information the opportunity is expected to reveal.
    pub novelty: f64,
    /// How likely the current evidence is to hold up.
    pub confidence: f64,
    /// Potential security impact if the hypothesis is confirmed.
    pub impact: f64,
    /// Whether BOSS has already checked this target against its authorization scope.
    #[serde(default)]
    pub in_scope: bool,
    /// Whether the next best action is proof-oriented validation.
    #[serde(default)]
    pub requires_validation: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BossResearchBias {
    pub explore_weight: f64,
    pub exploit_weight: f64,
    pub validation_weight: f64,
    pub require_in_scope: bool,
}

impl Default for BossResearchBias {
    fn default() -> Self {
        Self {
            explore_weight: 0.15,
            exploit_weight: 0.35,
            validation_weight: 0.45,
            require_in_scope: true,
        }
    }
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
    let mut ranked = opportunities
        .iter()
        .filter(|opportunity| !bias.require_in_scope || opportunity.in_scope)
        .map(|opportunity| {
            let mode = recommended_mode(opportunity);
            let dispatch_class = dispatch_class_for(opportunity, mode);
            let mode_bias = match mode {
                ResearchMode::Explore => bias.explore_weight,
                ResearchMode::Exploit => bias.exploit_weight,
                ResearchMode::Validate => bias.validation_weight,
            };
            let score = 0.30 * clamp01(opportunity.priority)
                + 0.20 * clamp01(opportunity.novelty)
                + 0.20 * clamp01(opportunity.confidence)
                + 0.30 * clamp01(opportunity.impact)
                + mode_bias
                + surface_prior(opportunity.surface);
            BossDispatchHint {
                node_id: opportunity.node_id.clone(),
                surface: opportunity.surface,
                mode,
                dispatch_class,
                score,
                focus: focus_for(opportunity, mode),
                rationale: format!(
                    "surface={:?} priority={:.2} novelty={:.2} confidence={:.2} impact={:.2}",
                    opportunity.surface,
                    clamp01(opportunity.priority),
                    clamp01(opportunity.novelty),
                    clamp01(opportunity.confidence),
                    clamp01(opportunity.impact)
                ),
            }
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    ranked
}

pub fn dispatch_class_for(opportunity: &BossOpportunity, mode: ResearchMode) -> BossDispatchClass {
    match mode {
        ResearchMode::Explore => BossDispatchClass::SurfaceExplorer,
        ResearchMode::Validate => BossDispatchClass::ExploitValidator,
        ResearchMode::Exploit => match opportunity.surface {
            BossSurfaceKind::Hypothesis => BossDispatchClass::ExploitValidator,
            BossSurfaceKind::Finding | BossSurfaceKind::Endpoint => {
                BossDispatchClass::VulnerabilityProber
            }
            BossSurfaceKind::Target | BossSurfaceKind::Evidence => {
                BossDispatchClass::SurfaceExplorer
            }
        },
    }
}

fn recommended_mode(opportunity: &BossOpportunity) -> ResearchMode {
    if opportunity.requires_validation {
        return ResearchMode::Validate;
    }
    match opportunity.surface {
        BossSurfaceKind::Target | BossSurfaceKind::Evidence => ResearchMode::Explore,
        BossSurfaceKind::Endpoint => {
            if opportunity.impact + opportunity.confidence >= opportunity.novelty + 0.25 {
                ResearchMode::Exploit
            } else {
                ResearchMode::Explore
            }
        }
        BossSurfaceKind::Finding | BossSurfaceKind::Hypothesis => ResearchMode::Exploit,
    }
}

fn surface_prior(surface: BossSurfaceKind) -> f64 {
    match surface {
        BossSurfaceKind::Target => 0.05,
        BossSurfaceKind::Endpoint => 0.12,
        BossSurfaceKind::Finding => 0.18,
        BossSurfaceKind::Hypothesis => 0.20,
        BossSurfaceKind::Evidence => 0.08,
    }
}

fn focus_for(opportunity: &BossOpportunity, mode: ResearchMode) -> String {
    match mode {
        ResearchMode::Explore => format!("map additional surface around {}", opportunity.node_id),
        ResearchMode::Exploit => format!(
            "probe the strongest concrete hypothesis on {}",
            opportunity.node_id
        ),
        ResearchMode::Validate => format!("validate proof end-to-end for {}", opportunity.node_id),
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
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
