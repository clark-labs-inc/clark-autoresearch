use serde::{Deserialize, Serialize};

use crate::graph::ResearchMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    Target,
    Endpoint,
    Finding,
    Hypothesis,
    Evidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchClass {
    SurfaceExplorer,
    HypothesisProber,
    ProofValidator,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResearchOpportunity {
    pub node_id: String,
    pub surface: SurfaceKind,
    /// Urgency from 0.0 to 1.0. Higher means the loop should spend attention sooner.
    pub priority: f64,
    /// How much new map information the opportunity is expected to reveal.
    pub novelty: f64,
    /// How likely the current evidence is to hold up.
    pub confidence: f64,
    /// Expected value if the hypothesis is confirmed.
    pub impact: f64,
    /// Whether the caller has already checked this opportunity against its constraints.
    #[serde(default)]
    pub in_scope: bool,
    /// Whether the next best action is proof-oriented validation.
    #[serde(default)]
    pub requires_validation: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResearchBias {
    pub explore_weight: f64,
    pub exploit_weight: f64,
    pub validation_weight: f64,
    pub require_in_scope: bool,
}

impl Default for ResearchBias {
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
pub struct DispatchHint {
    pub node_id: String,
    pub surface: SurfaceKind,
    pub mode: ResearchMode,
    pub dispatch_class: DispatchClass,
    pub score: f64,
    pub focus: String,
    pub rationale: String,
}

pub fn rank_opportunities(
    opportunities: &[ResearchOpportunity],
    bias: &ResearchBias,
) -> Vec<DispatchHint> {
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
            DispatchHint {
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

pub fn dispatch_class_for(opportunity: &ResearchOpportunity, mode: ResearchMode) -> DispatchClass {
    match mode {
        ResearchMode::Explore => DispatchClass::SurfaceExplorer,
        ResearchMode::Validate => DispatchClass::ProofValidator,
        ResearchMode::Exploit => match opportunity.surface {
            SurfaceKind::Hypothesis => DispatchClass::ProofValidator,
            SurfaceKind::Finding | SurfaceKind::Endpoint => DispatchClass::HypothesisProber,
            SurfaceKind::Target | SurfaceKind::Evidence => DispatchClass::SurfaceExplorer,
        },
    }
}

fn recommended_mode(opportunity: &ResearchOpportunity) -> ResearchMode {
    if opportunity.requires_validation {
        return ResearchMode::Validate;
    }
    match opportunity.surface {
        SurfaceKind::Target | SurfaceKind::Evidence => ResearchMode::Explore,
        SurfaceKind::Endpoint => {
            if opportunity.impact + opportunity.confidence >= opportunity.novelty + 0.25 {
                ResearchMode::Exploit
            } else {
                ResearchMode::Explore
            }
        }
        SurfaceKind::Finding | SurfaceKind::Hypothesis => ResearchMode::Exploit,
    }
}

fn surface_prior(surface: SurfaceKind) -> f64 {
    match surface {
        SurfaceKind::Target => 0.05,
        SurfaceKind::Endpoint => 0.12,
        SurfaceKind::Finding => 0.18,
        SurfaceKind::Hypothesis => 0.20,
        SurfaceKind::Evidence => 0.08,
    }
}

fn focus_for(opportunity: &ResearchOpportunity, mode: ResearchMode) -> String {
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
        let bias = ResearchBias::default();
        let ranked = rank_opportunities(
            &[
                ResearchOpportunity {
                    node_id: "outside".into(),
                    surface: SurfaceKind::Hypothesis,
                    priority: 1.0,
                    novelty: 1.0,
                    confidence: 1.0,
                    impact: 1.0,
                    in_scope: false,
                    requires_validation: true,
                },
                ResearchOpportunity {
                    node_id: "inside".into(),
                    surface: SurfaceKind::Target,
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
        let ranked = rank_opportunities(
            &[
                ResearchOpportunity {
                    node_id: "map".into(),
                    surface: SurfaceKind::Target,
                    priority: 0.9,
                    novelty: 1.0,
                    confidence: 0.2,
                    impact: 0.3,
                    in_scope: true,
                    requires_validation: false,
                },
                ResearchOpportunity {
                    node_id: "prove".into(),
                    surface: SurfaceKind::Hypothesis,
                    priority: 0.8,
                    novelty: 0.2,
                    confidence: 0.8,
                    impact: 0.9,
                    in_scope: true,
                    requires_validation: true,
                },
            ],
            &ResearchBias::default(),
        );

        assert_eq!(ranked[0].node_id, "prove");
        assert_eq!(ranked[0].mode, ResearchMode::Validate);
    }
}
