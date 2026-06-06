//! Clean-room primitives for autonomous research loops.
//!
//! The crate models the reusable parts of autoresearch systems:
//! experiment lineage, objective metrics, gates, outcomes, and frontier
//! selection. It intentionally avoids host-specific agent execution and any
//! scanner or exploit implementation so it can stay reusable and publishable.

pub mod boss;
pub mod frontier;
pub mod graph;
pub mod ids;
pub mod outcome;
pub mod policy;

pub use boss::{
    BossDispatchClass, BossDispatchHint, BossOpportunity, BossResearchBias, BossSurfaceKind,
    dispatch_class_for, rank_boss_opportunities,
};
pub use frontier::{FrontierStrategy, RankedCandidate, rank_frontier};
pub use graph::{
    ExperimentGraph, ExperimentId, ExperimentNode, ExperimentStatus, GraphError, Hypothesis,
    ResearchMode,
};
pub use outcome::{Metric, MetricDirection, OutcomeStatus, TaskScore, TrialOutcome};
pub use policy::{Gate, GateOutcome, GatePhase, ResearchPolicy};
