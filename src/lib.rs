//! Clean-room primitives for autonomous research loops.
//!
//! The crate models the reusable parts of autoresearch systems:
//! experiment lineage, objective metrics, gates, outcomes, and frontier
//! selection. It intentionally avoids host-specific agent execution and any
//! scanner or exploit implementation so it can stay reusable and publishable.

pub mod acceptance;
pub mod adapter;
pub mod boss;
pub mod cache;
pub mod embedding;
pub mod frontier;
pub mod graph;
pub mod ids;
pub mod lm_loop;
pub mod loop_opt;
pub mod opportunity;
pub mod outcome;
pub mod pareto;
pub mod policy;
pub mod proposer;
pub mod stop;

pub use acceptance::{AcceptanceCriterion, AcceptanceVerdict};
pub use adapter::{
    Candidate, EvaluationBatch, GateRunner, ObjectiveScores, ReflectiveDataset, ReflectiveEntry,
    ResearchAdapter,
};
pub use boss::{
    BossDispatchClass, BossDispatchHint, BossOpportunity, BossResearchBias, BossSurfaceKind,
    dispatch_class_for, rank_boss_opportunities,
};
pub use cache::{EvaluationCache, cached_evaluate, candidate_hash};
pub use embedding::Embedder;
pub use frontier::{FrontierStrategy, RankedCandidate, enforce_acceptance, rank_frontier};
pub use graph::{
    ExperimentGraph, ExperimentId, ExperimentNode, ExperimentStatus, GraphError, Hypothesis,
    ResearchMode,
};
pub use lm_loop::{
    AbsorbReport, ContextPack, EvidenceConfidence, ExperimentPlan, ExperimentState,
    HypothesisCandidate, HypothesisState, LedgerEvent, Observation, ResearchDossier,
    ResearchLedger, ResearchResult, ResultVerdict, WorkBudget, render_dossier,
};
pub use loop_opt::{OptimizationState, RejectionRecord, optimize};
pub use opportunity::{
    DispatchClass, DispatchHint, ResearchBias, ResearchOpportunity, SurfaceKind, rank_opportunities,
};
pub use outcome::{Metric, MetricDirection, Objectives, OutcomeStatus, TaskScore, TrialOutcome};
pub use pareto::{FrontierKey, FrontierType, ParetoFront, non_dominated_set, pareto_front};
pub use policy::{Gate, GateOutcome, GatePhase, ResearchPolicy};
pub use proposer::{DEFAULT_REFLECTION_TEMPLATE, LanguageModel, Proposer, ReflectiveMutation};
pub use stop::{LoopSnapshot, StopCondition};

// Optional semantic-similarity re-exports (only with the `similarity` feature).
#[cfg(feature = "similarity")]
pub use cache::SemanticCandidateCache;
#[cfg(feature = "similarity")]
pub use lm_loop::{SemanticSketches, relevance_ranked_dossier};
