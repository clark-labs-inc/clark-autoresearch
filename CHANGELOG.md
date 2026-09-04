# Changelog

## 0.2.0

Open-source cleanup release.

- **Breaking:** removed the `boss` module and its `Boss*` types
  (`BossDispatchClass`, `BossDispatchHint`, `BossOpportunity`,
  `BossResearchBias`, `BossSurfaceKind`, `rank_boss_opportunities`). Use the
  generic `opportunity` module (`rank_opportunities`, `DispatchClass`,
  `dispatch_class_for`) instead.
- Documentation: restructured README (how it works, published-crate install,
  badges) and replaced internal "clark" self-references in doc comments with
  crate-neutral prose.
- Metadata: added `documentation` (docs.rs) and `rust-version` (1.85 for the
  2024 edition).

Bring clark-autoresearch closer to GEPA ("improve_anything") while keeping
execution out of the core and adding the parts GEPA lacks.

- **Multi-objective state + per-instance Pareto.** `TrialOutcome` now carries
  `objective_scores`, `val_subscores`, and `objective_subscores`. New `pareto`
  module (`FrontierType::{Instance,Objective,Hybrid,Cartesian}`, true
  non-dominated set over the evaluated pool). New `FrontierStrategy::Pareto`.
  `ExperimentGraph::evaluated_pool` keeps discarded specialists selectable.
- **Acceptance criterion + stop conditions.** New `acceptance` module
  (`AcceptanceCriterion::{StrictImprovement,ImprovementOrEqual}` with reject
  reasons) and `stop` module (`StopCondition::{MaxMetricCalls,MaxIterations,
  NoImprovement,FileStopper,Composite}`). Frontier ranking can enforce
  acceptance via `enforce_acceptance` / `--acceptance`.
- **Execution-agnostic optimizer loop.** New `adapter` (`ResearchAdapter`,
  `GateRunner`), `proposer` (`LanguageModel`, `Proposer`,
  `ReflectiveMutation`), `cache` (`EvaluationCache`, `cached_evaluate`), and
  `loop_opt` (`OptimizationState`, `optimize`) modules. The loop runs
  propose → minibatch-eval → accept → full-eval → Pareto-update; the provider,
  sandbox, and evaluator stay behind host traits.
- **clark's unique findings.** Mode-aware parent selection (explore/exploit/
  validate from `ResearchBias`). Ledger-backed reflection (`render_dossier`
  fed into the proposer). Gate-gated `Validate` acceptance (inherited gates
  enforced in addition to score improvement — GEPA has no gate concept).
- **Optional `similarity` feature (clark-hash).** `Embedder` trait +
  `SemanticSketches` (semantic ledger retrieval via `find_similar`) +
  `SemanticCandidateCache` (near-duplicate guard) + novelty as
  `1.0 - max_similarity`. Default build stays dependency-light.
- **CLI + examples.** `optimize` subcommand drives the loop via a std-only
  HTTP eval adapter and a command proposer. `examples/optimize_loop.rs` and
  `examples/semantic_ledger.rs`.

## 0.1.0

- Initial public crate structure.
- Experiment graph, outcomes, gates, frontier ranking, opportunity ranking, CLI,
  examples, and open-source repository metadata.
