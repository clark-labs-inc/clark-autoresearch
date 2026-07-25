# clark-autoresearch

`clark-autoresearch` is a small Rust library and CLI for building autonomous
research loops: experiment lineage, scalar metrics, gates, outcomes, frontier
selection, and opportunity ranking.

It does not run model providers, scanners, browsers, or shell tools for you.
Instead, it gives agent systems a stable state model for deciding what to try
next and for recording whether a trial was worth keeping.

## What It Is Good For

- Tracking experiment trees for agent, prompt, model, retrieval, or evaluation
  work.
- Comparing candidates with maximize/minimize metrics.
- Keeping per-task scores so specialists are not lost behind one aggregate
  score.
- Ranking a frontier with `arg-max`, `top-k`, `epsilon-greedy`, `softmax`, or
  Pareto-per-task strategies.
- Carrying inherited gates such as `cargo test`, eval commands, smoke tests, or
  safety checks through an experiment lineage.
- Ranking generic research opportunities into `explore`, `exploit`, or
  `validate` modes.

## Install

From a local checkout:

```sh
cargo install --path .
```

Or use it directly:

```sh
cargo run -- --help
```

## CLI Quick Start

Initialize a state file:

```sh
clark-autoresearch init \
  --metric accuracy \
  --direction maximize \
  --gate test="cargo test"
```

Add a hypothesis:

```sh
clark-autoresearch spawn "shorter prompt with explicit answer checks" --mode explore
```

Record an outcome:

```sh
clark-autoresearch record exp_0000 0.82 \
  --status passed \
  --summary "accuracy improved on the smoke eval" \
  --task-score math=0.86:maximize \
  --task-score code=0.78:maximize
```

Commit or discard the trial:

```sh
clark-autoresearch commit exp_0000 --commit abc123
clark-autoresearch discard exp_0001 "regressed the code task"
```

Rank the next frontier:

```sh
clark-autoresearch frontier --strategy top-k --k 3
clark-autoresearch frontier --strategy pareto-per-task --k 5 --task-floor 0.5
```

Inspect state:

```sh
clark-autoresearch status
clark-autoresearch export > state.json
```

By default the CLI reads and writes `.autoresearch/state.json`. Use
`--state path/to/state.json` to work with another file.

## Opportunity Ranking

Opportunity ranking is useful when another system already has graph nodes or
work items and only needs a dispatch hint.

```sh
clark-autoresearch opportunity-rank examples/opportunities.json
```

Input can be either a JSON array of opportunities or an object with
`opportunities` and optional `bias`:

```json
{
  "bias": {
    "explore_weight": 0.15,
    "exploit_weight": 0.35,
    "validation_weight": 0.45,
    "require_in_scope": true
  },
  "opportunities": [
    {
      "node_id": "hypothesis:auth-cache",
      "surface": "hypothesis",
      "priority": 0.8,
      "novelty": 0.2,
      "confidence": 0.7,
      "impact": 0.9,
      "in_scope": true,
      "requires_validation": true
    }
  ]
}
```

## Library Example

```rust
use clark_autoresearch::{
    ExperimentGraph, FrontierStrategy, Hypothesis, Metric, TrialOutcome, rank_frontier,
};

fn main() -> anyhow::Result<()> {
    let metric = Metric::maximize("accuracy");
    let mut graph = ExperimentGraph::new("demo");

    let child = graph.allocate_child("root", Hypothesis::new("try a shorter prompt"))?;
    graph.record_outcome(&child, TrialOutcome::passed(0.82, "eval improved"))?;
    graph.commit(&child, "abc123")?;

    let ranked = rank_frontier(&graph, &metric, &FrontierStrategy::TopK { k: 3 });
    assert_eq!(ranked[0].id, child);
    Ok(())
}
```

See [examples/simple_loop.rs](examples/simple_loop.rs) for a complete example.

## Core Types

- `ExperimentGraph`: append-only experiment lineage with root, pending, active,
  evaluated, committed, discarded, failed, and pruned states.
- `Hypothesis`: the thing to try next, with optional target, rationale, and
  research mode.
- `TrialOutcome`: scalar score, pass/fail/inconclusive status, per-task scores,
  per-objective scores, and per-example validation subscores.
- `ResearchPolicy`: metric, gates, frontier strategy, and attempt budget.
- `FrontierStrategy`: deterministic policies for selecting the next candidate,
  including `Pareto` (per-instance / per-objective non-dominated set).
- `ResearchOpportunity`: generic opportunity score inputs for dispatch ranking.

## Optimization Loop (GEPA-inspired, execution-agnostic)

`optimize()` runs the GEPA loop shape — propose → minibatch-evaluate →
accept → full-evaluate → Pareto-update — but every provider call, sandbox, and
evaluator stays behind a host-implemented trait, so clark stays dep-light and
publishable.

- `ResearchAdapter` (host): evaluate a candidate on a batch and return
  scores + traces; build the reflective dataset.
- `Proposer` / `ReflectiveMutation<L>`: propose a new candidate from a parent
  and a reflective dataset, with an injected `LanguageModel`.
- `EvaluationCache`: skip redundant `(candidate, example)` rollouts.
- `AcceptanceCriterion` (`StrictImprovement` / `ImprovementOrEqual`) with
  reject reasons; `StopCondition` (`MaxMetricCalls`, `NoImprovement`,
  `FileStopper`, …).
- `OptimizationState` owns the loop: mode-aware parent selection, ledger-backed
  reflection, and gate-gated `Validate` acceptance.

The pieces GEPA lacks that clark adds: **mode-aware** parent selection
(explore/exploit/validate from `ResearchBias`), **ledger-backed reflection**
(the dossier is fed into the proposer so reflection reads the whole history),
and **gate-gated `Validate`** acceptance (inherited gates must pass in
addition to score improvement).

```sh
clark-autoresearch optimize \
  --seed '{"prompt":"You are a helpful assistant"}' \
  --eval-url http://127.0.0.1:8081/evaluate \
  --proposer-cmd 'your-proposer-script' \
  --trainset @train.json --valset @val.json \
  --max-metric-calls 150
```

See `examples/optimize_loop.rs` for a complete loop with mock adapter/proposer.

## Optional Similarity Feature (clark-hash)

Enable the `similarity` Cargo feature to add semantic retrieval backed by
[clark-hash](https://github.com/clark-labs-inc/clark-hash) (stateless sparse-JL
quantized sketches). The default build is unchanged and stays dependency-light.

```toml
[dependencies]
clark-autoresearch = { version = "0.1", features = ["similarity"] }
```

- `Embedder` (host): embed text into a `Vec<f32>`.
- `SemanticSketches`: a semantic index over the ledger's observations and
  hypotheses; `find_similar(query, k)` returns the most relevant past items
  (so reflection reuses receipts before rediscovering them).
- `SemanticCandidateCache`: reject a proposed candidate whose sketch is
  near-identical to an evaluated one (catches paraphrases the exact-match
  cache misses).
- `ResearchOpportunity.novelty` becomes `1.0 - max_similarity`.

See `examples/semantic_ledger.rs` (`cargo run --example semantic_ledger
--features similarity`).

## Design Notes

The crate intentionally keeps execution outside the core model. A downstream
agent runtime should own:

- provider calls,
- code editing,
- sandboxing,
- browser or network tools,
- authorization and egress controls,
- dashboard or streaming UI.

This keeps `clark-autoresearch` portable: the same state model works for prompt
optimization, benchmark sweeps, code-agent experiments, security research
orchestration, and evaluation pipelines.

## Inspired By

This project is a clean-room implementation inspired by the simplicity of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the
frontier/tree-search orientation of [evo-hq/evo](https://github.com/evo-hq/evo).
The optimization loop, per-instance Pareto frontier, acceptance criterion, and
reflective-mutation proposer are informed by
[GEPA](https://github.com/gepa-ai/gepa) ("improve_anything", arXiv:2507.19457);
clark reuses those mechanics but keeps execution behind host traits and adds
opportunity ranking, an evidence ledger, and inherited gates that GEPA lacks.
It does not vendor their code and is not affiliated with either project.

## Development

```sh
cargo fmt --all
cargo test --all-targets
cargo test --all-features --all-targets   # with the `similarity` feature
cargo build --no-default-features          # confirm the core stays dep-light
cargo clippy --all-targets -- -D warnings
```

## License

Apache-2.0.
