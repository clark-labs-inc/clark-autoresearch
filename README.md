# clark-autoresearch

[![crates.io](https://img.shields.io/crates/v/clark-autoresearch.svg)](https://crates.io/crates/clark-autoresearch)
[![docs.rs](https://img.shields.io/docsrs/clark-autoresearch)](https://docs.rs/clark-autoresearch)
[![License](https://img.shields.io/crates/l/clark-autoresearch.svg)](#license)
[![CI](https://github.com/clark-labs-inc/clark-autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/clark-labs-inc/clark-autoresearch/actions)

A small Rust library and CLI for building autonomous research loops:
experiment lineage, scalar metrics, gates, outcomes, frontier selection, and
opportunity ranking.

It does not run model providers, scanners, browsers, or shell tools for you.
Instead, it gives agent systems a stable state model for deciding what to try
next and for recording whether a trial was worth keeping.

## How It Works

The `optimize()` loop runs the GEPA shape — propose → minibatch-evaluate →
accept → full-evaluate → Pareto-update — with every provider call, sandbox,
and evaluator behind a host-implemented trait, so the crate stays
dependency-light and publishable:

- `ResearchAdapter` (host): evaluate a candidate on a batch and return
  scores + traces; build the reflective dataset.
- `Proposer` / `ReflectiveMutation<L>`: propose a new candidate from a parent
  and a reflective dataset, with an injected `LanguageModel`.
- `EvaluationCache`: skip redundant `(candidate, example)` rollouts.
- `AcceptanceCriterion` (`StrictImprovement` / `ImprovementOrEqual`) with
  reject reasons; `StopCondition` (`MaxMetricCalls`, `NoImprovement`,
  `FileStopper`, …).

Beyond GEPA, the crate adds:

- **Mode-aware parent selection** — explore/exploit/validate modes from
  `ResearchBias` pick the parent.
- **Ledger-backed reflection** — the evidence dossier is fed into the proposer
  so reflection reads the whole history, not just the last minibatch.
- **Gate-gated `Validate` acceptance** — inherited gates must pass in addition
  to score improvement (GEPA has no gate concept).

For simpler flows, the experiment graph alone tracks lineage:
`ExperimentGraph` (root/pending/active/evaluated/committed/discarded/
failed/pruned states), `TrialOutcome` with per-task and per-objective scores,
`ResearchPolicy` with gates and attempt budgets, and deterministic
`FrontierStrategy` ranking (`arg-max`, `top-k`, `epsilon-greedy`, `softmax`,
or Pareto).

```sh
clark-autoresearch optimize \
  --seed '{"prompt":"You are a helpful assistant"}' \
  --eval-url http://127.0.0.1:8081/evaluate \
  --proposer-cmd 'your-proposer-script' \
  --trainset @train.json --valset @val.json \
  --max-metric-calls 150
```

## Quick Start

Install the CLI from crates.io:

```sh
cargo install clark-autoresearch
```

Or depend on the library:

```sh
cargo add clark-autoresearch
```

From a local checkout instead: `cargo install --path .` (or
`cargo run -- --help`).

Initialize a state file:

```sh
clark-autoresearch init \
  --metric accuracy \
  --direction maximize \
  --gate test="cargo nextest run"
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

### Opportunity Ranking

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

## Library Usage

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

### Core Types

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

## Examples

- [`examples/simple_loop.rs`](examples/simple_loop.rs) — complete library walk
  through the experiment graph.
- [`examples/optimize_loop.rs`](examples/optimize_loop.rs) — the GEPA-shaped
  loop with a mock adapter/proposer.
- [`examples/semantic_ledger.rs`](examples/semantic_ledger.rs) — ledger-backed
  semantic retrieval (needs the `similarity` feature).
- [`examples/openrouter/`](examples/openrouter/README.md) — a live `optimize`
  run against a real LLM on OpenRouter.

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

## Optional Similarity Feature (clark-hash)

Enable the `similarity` Cargo feature to add semantic retrieval backed by
[clark-hash](https://github.com/clark-labs-inc/clark-hash) (stateless sparse-JL
quantized sketches). The default build is unchanged and stays dependency-light.

```toml
[dependencies]
clark-autoresearch = { version = "0.2", features = ["similarity"] }
```

- `Embedder` (host): embed text into a `Vec<f32>`.
- `SemanticSketches`: a semantic index over the ledger's observations and
  hypotheses; `find_similar(query, k)` returns the most relevant past items
  (so reflection reuses receipts before rediscovering them).
- `SemanticCandidateCache`: reject a proposed candidate whose sketch is
  near-identical to an evaluated one (catches paraphrases the exact-match
  cache misses).
- `ResearchOpportunity.novelty` becomes `1.0 - max_similarity`.

```sh
cargo run --example semantic_ledger --features similarity
```

## Development

```sh
cargo fmt --all
cargo install cargo-nextest --version 0.9.143 --locked
cargo nextest run --all-targets
cargo nextest run --all-features --all-targets   # with the `similarity` feature
cargo build --no-default-features          # confirm the core stays dep-light
cargo clippy --all-targets -- -D warnings
```

Keep the library small and deterministic — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Inspired By

This project is a clean-room implementation inspired by the simplicity of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the
frontier/tree-search orientation of [evo-hq/evo](https://github.com/evo-hq/evo).
The optimization loop, per-instance Pareto frontier, acceptance criterion, and
reflective-mutation proposer are informed by
[GEPA](https://github.com/gepa-ai/gepa) ("improve_anything", arXiv:2507.19457);
this crate reuses those mechanics but keeps execution behind host traits and
adds opportunity ranking, an evidence ledger, and inherited gates that GEPA
lacks. It does not vendor their code and is not affiliated with either project.

## License

Apache-2.0. See [LICENSE](LICENSE).
