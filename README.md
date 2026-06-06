# clark-autoresearch

`clark-autoresearch` is a clean-room Rust crate for autonomous research loops:
experiment graphs, metric comparison, gates, frontier selection, and BOSS-style
security opportunity ranking.

## Reference Takeaways

It is designed to be open-sourceable and does not vendor code from the reference
projects reviewed for this crate:

- `karpathy/autoresearch` at `228791fb499afffb54b46200aca536f79142f117`.
  Useful idea: fixed evaluator, one editable target, scalar metric, keep/discard
  loop.
- `evo-hq/evo` at `0090ce91832070ae641d0ae516254eba13c3691a`.
  Useful idea: tree search over experiments, inherited gates, per-task outcomes,
  and frontier policies.

For BOSS, this crate intentionally models orchestration and ranking only. It does
not include scanners, exploit payloads, or target access logic; authorization and
tool egress controls stay in the BOSS runtime.

## Public Surface

- `ExperimentGraph`: append-only experiment lineage with committed/evaluated/
  discarded/pruned states.
- `Metric` and `TrialOutcome`: scalar score plus optional per-task scores.
- `Gate`: inherited pre/post checks that can be collected along a lineage path.
- `FrontierStrategy`: argmax, top-k, epsilon-greedy, softmax, and Pareto-per-task
  frontier ranking.
- `BossOpportunity`: BOSS-facing security opportunity scoring that separates
  exploration, exploitation, and proof validation without importing BOSS types.
  Ranked hints include a dispatch class, not a concrete BOSS agent name, so the
  runtime owns tool and prompt binding.

## BOSS Integration Path

The first integration step should be non-invasive: derive `BossOpportunity`
records from BOSS graph nodes, rank them with `rank_boss_opportunities`, and use
the top hint to bias `planner::build_planner_prompt` or `runner::select_agent`.

The second step can move hypothesis validation into `ExperimentGraph`: each
CodeAct run becomes an experiment, confirmed findings become committed nodes,
false positives become discarded nodes, and BOSS gates keep scope and proof
quality explicit.
