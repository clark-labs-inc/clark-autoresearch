# OpenRouter `optimize` example

A complete, host-side harness that drives the
`clark-autoresearch optimize` loop against a **real LLM** served by
[OpenRouter](https://openrouter.ai) — without bringing any provider, network,
or sandbox code into the crate itself. The crate owns the loop body; these two
Python scripts are the *host* implementation of the two traits the loop calls:

| Trait (in crate)            | Host script here            | Role                                    |
|-----------------------------|-----------------------------|-----------------------------------------|
| `ResearchAdapter::evaluate` | `or_eval_server.py`         | Score a candidate prompt on a batch     |
| `Proposer::propose`         | `or_proposer.py`            | Propose an improved prompt              |

## The problem

Prompt optimization for a math Q&A assistant. The **candidate** is a system
prompt (a one-key `{"prompt": "..."}` mapping). The assistant must wrap its
final answer in `<answer>` tags, e.g. `<answer>42</answer>`. An answer is
scored `1.0` iff it contains `<answer>N</answer>` with `N` equal to the
expected value, else `0.0`.

This is deliberately **prompt-sensitive**: a weak seed prompt ("be helpful")
makes the model emit bare numbers (score 0), while a prompt that instructs
`<answer>` tags yields tagged answers (score 1) — so the optimization loop has
a real signal to climb, and strict-improvement acceptance visibly fires.

## Prerequisites

- A built CLI: `cargo build --release` (use `target/release/clark-autoresearch`).
- An OpenRouter API key. Provide it **one** of these ways (the scripts never
  print it):
  - `export OPENROUTER_API_KEY=sk-or-...` in the shell that runs the server and
    proposer, **or**
  - write it to `~/.openrouter_api_key` (one line).
- Python 3 (stdlib only — no pip installs).

## 1. Validate the wiring offline (no key, no network)

Both scripts take `--dry-run`: the eval server simulates a prompt-sensitive
model, and the proposer returns a canned improved prompt. This exercises the
full CLI → HTTP eval adapter → command proposer → propose/eval/accept/commit
path with zero cost.

```sh
# terminal A — simulated eval server
python3 examples/openrouter/or_eval_server.py --port 8081 --dry-run

# terminal B — run the loop
cargo run -- optimize \
  --seed '{"prompt":"You are a helpful assistant. Answer the question concisely."}' \
  --eval-url http://127.0.0.1:8081/evaluate \
  --proposer-cmd 'python3 examples/openrouter/or_proposer.py --dry-run' \
  --trainset @examples/openrouter/or_train.json \
  --valset   @examples/openrouter/or_val.json \
  --minibatch-size 2 --max-metric-calls 24 --max-iterations 8 \
  --acceptance strict
# expect: best score climbs 0.0 -> 1.0 on the first accepted iteration
```

## 2. Live run with OpenRouter (`qwen/qwen3.5-flash`)

```sh
# terminal A — live eval server (reads the key from env or ~/.openrouter_api_key)
python3 examples/openrouter/or_eval_server.py --port 8081 --model qwen/qwen3.5-flash

# terminal B — live proposer + loop
python3 examples/openrouter/or_proposer.py --dry-run >/dev/null  # smoke test parse
cargo run -- optimize \
  --seed '{"prompt":"You are a helpful assistant. Answer the question concisely."}' \
  --eval-url http://127.0.0.1:8081/evaluate \
  --proposer-cmd 'python3 examples/openrouter/or_proposer.py --model qwen/qwen3.5-flash' \
  --trainset @examples/openrouter/or_train.json \
  --valset   @examples/openrouter/or_val.json \
  --minibatch-size 2 --max-metric-calls 24 --max-iterations 10 \
  --acceptance strict
```

`--max-iterations` is a hard cap: it always terminates the loop even if the
proposer converges to a fixed prompt (cached evals cost zero metric calls, so
`--max-metric-calls` alone cannot bound that case).

## Files

- `or_eval_server.py` — HTTP eval server (implements `ResearchAdapter`).
- `or_proposer.py` — command proposer (implements `Proposer`).
- `or_train.json` / `or_val.json` — 3 training + 4 validation arithmetic items.

No secrets are committed. The scripts read the key from the environment or
`~/.openrouter_api_key` only.