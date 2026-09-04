# Examples

Run the library example:

```sh
cargo run --example simple_loop
```

Try the CLI workflow:

```sh
cargo run -- init --metric accuracy --direction maximize --force
cargo run -- spawn "try a shorter prompt with explicit checks" --mode explore
cargo run -- record exp_0000 0.82 --status passed --summary "accuracy improved"
cargo run -- commit exp_0000 --commit demo
cargo run -- frontier --strategy top-k --k 3
```

Rank generic research opportunities from JSON:

```sh
cargo run -- opportunity-rank examples/opportunities.json
```
