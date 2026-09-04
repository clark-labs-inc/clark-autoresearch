# Contributing

Thanks for improving `clark-autoresearch`.

## Development

```sh
cargo fmt --all
cargo install cargo-nextest --version 0.9.143 --locked
cargo nextest run --all-targets
cargo clippy --all-targets -- -D warnings
```

Keep the library small and deterministic. The crate should model research state,
frontier ranking, metrics, and policy; execution engines, model providers,
scanners, and application-specific tools should live in downstream projects.

## Pull Requests

- Include focused tests for behavior changes.
- Keep public API changes explicit in the README or examples.
- Avoid adding network or provider dependencies to the core library.
- Do not vendor code from projects that inspired the design.
