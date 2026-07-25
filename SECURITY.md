# Security Policy

`clark-autoresearch` is an orchestration and ranking library. It does not ship
network scanners, exploit payloads, model-provider clients, or target-access
logic.

If you find a vulnerability in this crate, please report it privately to the
maintainers before public disclosure. Include:

- affected version or commit,
- a minimal reproduction,
- impact and suggested mitigation if known.

Downstream applications are responsible for authorization, sandboxing, egress
controls, and safe execution of any tools they wire into their research loops.
