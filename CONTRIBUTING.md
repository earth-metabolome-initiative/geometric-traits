# Contributing

Contributions are welcome. This document describes the design philosophy behind the crate so that new code fits the existing conventions.

## Design Philosophy

Algorithms are trait-based. They are implemented generic over traits such as `BipartiteGraph` and `MonopartiteGraph`, so they work with any backing data structure that implements the required interface, including matrices, CSR, and adjacency lists. New algorithms should follow the same pattern rather than hard-coding a concrete representation.

Correctness is a primary focus. Key algorithms are continuously fuzzed with `honggfuzz` to harden them against edge cases and to verify their invariants, and many are cross-checked against independent references or ground-truth corpora. New algorithms should come with a fuzz harness under `fuzz/fuzz_targets` and, where a reference implementation exists, a differential or ground-truth test.

The crate is `no_std` compatible. The core traits and several implementations work in `no_std` environments, and feature flags enable `std` or `alloc` only when necessary. Keep new code `no_std` friendly, and gate any allocation behind the `alloc` feature.

## Checks

Before opening a pull request, please run the same gates the CI enforces: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, and `cargo doc --no-deps --document-private-items` with `RUSTDOCFLAGS=-D warnings`.
