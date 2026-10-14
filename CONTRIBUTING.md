# Contributing

Contributions are welcome. This document describes the design philosophy behind the crate so that new code fits the existing conventions.

## Design Philosophy

Algorithms are trait-based. They are implemented generic over traits such as `BipartiteGraph` and `MonopartiteGraph`, so they work with any backing data structure that implements the required interface, including matrices, CSR, and adjacency lists. New algorithms should follow the same pattern rather than hard-coding a concrete representation.

Correctness is a primary focus. Key algorithms are continuously fuzzed with libFuzzer on ClusterFuzzLite to harden them against edge cases and to verify their invariants, and many are cross-checked against independent references or ground-truth corpora. New algorithms should come with a fuzz harness under `fuzz/fuzz_targets` and, where a reference implementation exists, a differential or ground-truth test.

The crate is `no_std` compatible. The core traits and several implementations work in `no_std` environments, and feature flags enable `std` or `alloc` only when necessary. Keep new code `no_std` friendly, and gate any allocation behind the `alloc` feature.

## Checks

Run formatting, linting, tests and documentation before opening a pull request.

```bash
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features --lib --tests -- -D warnings
cargo clippy --no-default-features --features alloc --lib --tests -- -D warnings
cargo clippy --no-default-features --features "alloc hashbrown" --lib --tests -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

Use comparison assertions for string emptiness checks and for slices, arrays or vectors with `Debug` and `PartialEq` elements, so failures show the value.

## Coverage

Coverage includes all targets and doctests, using nightly Rust with `llvm-tools-preview` and `cargo-llvm-cov`.

```bash
rustup component add --toolchain nightly llvm-tools-preview
cargo install cargo-llvm-cov
cargo +nightly llvm-cov --workspace --all-features --all-targets --no-report
cargo +nightly llvm-cov --workspace --all-features --doc --no-clean --lcov --output-path target/coverage.lcov
```

## Mutation audits

The `Mutation audit` workflow runs weekly and on manual dispatch, outside pull-request and release gates. Scheduled runs rotate through algorithm files and their shards. Each batch tests at most 20 candidates against all unit and integration targets with `--all-features`, with one worker, at most eight CPUs and a 45-minute audit budget.

Run a specific file and shard locally with `cargo-mutants 27.1.0`.

```bash
SOURCE_FILE=src/impls/sorted_array.rs SHARD_INDEX=2 BATCH_SIZE=1 \
  bash .github/scripts/mutation_audit.sh
```

`SHARD_INDEX` is one-based. `BATCH_SIZE` accepts 1 through 20. `WALL_CLOCK_LIMIT`, `MUTANTS_BUILD_TIMEOUT` and `MUTANTS_TEST_TIMEOUT` set limits in seconds. The helper disables compiler wrappers so affinity and priority limits also constrain compiler processes.

Each invocation writes `mutation-audit-summary.md` and uses its own `mutation-audit.*/` report directory.

The job summary and artifact retain scope, completed candidates, logs, diffs and outcomes. Surviving candidates and per-mutant timeouts are advisory and require classification. Baseline, infrastructure and exhausted-budget failures fail the audit. Partial results carry an incomplete status and no mutation-score threshold applies.
