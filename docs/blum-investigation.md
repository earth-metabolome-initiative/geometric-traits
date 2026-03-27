# Blum Correctness And Sparse Benchmark Investigation

## Scope

This note records:

- the Blum correctness fixes that were made during the Karp-Sipser work
- the regression tests that now cover those failures
- the benchmark reruns for the sparse Blum case
- the current understanding of the large slowdown on `matching/Blum/n=500_d=6`

This document is intentionally narrow. It is not a full algorithm design note.

## Current Status

- `blum()` is solving the known reduced counterexamples directly. There is no `micali_vazirani()` fallback in the public Blum path.
- The Karp-Sipser `Degree1` Blum wrapper failures were not a separate KS bug. They were manifestations of the same base Blum bugs on the unreduced kernel.
- The large benchmark regression is real, but it is not uniform across sparse inputs. It is concentrated on the benchmark graph `n=500, d=6`.

## Correctness Work Completed

### Regressions Added

The following tests now pin down the failures that were found during fuzzing and reduction:

- [`tests/test_blum.rs:200`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L200): invalid non-edge from a reduced degree-1 kernel case
- [`tests/test_blum.rs:226`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L226): small explicit plain-Blum size mismatch
- [`tests/test_blum.rs:262`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L262): reused-vertex corpus-derived regression
- [`tests/test_blum.rs:291`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L291): second invalid non-edge corpus-derived regression
- [`tests/test_blum.rs:457`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L457): random reduced invalid-matching regression
- [`tests/test_blum.rs:482`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L482): random reduced repeated-vertex regression
- [`tests/test_blum.rs:516`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L516): random reduced size-mismatch regression
- [`tests/test_blum.rs:525`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_blum.rs#L525): phase-progression stall before the maximum matching
- [`tests/test_karp_sipser.rs:150`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_karp_sipser.rs#L150): corpus replay for the `Degree1` Blum wrapper invalid-kernel case
- [`tests/test_karp_sipser.rs:174`](/home/luca/github/geometric-traits3/geometric-traits/tests/test_karp_sipser.rs#L174): small explicit `Degree1` Blum-wrapper size mismatch

### Main Blum Fixes

The main algorithm changes live in [`src/traits/algorithms/blum/inner.rs`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs):

- MBFS bridge fix at [`inner.rs:313`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs#L313)
  - when the two bridge walks meet at a graph node, its twin now receives the missing second-level assignment and MBFS continues from there
  - this fixed the class of failures where MBFS stopped one edge before a valid augmenting path
- Historical-label fallback in MDFS at [`inner.rs:646`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs#L646)
  - if the direct `L[w]` entry has been cleared, MDFS can still follow the representative chain through `find_rep(w)`
- Source-label cleanup refactor at [`inner.rs:671`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs#L671)
  - moved into `clear_l_sources`
- Reverse-order backward search at [`inner.rs:719`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs#L719)
  - `R` and `E` are now scanned in reverse insertion order during backward search
- Recursive reconstruction fix at [`inner.rs:905`](/home/luca/github/geometric-traits3/geometric-traits/src/traits/algorithms/blum/inner.rs#L905)
  - nested expanded segments are reconstructed recursively instead of being flattened incorrectly

### Karp-Sipser Invariant Helper Fix

The KS invariant helper previously blamed KS wrappers for failures that actually came from plain Blum. That is fixed in [`src/test_utils.rs:319`](/home/luca/github/geometric-traits3/geometric-traits/src/test_utils.rs#L319):

- it now checks plain `graph.blum()` against Blossom first at [`src/test_utils.rs:323`](/home/luca/github/geometric-traits3/geometric-traits/src/test_utils.rs#L323)
- only after that does it compare the KS wrappers at [`src/test_utils.rs:331`](/home/luca/github/geometric-traits3/geometric-traits/src/test_utils.rs#L331)

This prevents a future base-Blum regression from being misreported as a KS wrapper bug.

## Benchmark Reruns

### Sparse Blum Case

The benchmark point was rerun on:

- the current working tree
- a detached baseline worktree at the same commit but without the local Blum changes

Command:

```bash
cargo bench --bench matching -- --noplot 'matching/Blum/n=500_d=6'
```

Results:

| Case | Median time |
| --- | ---: |
| Current tree | `984.93 µs` |
| Baseline worktree | `454.81 µs` |

This is about a `2.17x` slowdown. It is reproducible.

### Nearby Sparse Points

To check whether this was a broad sparse slowdown or a graph-specific pathology, the nearby points were rerun too.

Command:

```bash
cargo bench --bench matching -- --noplot 'matching/Blum/n=200_d=6'
cargo bench --bench matching -- --noplot 'matching/Blum/n=1000_d=6'
```

Results:

| Benchmark | Current | Baseline | Interpretation |
| --- | ---: | ---: | --- |
| `matching/Blum/n=200_d=6` | `118.06 µs` | `116.78 µs` | basically unchanged |
| `matching/Blum/n=500_d=6` | `984.93 µs` | `454.81 µs` | large regression |
| `matching/Blum/n=1000_d=6` | `928.70 µs` | `907.69 µs` | only slight slowdown |

This means the regression is not a simple global factor on sparse graphs. It is concentrated on the specific `n=500, d=6` benchmark graph.

### Earlier Dense Comparison

An earlier dense comparison showed only a small difference:

| Benchmark | Current | Baseline |
| --- | ---: | ---: |
| `matching_dense/Blum/n=200_d=100` | `688.84 µs` | `675.06 µs` |

That result is included here for context. The primary problem remains the sparse `n=500, d=6` graph.

## Temporary Probe Findings

A temporary documented local probe was used to inspect Blum's internal activity on the benchmark graph family. The probe has been removed after use; no probe code or stats hooks remain in the tree.

The probe measured the exact benchmark graph generator on `n = 200, 500, 1000` with `avg_degree = 6`.

Observed counters:

| Metric | `n=200` | `n=500` | `n=1000` |
| --- | ---: | ---: | ---: |
| `phase_count` | `3` | `6` | `4` |
| `multi_path_found` | `100` | `248` | `500` |
| `fallback_runs` | `0` | `2` | `0` |
| `step_calls` | `566` | `6855` | `4391` |
| `edges_seen` | `1955` | `25307` | `16092` |
| `backward_search_calls` | `88` | `1768` | `962` |
| `backward_r_scans` | `0` | `28` | `0` |
| `backward_e_scans` | `4` | `1000` | `85` |
| `find_rep_calls` | `3` | `1073` | `76` |
| `find_rep_steps` | `0` | `1634` | `0` |
| `direct_label_hits` | `0` | `3` | `0` |
| `rep_label_hits` | `0` | `0` | `0` |
| `e_push_ink` | `0` | `751` | `0` |
| `e_push_unlabeled` | `34` | `2942` | `575` |
| `r_push` | `0` | `547` | `3` |
| `clear_l_sources` | `0` | `18` | `0` |

Interpretation:

- the slow graph triggers `2` full single-path fallback runs
- it does far more backward-search work than either neighboring point
- it calls `find_rep()` more than a thousand times
- despite those calls, it never records a useful `rep_label_hit`
- `E` grows much more aggressively on this graph than on the nearby points

This strongly suggests a graph-specific blow-up in MDFS label management and backward search, not a broad cost increase in MBFS.

## Resolution

### Root Cause

Systematic bisection showed that the **MBFS meet-twin fix** was the sole cause of the ~2.17x regression. All other changes (historical-label fallback, E push condition, reverse-order backward search, recursive reconstruction) had zero measurable performance impact.

The initial hypothesis that MDFS label management and backward search were responsible was wrong — the probe counters blew up as a *secondary effect* of the changed MBFS level structure, not as a primary cause.

### Why the Inline Fix Was Expensive

When the meet-twin fix ran **inline during bridge processing**, it:

1. Assigned a level to the twin and scanned its neighbors via `mbfs_scan`
2. The scan discovered new nodes and appended new bridges to the `bridges` vector
3. These new bridges were processed in subsequent iterations of the `while bi < bridges.len()` loop
4. Each new bridge could trigger its own meet-twin fix, causing more scans and more bridges
5. This positive feedback loop cascaded until no more unleveled twins remained

On the `n=500, d=6` graph, this cascade was extremely expensive.

### The Fix: Post-Bridge Twin Pass

Moving the twin fix to a single pass **after all bridges are processed** eliminates the cascading bridge generation:

1. All bridges are processed normally (identical to baseline)
2. After the bridge loop, scan all graph nodes for unleveled twins
3. Assign levels and run one BFS scan for all discovered twins

This breaks the cascade because new bridges from the post-bridge scan are never fed back into the bridge processing loop. The MDFS fallback handles any augmenting paths that the slightly less complete level assignment might miss.

### Additional Optimizations Applied

- **E push condition** reverted from `self.l[w].is_none()` to the tighter `!self.l_ever[w]` (no measurable impact, but theoretically tighter)
- **Path-halving compression** added to `find_rep()` (removes the cycle guard, no measurable impact on this graph)

### Benchmark Results After Fix

| Benchmark | Baseline | After Fix | Status |
| --- | ---: | ---: | --- |
| `matching/Blum/n=200_d=6` | `116.78 µs` | `115.92 µs` | matched |
| `matching/Blum/n=500_d=6` | `454.81 µs` | `456.18 µs` | **fixed** (was `1.04 ms`) |
| `matching/Blum/n=1000_d=6` | `907.69 µs` | `924.92 µs` | matched |
| `matching_dense/Blum/n=200_d=100` | `675.06 µs` | `659.16 µs` | matched |

All 66 tests pass (53 Blum + 13 Karp-Sipser). No fallback to another matcher was reintroduced.
