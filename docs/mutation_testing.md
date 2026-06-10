# Mutation testing roadmap

This document tracks the mutation-testing campaign across every algorithm in the
crate. Mutation testing (via [cargo-mutants](https://mutants.rs)) introduces small
deliberate faults into the source one at a time and re-runs the test suite for
each. A mutant that survives marks a place the tests cannot tell correct code from
broken code, which is a test blind spot rather than necessarily a real bug.

## How to run it

The configuration lives in `.cargo/mutants.toml` (release profile, unit and
integration tests, fuzz and bench targets excluded). To work a single module,
pass both the top-level file and its submodule subtree, since some modules are
directories (for example `assignment/`, `blossom_v/`, `canon/`):

```
cargo mutants -j 8 \
  --file 'src/traits/algorithms/<module>.rs' \
  --file 'src/traits/algorithms/<module>/**/*.rs'
```

For the large modules, split the work into shards across several runs:

```
cargo mutants -j 8 --file 'src/traits/algorithms/<module>/**/*.rs' --shard 1/4
```

The pull-request CI workflow (`.github/workflows/mutants.yml`) runs
`cargo mutants --in-diff` on each PR's changed lines only. It is advisory: it
reports surviving mutants as an artifact but never fails the check, because
equivalent mutants can legitimately survive.

## Definition of done for a module

1. Run cargo-mutants over the module.
2. Triage each surviving mutant into one of two buckets:
   - a real test gap, which gets a new test that kills it;
   - an equivalent or robustness-equivalent mutant, which no input can
     distinguish (common in greedy or iterative numerical code). Classify it, and
     for the provably-equivalent ones add a `// mutants: skip` with a one-line
     reason.
3. Re-run, record the score, and tick the box.

## Status legend

- `todo`: not yet run.
- `wip`: run, triage or fixes in progress.
- `done`: every survivor is either killed or classified as equivalent.

## Progress

Mutant counts are a snapshot from the time of writing and drift as the code
changes. The score is caught over viable mutants.

| Module | Mutants | Status | Score | Notes |
| --- | ---: | --- | --- | --- |
| assignment | 38 | done | ~97% | One equivalent survivor: `bfs` `<` to `<=` only expands BFS layers at distance >= `null_distance`, which the DFS never follows. |
| biconnected_components | 173 | done | 100% | Most mutants are unviable (heavy generic bounds); the one survivor was the indexed `vertex_biconnected_component` accessor, now cross-checked against the iterator form. |
| bipartite_detection | 20 | done | 100% | Added a direct test for the `pub(crate)` `sparse_matrix_bipartite_coloring` helper; one mutant is detected via timeout (it makes the coloring BFS loop forever). |
| blossom | 24 | todo | | |
| blossom_v | 1960 | todo | | Largest module; shard the run. |
| cactus_detection | 27 | todo | | |
| canon | 1193 | todo | | Large; shard the run. |
| chordal_detection | 54 | todo | | |
| clique_ranking | 41 | todo | | |
| connected_components | 40 | todo | | |
| cycle_detection | 8 | todo | | |
| delta_y_exchange | 13 | todo | | |
| diameter | 49 | todo | | |
| directed_community | 223 | done | ~83% | Survivors are equivalent or robustness-equivalent (greedy optimizer). |
| directed_leiden | 4 | done | 100% | Thin trait over the shared driver. |
| directed_louvain | 4 | done | 100% | Thin trait over the shared driver. |
| directed_modularity | 41 | todo | | |
| essential_cycles | 38 | todo | | |
| floyd_warshall | 45 | todo | | |
| forceatlas2 | 496 | todo | | Large; shard the run. |
| gabow_1976 | 46 | todo | | |
| graph_similarities | 191 | todo | | |
| gth | 144 | todo | | |
| information_content | 17 | todo | | |
| initial_cycle_families | 84 | todo | | |
| jacobi | 276 | todo | | |
| johnson | 41 | todo | | |
| k23_homeomorph_detection | 4 | todo | | |
| k33_homeomorph_detection | 4 | todo | | |
| k4_homeomorph_detection | 4 | todo | | |
| kahn | 12 | todo | | |
| kocay | 222 | todo | | |
| labeled_line_graph | 16 | todo | | |
| leicht_newman | 237 | todo | | |
| leiden | 137 | todo | | |
| lin | 24 | todo | | |
| line_graph | 16 | todo | | |
| louvain | 37 | todo | | |
| matching_utils | 5 | todo | | Shared helper. |
| maximum_clique | 464 | todo | | Large; shard the run. |
| mces | 389 | todo | | Large; shard the run. |
| mds | 190 | todo | | |
| micali_vazirani | 178 | todo | | |
| minimum_cost_balanced_flow | 182 | todo | | |
| minimum_cycle_basis | 207 | todo | | |
| modularity | 155 | todo | | Shared community-detection internals. |
| modular_product | 83 | todo | | |
| node_classification | 9 | todo | | Shared helper. |
| node_ordering | 681 | todo | | Large; shard the run. |
| outerplanarity_detection | 2 | todo | | |
| pairwise_bfs | 17 | todo | | |
| pairwise_dijkstra | 27 | todo | | |
| planarity_detection | 900 | todo | | Large; shard the run. |
| randomized_graphs | 632 | todo | | Large; shard the run. |
| relevant_cycles | 32 | todo | | |
| resnik | 28 | todo | | |
| root_nodes | 3 | todo | | |
| simple_path | 9 | todo | | |
| singleton_nodes | 5 | todo | | |
| sink_nodes | 4 | todo | | |
| tarjan | 19 | todo | | |
| tree_detection | 16 | todo | | |
| undirected_modularity | 7 | todo | | |
| vertex_match_inference | 19 | todo | | Shared helper. |
| vf2 | 325 | todo | | |
| weighted_assignment | 323 | todo | | |
| weisfeiler_lehman | 27 | todo | | |
| wu_palmer | 50 | todo | | |
