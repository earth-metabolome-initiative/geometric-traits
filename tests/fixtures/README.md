# Ground-Truth Modularity Fixtures

`modularity_ground_truth.json` was generated offline from external references:

- Louvain: `python-igraph` (`Graph.community_multilevel`)
- Leiden: `leidenalg` (`find_partition(..., ModularityVertexPartition)`)

The fixture contains deterministic graph cases plus expected partitions/modularity for both algorithms.

# Directed Modularity Ground-Truth Fixtures

`directed_modularity_ground_truth.json.gz` is the checked-in NetworkX oracle for the directed (Leicht-Newman 2008) modularity metric and the Leicht-Newman spectral community detector.

It stores, per directed weighted case:

- the directed edge list `[source, destination, weight]`
- several named partitions (singletons, single community, planted when
  applicable, and two random labelings), each with the directed modularity
  reported by NetworkX `community.modularity` (the directed branch is exactly
  the Leicht-Newman generalization)
- a `reference` partition from NetworkX `greedy_modularity_communities` with its
  directed modularity and the NetworkX wall-clock time in nanoseconds (used as a
  quality floor for the detector and as the reference timing in the benchmark)

The cases span tiny hand-checkable graphs (single edge, bidirectional pair, two directed cycles, self-loops), weakly-linked directed triangles, weighted two-block graphs, several planted directed block models (16 to 30 nodes, 2 to 4 communities), and a structureless random digraph.

It is a one-shot recording from NetworkX `3.3`. The generator is not part of the repository, and the fixture stores `networkx_version` for provenance.

It is consumed by `tests/test_directed_modularity_ground_truth.rs`, `tests/test_leicht_newman.rs`, and `benches/leicht_newman.rs`.

# Blossom V Ground-Truth Fixtures

The large Blossom V ground-truth corpora are local-only inputs. They are not tracked in git.

If you keep a local shard directory for manual sweeps, point `BLOSSOM_V_GROUND_TRUTH_SOURCE` at it.

# Planarity Ground-Truth Fixtures

`planarity_ground_truth_100k.json.gz` is the checked-in Boyer reference corpus for planarity and outerplanarity validation.

The current path is:

```text
tests/fixtures/planarity_ground_truth_100k.json.gz
```

It is consumed by `tests/test_planarity_reference_corpus.rs`.

# K23 Homeomorph Ground-Truth Fixtures

`k23_homeomorph_ground_truth_100k.json.gz` is the checked-in Boyer reference corpus for `K_{2,3}` homeomorph validation against `planarity -2`.

The current path is:

```text
tests/fixtures/k23_homeomorph_ground_truth_100k.json.gz
```

It is consumed by `tests/test_k23_homeomorph_reference_corpus.rs`.

# K33 Homeomorph Ground-Truth Fixtures

`k33_homeomorph_ground_truth_100k.json.gz` is the checked-in Boyer reference corpus for `K_{3,3}` homeomorph validation against `planarity -3`.

The current path is:

```text
tests/fixtures/k33_homeomorph_ground_truth_100k.json.gz
```

It is consumed by `tests/test_k33_homeomorph_reference_corpus.rs`.

# Combined Topological Validity Ground-Truth Fixtures

`topological_validity_ground_truth_1m_v4.json.gz` is the checked-in Boyer reference corpus for the combined boolean validity surface:

- planarity
- outerplanarity
- `K_{2,3}` homeomorph detection against `planarity -2`
- `K_{3,3}` homeomorph detection against `planarity -3`
- `K_4` homeomorph detection against `planarity -4`

The current path is:

```text
tests/fixtures/topological_validity_ground_truth_1m_v4.json.gz
```

# VF2 Ground-Truth Fixtures

`vf2_networkx_fixture_suite.json.gz` is the checked-in `NetworkX` oracle suite for the generic VF2 core.

It merges the old split VF2 corpora into one gzip-compressed JSON file with:

- large boolean structural cases
- equality-labeled boolean cases
- exact-embedding structural cases
- exact-embedding equality-labeled cases
- explicit monomorphism cases alongside isomorphism and subgraph modes
- directed and undirected coverage
- self-loop coverage
- per-case stored `NetworkX` timing in nanoseconds

This fixture is consumed by `tests/test_vf2_fixture_suite.rs` and `benches/vf2.rs`.

# Node Ordering Ground-Truth Fixtures

`node_ordering_ground_truth.json.gz` is the checked-in 10k-case node-ordering oracle corpus for graph-level smallest-last / degeneracy ordering and the related Welsh-Powell, DSATUR, `2.2` degeneracy-with-degree, `3.1` PageRank, Katz-centrality, and betweenness-centrality, closeness-centrality, triangle-count, local-clustering, BFS-from-max-degree, and DFS-from-max-degree orders.

The catalog's "smallest-last coloring" entry is represented by the same smallest-last / degeneracy order already stored here. No separate oracle field is needed for it at the node-ordering layer.

It stores, per graph case:

- the normalized undirected edge list
- the raw `NetworkX` `strategy_smallest_last` output
- a deterministic canonical smallest-last order retained for provenance
- node-id-indexed `NetworkX` core numbers
- the deterministic `2.2` order
  `(core_number desc, degree desc, node_id asc)`
- the deterministic Welsh-Powell order
  `(degree desc, node_id asc)`
- the deterministic DSATUR order
  `(saturation_degree desc, degree desc, node_id asc)`
- the deterministic BFS-from-max-degree order
  `(seed/restart by out_degree desc then node_id asc; neighbors in node_id asc)`
- the deterministic DFS-from-max-degree order
  `(seed/restart by out_degree desc then node_id asc; neighbors in node_id asc)`
- node-id-indexed `NetworkX` PageRank scores
- the deterministic `3.1` order
  `(pagerank desc, node_id asc)`
- node-id-indexed `NetworkX` Katz centrality scores
- the deterministic Katz order
  `(katz desc, node_id asc)`
- node-id-indexed `NetworkX` betweenness centrality scores
- the deterministic betweenness order
  `(betweenness desc, node_id asc)`
- node-id-indexed `NetworkX` closeness centrality scores
- the deterministic closeness order
  `(closeness desc, node_id asc)`
- node-id-indexed `NetworkX` triangle counts
- the deterministic triangle-count order
  `(triangle_count desc, node_id asc)`
- node-id-indexed `NetworkX` local clustering coefficients
- the deterministic local-clustering order
  `(local_clustering desc, node_id asc)`

The canonical order uses the explicit removal tie-break `(current_degree, node_id)`. It is stored as an auxiliary reference only. The crate's `DegeneracySorter` implements the linear Matula-Beck smallest-last algorithm, so the Rust tests and benchmarks validate the smallest-last invariant rather than exact tie-for-tie parity with that auxiliary order.

The PageRank oracle stores a per-case parameter triple `(pagerank_alpha, pagerank_max_iter, pagerank_tol)`, drawn from a small fixed set of `NetworkX`-compatible configurations on undirected graphs. Each case still uses uniform initialization and uniform dangling redistribution. Scores are rounded to 12 decimal places before the stored `pagerank_descending` order is derived, so floating near-ties have a stable node-id fallback.

The Katz oracle stores a per-case parameter tuple `(katz_alpha, katz_beta, katz_max_iter, katz_tol, katz_normalized)`. The stored `katz_alpha` values are chosen from a safe max-degree-based bound, so the `NetworkX` runs converge on the full corpus. Katz scores are also rounded to 12 decimal places before the stored `katz_descending` order is derived, again to keep floating near-ties deterministic. The crate mirrors `NetworkX`'s `alpha=0.1` default, but also exposes conservative `safe_alpha_*` helpers for callers that want a max-degree-based choice.

The betweenness oracle stores a per-case parameter pair `(betweenness_normalized, betweenness_endpoints)`, cycling over all four exact `NetworkX` combinations for the supported unweighted undirected scorer. Betweenness scores are rounded to 12 decimal places before the stored `betweenness_descending` order is derived.

The closeness oracle stores a per-case parameter flag `closeness_wf_improved`, cycling over both exact `NetworkX` values for the supported unweighted undirected scorer. Closeness scores are rounded to 12 decimal places before the stored `closeness_descending` order is derived.

The triangle oracle stores exact integer triangle counts from `NetworkX triangles()`, and the stored `triangle_descending` order applies the usual deterministic `(triangle_count desc, node_id asc)` tie break.

The local-clustering oracle stores exact unweighted undirected scores from `NetworkX clustering()`. These scores are rounded to 12 decimal places before the stored `local_clustering_descending` order is derived.

The BFS-from-max-degree and DFS-from-max-degree fields are deterministic local contract oracles rather than direct `NetworkX` outputs, because the crate uses explicit seed/restart and neighbor-order policies for these traversal-based orderings.

This fixture is consumed by `tests/test_node_ordering.rs` and `benches/node_ordering.rs`.

# Hopcroft-Tarjan Biconnected Fixtures

The small semantic corpus is now stored directly in Rust code via `tests/support/biconnected_fixture.rs::semantic_cases()`.

Those cases pin the conventions that matter for chemistry-facing ring preprocessing:

- the `K2` dyad convention
- isolate omission from edge bicomponents
- pure tree / bridge behavior
- fused vs. spiro ring systems
- ring plus chain attachments
- bridged bicyclic cores
- disconnected mixed graphs

They are consumed directly by `tests/test_biconnected_components.rs`, `tests/test_biconnected_oracles.rs`, and `benches/biconnected_components.rs`.

`biconnected_components_order5_exhaustive.json.gz` is the exact external oracle corpus. It contains every simple undirected labeled graph on 5 vertices:

- `2^(5 choose 2) = 1024` total graphs
- deterministic case names keyed by edge-mask index
- density buckets `edge_count_0` through `edge_count_10`
- the same checked-in fields as the handcrafted suite

Unlike the handcrafted file, this corpus is generated from exact graph definitions instead of an upstream implementation:

- connected components by traversal
- articulation points by vertex deletion
- bridges by edge deletion
- biconnected blocks by maximal induced subgraphs with no articulation point,
  plus the `K2` dyad convention

Regenerate with:

```bash
python3 tests/fixtures/generate_biconnected_ground_truth.py
```

This script uses only the Python standard library.

# Maximum-Flow Ground-Truth Fixtures

`max_flow_ground_truth.json.gz` is the checked-in NetworkX oracle for s-t maximum flow, shared by every max-flow algorithm in the crate (`Dinic`, `EdmondsKarp`).

Each case stores a directed capacity graph as integer `[source, destination, capacity]` triples, the `source` and `sink` node indices, the maximum-flow `max_flow` value, and the minimum `dinitz` and `edmonds_karp` wall-clock times in nanoseconds (used as the reference timings in the benchmark).

The stored value is a consensus value. At generation time it is cross-checked across four independent NetworkX max-flow implementations (`dinitz`, `edmonds_karp`, `shortest_augmenting_path`, `preflow_push`) and, when the package is importable, against igraph `maxflow_value`. The generator aborts if any reference disagrees.

The cases span tiny hand-checkable networks (single arc, path bottleneck, diamond, antiparallel arcs, the CLRS textbook network, self-loops, a disconnected source and sink, a wider layered network), bipartite matching reductions of growing size, and several seeded random directed capacity networks.

It is consumed by `tests/test_max_flow.rs` and `benches/max_flow.rs`. It is a one-shot recording from NetworkX `3.3`, and the generator is not part of the repository.

# Minimum-Spanning-Tree Ground-Truth Fixtures

`minimum_spanning_tree_networkx.json.gz` is the shared NetworkX oracle for the `Kruskal`, `Prim`, and `Boruvka` traits. Each case stores an undirected weighted graph as `[u, v, weight]` triples plus the NetworkX summary (`mst_total_weight`, `mst_edge_count`, `number_of_components`, a `weights_distinct` flag, and `mst_edges`). Exact-edge comparison is used only when `weights_distinct` is true, since an MST is not unique under tied weights.

Cases span tiny hand-checkable graphs, disconnected forests, and seeded `G(n, p)` graphs (`n` from 10 to 1000) with both distinct and tied weights. Consumed by `tests/test_minimum_spanning_tree.rs` and `benches/minimum_spanning_tree.rs`. A one-shot recording from NetworkX `3.3`. The generator is not part of the repository.
