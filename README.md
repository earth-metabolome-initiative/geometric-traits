# Geometric traits

[![CI](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Rust%20CI/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![Security Audit](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Security%20Audit/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits)

Rust crate providing algebraic and graph algorithms, and basic structs, designed with a trait-first, correctness-focused approach. It supports `no_std` environments and offers optional `alloc` support for algorithms requiring dynamic memory allocation. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the design philosophy and how to contribute.

## Available Algorithms

The table below lists the main algorithm entrypoints currently exported from `geometric_traits::traits::algorithms`. All listed algorithms require the `alloc` feature. `RandomizedDAG` additionally requires either `std` or `hashbrown`. Full citations live in the API docs, and the fuzz harnesses are under [`fuzz/fuzz_targets`](fuzz/fuzz_targets).

| Algorithm | Trait | Complexity | Reference |
|-----------|-------|------------|-----------|
| **Bipartite Maximum Matching** | `HopcroftKarp` | O(E√V) | [Hopcroft & Karp (1973)](https://doi.org/10.1137/0202019) |
| **General Maximum Matching** | `Blossom` | O(V²E) | [Edmonds (1965)](https://doi.org/10.4153/CJM-1965-045-4) |
| **General Min-Cost Perfect Matching** | `BlossomV` | O(V²E) believed | [Kolmogorov (2009)](https://doi.org/10.1007/s12532-009-0002-8) |
| **General Maximum Matching** | `Gabow1976` | O(V³) | [Gabow (1976)](https://doi.org/10.1145/321941.321942) |
| **General Maximum Matching** | `MicaliVazirani` | O(E√V) | [Micali & Vazirani (1980)](https://doi.org/10.1109/SFCS.1980.12) |
| **Linear Assignment (Dense LAPJV)** | `LAPJV` | O(n³) | [Jonker & Volgenant (1987)](https://doi.org/10.1007/BF02278710) |
| **Linear Assignment (Sparse + Padding)** | `SparseLAPJV` | O(n³) | [Jonker & Volgenant (1987)](https://doi.org/10.1007/BF02278710) |
| **Linear Assignment (Sparse Core)** | `LAPMOD` | O(n³) | [Volgenant (1996)](https://doi.org/10.1016/0305-0548(96)00010-X) |
| **Rectangular Assignment (Diagonal Cost Extension)** | `Jaqaman` | O((L+R)³) | [Jaqaman et al. (2008)](https://doi.org/10.1038/nmeth.1237) |
| **Rectangular Assignment (Crouse LAPJV)** | `Crouse` | O(min(n,m)²·max(n,m)) | [Crouse (2016)](https://doi.org/10.1109/TAES.2016.140952) |
| **Topological Sorting** | `Kahn` | O(V+E) | [Kahn (1962)](https://doi.org/10.1145/368996.369025) |
| **Elementary Circuit Enumeration** | `Johnson` | O((V+E)(C+1)) | [Johnson (1975)](https://doi.org/10.1137/0204007) |
| **All-Pairs Shortest Paths (Weighted)** | `FloydWarshall` | O(V³) | [Floyd (1962)](https://doi.org/10.1145/367766.368168), Warshall (1962) |
| **All-Pairs Shortest Paths (Non-Negative Weighted)** | `PairwiseDijkstra` | O(V·(V+E)·log V) | [Dijkstra (1959)](https://doi.org/10.1007/BF01386390) |
| **All-Pairs Shortest Paths (Unweighted)** | `PairwiseBFS` | O(V·(V+E)) | Moore (1959) |
| **Diameter (Exact, Undirected Unweighted)** | `Diameter` | worst-case O(V·(V+E)) | Crescenzi et al. (2013) |
| **Strongly Connected Components** | `Tarjan` | O(V+E) | [Tarjan (1972)](https://doi.org/10.1137/0201010) |
| **Maximum s-t Flow / Minimum Cut** | `Dinic` | O(V²E) | Dinitz (1970) |
| **Maximum s-t Flow / Minimum Cut** | `EdmondsKarp` | O(VE²) | [Edmonds & Karp (1972)](https://doi.org/10.1145/321694.321699) |
| **Feedback Arc Set (Greedy Heuristic)** | `EadesLinSmyth` | O(V+E+cV), c = tangled-core size | [Eades, Lin & Smyth (1993)](https://doi.org/10.1016/0020-0190(93)90079-O) |
| **Biconnected Components / Articulation Points / Bridges** | `BiconnectedComponents` | O(V+E) | [Hopcroft & Tarjan (1973)](https://doi.org/10.1145/362248.362272) |
| **Cycle Detection (DFS)** | `CycleDetection` | O(V+E) | DFS back-edge detection |
| **Connected Components (Undirected)** | `ConnectedComponents` | O(V+E) | Linear-time graph traversal |
| **Bipartite Detection / 2-Coloring** | `BipartiteDetection` | O(V+E) | BFS 2-coloring |
| **Tree / Forest Detection** | `TreeDetection` | O(V+E) | Component and edge-count predicates |
| **Planarity Testing / Embedding** | `PlanarityDetection` | O(V+E) | [Boyer & Myrvold (2004)](https://doi.org/10.7155/jgaa.00091) |
| **Outerplanarity Testing** | `OuterplanarityDetection` | O(V+E) | [Boyer (2012)](https://doi.org/10.7155/jgaa.00268) |
| **K_{2,3} Homeomorph Detection** | `K23HomeomorphDetection` | O(V+E) | [Boyer (2012)](https://doi.org/10.7155/jgaa.00268) |
| **K_{3,3} Homeomorph Detection** | `K33HomeomorphDetection` | O(V+E) | [Boyer (2012)](https://doi.org/10.7155/jgaa.00268) |
| **Canonical Labeling (Labeled Simple Graphs)** | `CanonicalLabeling` | worst-case exponential backtracking | Junttila & Kaski (2007) |
| **Subgraph Isomorphism** | `Vf2` | worst-case exponential backtracking | [Cordella et al. (2004)](https://doi.org/10.1109/TPAMI.2004.75) |
| **Community Detection** | `Louvain` | O(V+E) per level | [Blondel et al. (2008)](https://doi.org/10.1088/1742-5468/2008/10/P10008) |
| **Community Detection** | `Leiden` | O(L·E), L = iterations | [Traag et al. (2019)](https://doi.org/10.1038/s41598-019-41695-z) |
| **Root Node Extraction** | `RootNodes` | O(V+E) | Graph primitive |
| **Sink Node Extraction** | `SinkNodes` | O(V+E) | Graph primitive |
| **Singleton Node Extraction** | `SingletonNodes` | O(V+E) | Graph primitive |
| **Simple Path Detection** | `SimplePath` | O(V+E) | Graph property check |
| **Information Content Propagation** | `InformationContent` | O(V+E) | [Resnik (1995)](https://arxiv.org/abs/cmp-lg/9511007) |
| **Resnik Semantic Similarity** | `Resnik` | O(R·(V+E)) per query | [Resnik (1995)](https://arxiv.org/abs/cmp-lg/9511007) |
| **Lin Semantic Similarity** | `Lin` | O(R·(V+E)) per query | Lin (1998) |
| **Wu-Palmer Semantic Similarity** | `WuPalmer` | O(R·(V+E)) per query | [Wu & Palmer (1994)](https://doi.org/10.3115/981732.981751) |
| **Line Graph** | `LineGraph` | O(∑deg²) | Whitney (1932) |
| **Labeled Line Graph** | `LabeledLineGraph` | O(∑deg²) | Labeled variant for MCES / RASCAL |
| **Modular Product** | `ModularProduct` | O(\|P\|²) | [Barrow & Burstall (1976)](https://doi.org/10.1016/0020-0190(76)90049-1) |
| **Maximum Clique Enumeration** | `MaximumClique` | O(3^(n/3)) worst case | Tomita & Seki (2003); Prosser (2012) |
| **Delta-Y Exchange Detection** | `DeltaYExchange` | O(V+E) | Delta-Y / Y-Delta detector for MCES filtering |
| **Balanced Network Flow** | `Kocay` | O(K·(V+E)) | [Kocay & Stone (1995)](https://combinatorialpress.com/jcmcc-articles/volume-019/an-algorithm-for-balanced-flows/) |
| **Minimum-Cost Maximum Balanced Flow** | `MinimumCostBalancedFlow` | hybrid exact (tree DP, bipartite min-cost flow, Blossom-V fallback) | Kocay & Stone (1995); Kolmogorov (2009) |
| **Stationary Distribution (Dense GTH)** | `Gth` | O(n³) | [Grassmann et al. (1985)](https://doi.org/10.1287/opre.33.5.1107) |
| **Eigenvalue Decomposition** | `Jacobi` | O(n³) | Jacobi (1846) |
| **Classical MDS** | `ClassicalMds` | O(n³) | [Torgerson (1952)](https://doi.org/10.1007/BF02288916) |
| **Force-Directed Layout** | `ForceAtlas2` | O(I·(V²+E)) exact, O(I·(V log V+E)) Barnes-Hut | [Jacomy et al. (2014)](https://doi.org/10.1371/journal.pone.0098679) |
| **Random DAG Generation** | `RandomizedDAG` | O(V² log V) | Utility generator (needs `std` or `hashbrown`) |

## Node Ordering Primitives

The crate also exports reusable graph-level node ordering and node scoring building blocks from `geometric_traits::traits::algorithms`.

| Primitive | Kind | Complexity | Reference |
|-----------|------|------------|-----------|
| **DegeneracySorter** | smallest-last node ordering | O(V+E) | [Matula & Beck (1983)](https://doi.org/10.1145/2402.322385) |
| **LayeredLabelPropagationSorter** | multiresolution node ordering | O(\|Γ\| · U · (V+E) + \|Γ\|² · V log V) | [Boldi et al. (2011)](https://doi.org/10.1145/1963405.1963488) |
| **CoreNumberScorer** | k-core / shell score | O(V+E) | [Batagelj & Zaveršnik (2003)](https://arxiv.org/abs/cs/0310049) |
| **DegreeScorer** | node score | O(V+E) | Graph primitive |
| **SecondOrderDegreeScorer** | node score | O(V+E) | Degree-of-neighbors score |
| **TriangleCountScorer** | node score | O(\|V̂\| · d_cover²) | Exact triangle count |
| **SquareCountScorer** | node score | O(\|V̂\| · d_cover² · d_graph) | Exact 4-cycle count |
| **LocalClusteringCoefficientScorer** | node score | O(\|V̂\| · d_cover² + V+E) | Matches `NetworkX` `clustering()` |
| **SquareClusteringCoefficientScorer** | node score | O(V · d_graph³) | Matches `NetworkX` `square_clustering()` |
| **PageRankScorer** | centrality score | O(iterations · (V+E)) | Brin & Page (1998); `NetworkX`-aligned |
| **PowerIterationEigenvectorCentralityScorer** | centrality score | O(iterations · (V+E)) | Shifted power iteration; `NetworkX`-aligned |
| **KatzCentralityScorer** | centrality score | O(iterations · (V+E)) | Katz (1953); `NetworkX`-aligned |
| **BetweennessCentralityScorer** | centrality score | O(V · (V+E)) | [Brandes (2001)](https://doi.org/10.1080/0022250X.2001.9990249) |
| **ClosenessCentralityScorer** | centrality score | O(V · (V+E)) | [Freeman (1979)](https://doi.org/10.1016/0378-8733(78)90021-7) |
| **DescendingLexicographicScoreSorter** | two-key node ordering | O(V log V) plus scorer cost | Lexicographic two-key sorter |

## Undirected Graph Generators

Standalone functions for generating undirected graphs, all returning `SymmetricCSR2D<CSR2D<usize, usize, usize>>`. All require the `alloc` feature. Random generators additionally take a `seed: u64` parameter.

### Deterministic Generators

| Generator | Function | Parameters |
|-----------|----------|------------|
| **Complete Graph** K_n | `complete_graph(n)` | n = vertices |
| **Cycle** C_n | `cycle_graph(n)` | n = vertices |
| **Path** P_n | `path_graph(n)` | n = vertices |
| **Star** S_n | `star_graph(n)` | n = leaves |
| **Grid** G_{r×c} | `grid_graph(rows, cols)` | rows, cols |
| **Hexagonal / Honeycomb Lattice** | `hexagonal_lattice_graph(rows, cols)` | rows, cols = hexagon rows, cols |
| **Triangular Lattice** | `triangular_lattice_graph(rows, cols)` | rows, cols = vertex-grid rows, cols |
| **Torus** T_{r×c} | `torus_graph(rows, cols)` | rows, cols |
| **Hypercube** Q_d | `hypercube_graph(d)` | d = dimension |
| **Barbell** B(k, p) | `barbell_graph(clique_size, path_len)` | clique_size, path_len |
| **Crown** Cr_n | `crown_graph(n)` | n = vertices per side |
| **Wheel** W_n | `wheel_graph(n)` | n = rim vertices |
| **Complete Bipartite** K_{m,n} | `complete_bipartite_graph(m, n)` | m, n = partition sizes |
| **Petersen** | `petersen_graph()` | - |
| **Turán** T(n, r) | `turan_graph(n, r)` | n = vertices, r = partitions |
| **Windmill** Wd(k, n) | `windmill_graph(num_cliques, clique_size)` | num_cliques, clique_size |
| **Friendship** F_n | `friendship_graph(n)` | n = triangles = `windmill_graph(n, 3)` |

### Random Generators

Random generators require `std` or `hashbrown` in addition to `alloc` when they use a `HashSet` internally.

| Generator | Function | Parameters | Reference |
|-----------|----------|------------|-----------|
| **Erdős-Rényi** G(n, m) | `erdos_renyi_gnm(seed, n, m)` | n = vertices, m = edges | Erdős & Rényi (1959) |
| **Erdős-Rényi** G(n, p) | `erdos_renyi_gnp(seed, n, p)` | n = vertices, p = edge probability | Gilbert (1959); geometric skip: Batagelj & Brandes (2005) |
| **Barabási-Albert** | `barabasi_albert(seed, n, m)` | n = vertices, m = edges/step; initial clique size = m + 1 | Barabási & Albert (1999) |
| **Watts-Strogatz** | `watts_strogatz(seed, n, k, beta)` | n = vertices, k = neighbours, β = rewiring prob. | Watts & Strogatz (1998) |
| **Random Regular** | `random_regular_graph(seed, n, k) -> Result<_, _>` | n = vertices, k = degree | Configuration model; Wormald (1999) |
| **Stochastic Block Model** | `stochastic_block_model(seed, sizes, p_intra, p_inter)` | community sizes, within-community p, between-community p | Holland, Laskey & Leinhardt (1983) |
| **Configuration Model** | `configuration_model(seed, degrees)` | degree sequence | Molloy & Reed (1995) |
| **Chung-Lu** | `chung_lu(seed, weights)` | weight vector | Chung & Lu (2002) |
| **Random Geometric** | `random_geometric_graph(seed, n, radius)` | n = vertices, r = connection radius | Gilbert (1961); Penrose (2003) |

## Graph & Set Similarity Metrics

Standalone free functions and a `GraphSimilarities` trait for comparing graphs or sets by their overlap. The trait provides edge-based, vertex-based, and combined similarity methods via default implementations. Any type that reports matched counts and graph sizes gets all metrics for free. These do **not** require the `alloc` feature.

| Metric | Function | Formula | Range | Reference |
|--------|----------|---------|-------|-----------|
| **Jaccard / Tanimoto** | `tanimoto_similarity` | \|A∩B\| / \|A∪B\| | [0, 1] | Jaccard (1901) |
| **Dice / Sorensen** | `dice_similarity` | 2\|A∩B\| / (\|A\|+\|B\|) | [0, 1] | Dice (1945); Sørensen (1948) |
| **Overlap / Szymkiewicz-Simpson** | `overlap_similarity` | \|A∩B\| / min(\|A\|,\|B\|) | [0, 1] | Szymkiewicz (1934); Simpson (1943) |
| **Cosine** | `cosine_similarity` | \|A∩B\| / sqrt(\|A\|\|B\|) | [0, 1] | Salton & McGill (1983) |
| **Tversky** | `tversky_similarity` | \|A∩B\| / (\|A∩B\| + α\|A\\B\| + β\|B\\A\|) | [0, 1] | Tversky (1977) |
| **Kulczynski** (2nd) | `kulczynski_similarity` | 0.5(\|A∩B\|/\|A\| + \|A∩B\|/\|B\|) | [0, 1] | Kulczyński (1927) |
| **Braun-Blanquet** | `braun_blanquet_similarity` | \|A∩B\| / max(\|A\|,\|B\|) | [0, 1] | Braun-Blanquet (1932) |
| **Sokal-Sneath** (1st) | `sokal_sneath_similarity` | \|A∩B\| / (\|A∩B\| + 2\|AΔB\|) | [0, 1] | Sokal & Sneath (1963) |
| **McConnaughey** | `mcconnaughey_similarity` | (\|A∩B\|² - \|A\\B\|\|B\\A\|) / (\|A\|\|B\|) | [-1, 1] | McConnaughey (1964) |
| **Johnson** | `johnson_similarity` | (E_c+V_c)² / ((V₁+E₁)(V₂+E₂)) | [0, 1] | Raymond et al. (2002) |
