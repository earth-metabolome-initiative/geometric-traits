# Geometric traits

[![CI](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Rust%20CI/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![Security Audit](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Security%20Audit/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits)

Rust crate providing algebraic & graph algorithms, and basic structs.

## Main Features

This crate provides a collection of graph and algebraic algorithms designed with a trait-first approach. It supports `no_std` environments and offers optional `alloc` support for algorithms requiring dynamic memory allocation.

### Available Algorithms

The table below lists all algorithm traits currently exported from `geometric_traits::traits::algorithms`.
All listed algorithms require the `alloc` feature.
`RandomizedDAG` additionally requires either `std` or `hashbrown`.

| Algorithm | Trait | Complexity | Fuzzing Harness | Reference |
|-----------|-------|------------|-----------------|-----------|
| **Bipartite Maximum Matching** | `HopcroftKarp` | O(E√V) | [`hopcroft_karp.rs`](fuzz/fuzz_targets/hopcroft_karp.rs) | Hopcroft, J. E., & Karp, R. M. (1973). [An n^(5/2) algorithm for maximum matchings in bipartite graphs](https://doi.org/10.1137/0202019). *SIAM Journal on Computing*, 2(4), 225-231. |
| **General Maximum Matching** | `Blossom` | O(V²E) | [`blossom.rs`](fuzz/fuzz_targets/blossom.rs) | Edmonds, J. (1965). [Paths, trees, and flowers](https://doi.org/10.4153/CJM-1965-045-4). *Canadian Journal of Mathematics*, 17, 449-467. |
| **General Min-Cost Perfect Matching** | `BlossomV` | O(V²E) believed | [`blossom_v.rs`](fuzz/fuzz_targets/blossom_v.rs), [`blossom_v_structured.rs`](fuzz/fuzz_targets/blossom_v_structured.rs) | Kolmogorov, V. (2009). [Blossom V: A new implementation of a minimum cost perfect matching algorithm](https://doi.org/10.1007/s12532-009-0002-8). *Mathematical Programming Computation*, 1(1), 43-67. |
| **General Maximum Matching** | `Gabow1976` | O(V³) | [`gabow_1976.rs`](fuzz/fuzz_targets/gabow_1976.rs) | Gabow, H. N. (1976). [An efficient implementation of Edmonds' algorithm for maximum matching on graphs](https://doi.org/10.1145/321941.321942). *Journal of the ACM*, 23(2), 221-234. |
| **General Maximum Matching** | `MicaliVazirani` | O(E√V) | [`micali_vazirani.rs`](fuzz/fuzz_targets/micali_vazirani.rs) | Micali, S., & Vazirani, V. V. (1980). [An O(√\|V\| · \|E\|) algorithm for finding maximum matching in general graphs](https://doi.org/10.1109/SFCS.1980.12). *FOCS*, 17-27. Peterson, P. A., & Loui, M. C. (1988). [The general maximum matching algorithm of Micali and Vazirani](https://doi.org/10.1007/BF01762129). *Algorithmica*, 3, 511-533. |
| **General Maximum Matching** | `Blum` | O(V·(V+E)) worst case | [`blum.rs`](fuzz/fuzz_targets/blum.rs) | Blum, N. (1990). [A new approach to maximum matching in general graphs](https://doi.org/10.1007/BFb0032060). *ICALP*, LNCS 443, 586-597. Revised 2015 ([arXiv:1509.04927](https://arxiv.org/abs/1509.04927)). Corrections: Dandeh, A., & Lukovszki, T. (2025). [Experimental evaluation of Blum's maximum matching algorithm](https://ceur-ws.org/Vol-4039/paper23.pdf). *ICTCS*. |
| **Exact Karp-Sipser Preprocessing** | `KarpSipser` | O(T_preprocess + T_exact(kernel)) | [`karp_sipser.rs`](fuzz/fuzz_targets/karp_sipser.rs) | Karp, R. M., & Sipser, M. (1981). [Maximum matchings in sparse random graphs](https://doi.org/10.1109/SFCS.1981.21). *FOCS*, 364-375. For the exact degree-1 / degree-2 reduction rules on general graphs used here, see Mertzios, G. B., Nichterlein, A., & Niedermeier, R. (2020). [The power of linear-time data reduction for maximum matching](https://doi.org/10.1007/s00453-020-00736-0). *Algorithmica*, 82, 3521-3565. Related bipartite kernelization work: Kaya, K., Langguth, J., Panagiotas, I., & Uçar, B. (2020). [Karp-Sipser based kernels for bipartite graph matching](https://doi.org/10.1137/1.9781611976007.11). *ALENEX*, 134-145. |
| **Linear Assignment (Dense LAPJV)** | `LAPJV` | O(n³) | [`lap.rs`](fuzz/fuzz_targets/lap.rs) | Jonker, R., & Volgenant, A. (1987). [A shortest augmenting path algorithm for dense and sparse linear assignment problems](https://doi.org/10.1007/BF02278710). *Computing*, 38(4), 325-340. |
| **Linear Assignment (Sparse + Padding)** | `SparseLAPJV` | O(n³) | [`lap.rs`](fuzz/fuzz_targets/lap.rs) | Jonker, R., & Volgenant, A. (1987). [A shortest augmenting path algorithm for dense and sparse linear assignment problems](https://doi.org/10.1007/BF02278710). *Computing*, 38(4), 325-340. |
| **Linear Assignment (Sparse Core)** | `LAPMOD` | O(n³) | [`lap.rs`](fuzz/fuzz_targets/lap.rs) | Volgenant, A. (1996). [Linear and semi-assignment problems: A core oriented approach](https://doi.org/10.1016/0305-0548(96)00010-X). *Computers & Operations Research*, 23(10), 917-932. |
| **Rectangular Assignment (Diagonal Cost Extension)** | `Jaqaman` | O((L+R)³) | [`lap.rs`](fuzz/fuzz_targets/lap.rs) | Jaqaman, K., et al. (2008). [Robust single-particle tracking in live-cell time-lapse sequences](https://doi.org/10.1038/nmeth.1237). *Nature Methods*, 5(8), 695-702. See also Ramshaw, L., & Tarjan, R. E. (2012). *On minimum-cost assignments in unbalanced bipartite graphs* (Tech. Rep. HPL-2012-40). Related conference paper: [A weight-scaling algorithm for min-cost imperfect matchings in bipartite graphs](https://doi.org/10.1109/FOCS.2012.9). |
| **Rectangular Assignment (Crouse LAPJV)** | `Crouse` | O(min(n,m)²·max(n,m)) | - | Crouse, D. F. (2016). *On implementing 2D rectangular assignment algorithms*. *IEEE Transactions on Aerospace and Electronic Systems*, 52(4), 1679-1696. DOI: `10.1109/TAES.2016.140952`. |
| **Topological Sorting** | `Kahn` | O(V+E) | [`kahn.rs`](fuzz/fuzz_targets/kahn.rs) | Kahn, A. B. (1962). [Topological sorting of large networks](https://doi.org/10.1145/368996.369025). *Communications of the ACM*, 5(11), 558-562. |
| **Elementary Circuit Enumeration** | `Johnson` | O((V+E)(C+1)) | [`johnson_cycle.rs`](fuzz/fuzz_targets/johnson_cycle.rs) | Johnson, D. B. (1975). [Finding all the elementary circuits of a directed graph](https://doi.org/10.1137/0204007). *SIAM Journal on Computing*, 4(1), 77-84. |
| **All-Pairs Shortest Paths (Weighted)** | `FloydWarshall` | O(V³) | [`floyd_warshall.rs`](fuzz/fuzz_targets/floyd_warshall.rs) | Floyd, R. W. (1962). [Algorithm 97: Shortest path](https://doi.org/10.1145/367766.368168). *Communications of the ACM*, 5(6), 345. Warshall, S. (1962). [A theorem on Boolean matrices](https://doi.org/10.1145/321105.321107). *Journal of the ACM*, 9(1), 11-12. |
| **All-Pairs Shortest Paths (Non-Negative Weighted)** | `PairwiseDijkstra` | O(V·(V+E)·log V) | [`pairwise_dijkstra.rs`](fuzz/fuzz_targets/pairwise_dijkstra.rs) | Dijkstra, E. W. (1959). [A note on two problems in connexion with graphs](https://doi.org/10.1007/BF01386390). *Numerische Mathematik*, 1, 269-271. |
| **All-Pairs Shortest Paths (Unweighted)** | `PairwiseBFS` | O(V·(V+E)) | [`pairwise_bfs.rs`](fuzz/fuzz_targets/pairwise_bfs.rs) | Repeated breadth-first search for unweighted APSP; see Moore, E. F. (1959). *The shortest path through a maze*. In *Proceedings of the International Symposium on the Theory of Switching*, 285-292. |
| **Strongly Connected Components** | `Tarjan` | O(V+E) | [`tarjan.rs`](fuzz/fuzz_targets/tarjan.rs) | Tarjan, R. E. (1972). [Depth-first search and linear graph algorithms](https://doi.org/10.1137/0201010). *SIAM Journal on Computing*, 1(2), 146-160. |
| **Cycle Detection (DFS)** | `CycleDetection` | O(V+E) | - | Standard depth-first back-edge detection (no single canonical paper citation). |
| **Connected Components (Undirected)** | `ConnectedComponents` | O(V+E) | - | Standard linear-time graph traversal (no single canonical paper citation). |
| **Community Detection** | `Louvain` | O(V+E) per level | [`louvain.rs`](fuzz/fuzz_targets/louvain.rs) | Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). [Fast unfolding of communities in large networks](https://doi.org/10.1088/1742-5468/2008/10/P10008). *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. |
| **Community Detection** | `Leiden` | O(L·E), L = iterations | [`leiden.rs`](fuzz/fuzz_targets/leiden.rs) | Traag, V. A., Waltman, L., & van Eck, N. J. (2019). [From Louvain to Leiden: guaranteeing well-connected communities](https://doi.org/10.1038/s41598-019-41695-z). *Scientific Reports*, 9, 5233. |
| **Root Node Extraction** | `RootNodes` | O(V+E) | [`root_nodes.rs`](fuzz/fuzz_targets/root_nodes.rs) | Graph primitive (no specific paper citation). |
| **Sink Node Extraction** | `SinkNodes` | O(V+E) | [`sink_nodes.rs`](fuzz/fuzz_targets/sink_nodes.rs) | Graph primitive (no specific paper citation). |
| **Singleton Node Extraction** | `SingletonNodes` | O(V+E) | - | Graph primitive (no specific paper citation). |
| **Simple Path Detection** | `SimplePath` | O(V+E) | [`simple_path.rs`](fuzz/fuzz_targets/simple_path.rs) | Graph property check (no specific paper citation). |
| **Information Content Propagation** | `InformationContent` | O(V+E) | - | Resnik, P. (1995). [Using information content to evaluate semantic similarity in a taxonomy](https://arxiv.org/abs/cmp-lg/9511007). In *Proceedings of IJCAI-95*, 448-453. |
| **Resnik Semantic Similarity** | `Resnik` | O(R·(V+E)) per query | - | Resnik, P. (1995). [Using information content to evaluate semantic similarity in a taxonomy](https://arxiv.org/abs/cmp-lg/9511007). In *Proceedings of IJCAI-95*, 448-453. |
| **Lin Semantic Similarity** | `Lin` | O(R·(V+E)) per query | [`lin.rs`](fuzz/fuzz_targets/lin.rs) | Lin, D. (1998). *An Information-Theoretic Definition of Similarity*. In *Proceedings of ICML 1998*, 296-304. |
| **Wu-Palmer Semantic Similarity** | `WuPalmer` | O(R·(V+E)) per query | [`wu_palmer.rs`](fuzz/fuzz_targets/wu_palmer.rs) | Wu, Z., & Palmer, M. (1994). [Verb Semantics and Lexical Selection](https://aclanthology.org/P94-1019/). In *Proceedings of ACL 1994*, 133-138. DOI: `10.3115/981732.981751`. |
| **Line Graph** | `LineGraph` | O(∑deg²) | [`line_graph.rs`](fuzz/fuzz_targets/line_graph.rs) | Whitney, H. (1932). Congruent graphs and the connectivity of graphs. *American Journal of Mathematics*, 54(1), 150–168. |
| **Labeled Line Graph** | `LabeledLineGraph` | O(∑deg²) | - | Labeled variant: edges carry the node type of the shared endpoint. Building block for labeled MCES / RASCAL. |
| **Modular Product** | `ModularProduct` | O(\|P\|²) | [`modular_product.rs`](fuzz/fuzz_targets/modular_product.rs), [`labeled_modular_product.rs`](fuzz/fuzz_targets/labeled_modular_product.rs) | Barrow, H. G., & Burstall, R. M. (1976). [Subgraph isomorphism, matching relational structures and maximal cliques](https://doi.org/10.1016/0020-0190(76)90049-1). *Information Processing Letters*, 4(4), 83–84. Unlabeled and labeled variants with custom edge comparator. |
| **Maximum Clique Enumeration** | `MaximumClique` | O(3^(n/3)) worst case | [`maximum_clique.rs`](fuzz/fuzz_targets/maximum_clique.rs) | Tomita, E., & Seki, T. (2003). An efficient branch-and-bound algorithm for finding a maximum clique. *LNCS* 2731:278-289. San Segundo, P., et al. (2011). An exact bit-parallel algorithm for the maximum clique problem. *Computers & OR* 38(2). Prosser, P. (2012). Exact Algorithms for Maximum Clique. *Algorithms* 5(4):545-587. |
| **Delta-Y Exchange Detection** | `DeltaYExchange` | O(V+E) | [`delta_y_exchange.rs`](fuzz/fuzz_targets/delta_y_exchange.rs) | Detects whether two graphs are related by a Delta-Y or Y-Delta exchange. Used for MCES filtering. |
| **Balanced Network Flow** | `Kocay` | O(K·(V+E)) | [`kocay.rs`](fuzz/fuzz_targets/kocay.rs) | Kocay, W., & Stone, D. (1995). [An Algorithm for Balanced Flows](https://combinatorialpress.com/jcmcc-articles/volume-019/an-algorithm-for-balanced-flows/). *Journal of Combinatorial Mathematics and Combinatorial Computing*, 19, 3–31. Earlier exposition: Kocay, W., & Stone, D. (1993). *Balanced network flows*. *Bulletin of the Institute of Combinatorics and its Applications*, 7, 17–32. |
| **Stationary Distribution (Dense GTH)** | `Gth` | O(n³) | - | Grassmann, W. K., Taksar, M. I., & Heyman, D. P. (1985). [Regenerative Analysis and Steady State Distributions for Markov Chains](https://doi.org/10.1287/opre.33.5.1107). *Operations Research*, 33(5), 1107-1116. |
| **Eigenvalue Decomposition** | `Jacobi` | O(n³) | [`jacobi.rs`](fuzz/fuzz_targets/jacobi.rs) | Jacobi, C. G. J. (1846). Über ein leichtes Verfahren die in der Theorie der Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen. *Journal für die reine und angewandte Mathematik*, 30, 51–94. See Golub & Van Loan (2013), §8.5. |
| **Classical MDS** | `ClassicalMds` | O(n³) | [`mds.rs`](fuzz/fuzz_targets/mds.rs) | Torgerson, W. S. (1952). [Multidimensional scaling: I. Theory and method](https://doi.org/10.1007/BF02288916). *Psychometrika*, 17(4), 401–419. |
| **Random DAG Generation** | `RandomizedDAG` | O(V² log V) | - | Utility generator (requires `std` or `hashbrown` in addition to `alloc`). |

`KarpSipser` in this crate is an exact preprocessing layer for maximum matching on general graphs. The ALENEX 2020 paper above is relevant related work, but it is specifically about bipartite graph matching rather than the general-graph wrapper implemented here.

**Blum note:** the papers claim a phased O(√V·(V+E)) bound. This implementation keeps that fast path, but adds correctness fallbacks for three bugs we found in the published algorithm. The fallbacks make the worst case O(V·(V+E)). Dandeh & Lukovszki (ICTCS 2025) independently found two additional single-path MDFS bugs; their counterexample graphs are in our test suite but do not trigger in our phased architecture.

### Undirected Graph Generators

Standalone functions for generating undirected graphs, all returning `SymmetricCSR2D<CSR2D<usize, usize, usize>>`. All require the `alloc` feature. Random generators additionally take a `seed: u64` parameter.

#### Deterministic Generators

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
| **Petersen** | `petersen_graph()` | — |
| **Turán** T(n, r) | `turan_graph(n, r)` | n = vertices, r = partitions |
| **Windmill** Wd(k, n) | `windmill_graph(num_cliques, clique_size)` | num_cliques, clique_size |
| **Friendship** F_n | `friendship_graph(n)` | n = triangles = `windmill_graph(n, 3)` |

#### Random Generators

Random generators require `std` or `hashbrown` in addition to `alloc` when they use a `HashSet` internally.

| Generator | Function | Parameters | Reference |
|-----------|----------|------------|-----------|
| **Erdős–Rényi** G(n, m) | `erdos_renyi_gnm(seed, n, m)` | n = vertices, m = edges | Erdős & Rényi (1959) |
| **Erdős–Rényi** G(n, p) | `erdos_renyi_gnp(seed, n, p)` | n = vertices, p = edge probability | Gilbert (1959); geometric skip: Batagelj & Brandes (2005) |
| **Barabási–Albert** | `barabasi_albert(seed, n, m)` | n = vertices, m = edges/step; initial clique size = m + 1 | Barabási & Albert (1999) |
| **Watts–Strogatz** | `watts_strogatz(seed, n, k, beta)` | n = vertices, k = neighbours, β = rewiring prob. | Watts & Strogatz (1998) |
| **Random Regular** | `random_regular_graph(seed, n, k) -> Result<_, _>` | n = vertices, k = degree | Configuration model; Wormald (1999) |
| **Stochastic Block Model** | `stochastic_block_model(seed, sizes, p_intra, p_inter)` | community sizes, within-community p, between-community p | Holland, Laskey & Leinhardt (1983) |
| **Configuration Model** | `configuration_model(seed, degrees)` | degree sequence | Molloy & Reed (1995) |
| **Chung–Lu** | `chung_lu(seed, weights)` | weight vector | Chung & Lu (2002) |
| **Random Geometric** | `random_geometric_graph(seed, n, radius)` | n = vertices, r = connection radius | Gilbert (1961); Penrose (2003) |

### Graph & Set Similarity Metrics

Standalone free functions and a `GraphSimilarities` trait for comparing graphs or sets by their overlap. The trait provides edge-based, vertex-based, and combined similarity methods via default implementations — any type that reports matched counts and graph sizes gets all metrics for free. These do **not** require the `alloc` feature.

| Metric | Function | Formula | Range | Common Uses | Reference |
|--------|----------|---------|-------|-------------|-----------|
| **Jaccard / Tanimoto** | `tanimoto_similarity` | \|A∩B\| / \|A∪B\| | [0, 1] | Molecular fingerprint comparison, ecology, information retrieval | Jaccard, P. (1901). Distribution de la flore alpine dans le bassin des Dranses et dans quelques regions voisines. *Bulletin de la Societe Vaudoise des Sciences Naturelles*, 37, 241-272. |
| **Dice / Sorensen** | `dice_similarity` | 2\|A∩B\| / (\|A\|+\|B\|) | [0, 1] | Medical image segmentation, ecological vegetation surveys, NLP token overlap; equivalent to F1 score | Dice, L. R. (1945). Measures of the amount of ecologic association between species. *Ecology*, 26(3), 297-302. Sorensen, T. (1948). A method of establishing groups of equal amplitude in plant sociology. *Kongelige Danske Videnskabernes Selskab*, 5(4), 1-34. |
| **Overlap / Szymkiewicz-Simpson** | `overlap_similarity` | \|A∩B\| / min(\|A\|,\|B\|) | [0, 1] | Genomic region overlap, graph link prediction (RAPIDS cuGraph), containment queries | Szymkiewicz, D. (1934). Une contribution statistique a la geographie floristique. *Acta Societatis Botanicorum Poloniae*, 11(3), 249-265. Simpson, G. G. (1943). Mammals and the nature of continents. *American Journal of Science*, 241(1), 1-31. |
| **Cosine** | `cosine_similarity` | \|A∩B\| / sqrt(\|A\|\|B\|) | [0, 1] | Document/text similarity (NLP), sparse high-dimensional feature comparison, recommendation systems | Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill. |
| **Tversky** | `tversky_similarity` | \|A∩B\| / (\|A∩B\| + α\|A\\B\| + β\|B\\A\|) | [0, 1] | Asymmetric substructure search, scaffold hopping in drug discovery; generalizes Jaccard (α=β=1) and Dice (α=β=0.5) | Tversky, A. (1977). Features of similarity. *Psychological Review*, 84(4), 327-352. |
| **Kulczynski** (2nd) | `kulczynski_similarity` | 0.5(\|A∩B\|/\|A\| + \|A∩B\|/\|B\|) | [0, 1] | Botanical community comparison, balanced view when sets differ in size | Kulczynski, S. (1927). Die Pflanzenassoziationen der Pieninen. *Bulletin International de l'Academie Polonaise des Sciences et des Lettres*, Classe des Sciences Mathematiques et Naturelles, Serie B, 57-203. |
| **Braun-Blanquet** | `braun_blanquet_similarity` | \|A∩B\| / max(\|A\|,\|B\|) | [0, 1] | Phytosociology, conservative containment assessment; counterpart of overlap (uses max instead of min) | Braun-Blanquet, J. (1932). *Plant Sociology: The Study of Plant Communities*. McGraw-Hill. |
| **Sokal-Sneath** (1st) | `sokal_sneath_similarity` | \|A∩B\| / (\|A∩B\| + 2\|AΔB\|) | [0, 1] | Numerical taxonomy, strict classification with double penalty for mismatches | Sokal, R. R., & Sneath, P. H. A. (1963). *Principles of Numerical Taxonomy*. W. H. Freeman. |
| **McConnaughey** | `mcconnaughey_similarity` | (\|A∩B\|² - \|A\\B\|\|B\\A\|) / (\|A\|\|B\|) | [-1, 1] | Spectral matching in metabolomics, compound identification in mass spectrometry; correlation-like (can detect anti-correlation) | McConnaughey, B. H. (1964). The determination and analysis of plankton communities. *Marine Research in Indonesia*, 1-40. |
| **Johnson** | `johnson_similarity` | (E_c+V_c)² / ((V₁+E₁)(V₂+E₂)) | [0, 1] | MCES result scoring (RASCAL algorithm); combines matched edge and vertex counts | Raymond, J. W., Gardiner, E. J., & Willett, P. (2002). RASCAL: Calculation of graph similarity using maximum common edge subgraphs. *The Computer Journal*, 45(6), 631-644. |

### Design Philosophy

* **Trait-Based**: Algorithms are implemented generic over traits such as `BipartiteGraph` and `MonopartiteGraph`, allowing them to be used with any backing data structure that implements the required interface (e.g., Matrices, CSR, Adjacency Lists).
* **Fuzzing & Correctness**: A significant focus is placed on correctness. Key algorithms are continuously fuzzed using `honggfuzz` to ensure robustness against edge cases and to verify invariants.
* **`no_std` Compatible**: The core traits and several implementations are designed to work in `no_std` environments. Feature flags allow enabling `std` or `alloc` only when necessary.
