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

| Algorithm | Trait | Fuzzing Harness | Reference |
|-----------|-------|-----------------|-----------|
| **Bipartite Maximum Matching** | `HopcroftKarp` | [`hopcroft_karp.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/hopcroft_karp.rs) | Hopcroft, J. E., & Karp, R. M. (1973). [An n^(5/2) algorithm for maximum matchings in bipartite graphs](https://doi.org/10.1137/0202019). *SIAM Journal on Computing*, 2(4), 225-231. |
| **Linear Assignment (Dense LAPJV)** | `LAPJV` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Jonker, R., & Volgenant, A. (1987). [A shortest augmenting path algorithm for dense and sparse linear assignment problems](https://doi.org/10.1007/BF02278710). *Computing*, 38(4), 325-340. |
| **Linear Assignment (Sparse + Padding)** | `SparseLAPJV` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Jonker, R., & Volgenant, A. (1987). [A shortest augmenting path algorithm for dense and sparse linear assignment problems](https://doi.org/10.1007/BF02278710). *Computing*, 38(4), 325-340. |
| **Linear Assignment (Sparse Core)** | `LAPMOD` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Volgenant, A. (1996). [Linear and semi-assignment problems: A core oriented approach](https://doi.org/10.1016/0305-0548(96)00010-X). *Computers & Operations Research*, 23(10), 917-932. |
| **Rectangular Assignment (Diagonal Cost Extension)** | `Jaqaman` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Jaqaman, K., et al. (2008). [Robust single-particle tracking in live-cell time-lapse sequences](https://doi.org/10.1038/nmeth.1237). *Nature Methods*, 5(8), 695-702. See also Ramshaw, L., & Tarjan, R. E. (2012). [On minimum-cost assignments in unbalanced bipartite graphs](https://web.archive.org/web/20230130060925/http://www.hpl.hp.com/techreports/2012/HPL-2012-40.pdf). |
| **Rectangular Assignment (Crouse LAPJV)** | `Crouse` | - | Crouse, D. F. (2016). [On implementing 2D rectangular assignment algorithms](https://doi.org/10.1109/TAES.2016.140952). *IEEE Transactions on Aerospace and Electronic Systems*, 52(4), 1679-1696. |
| **Topological Sorting** | `Kahn` | [`kahn.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/kahn.rs) | Kahn, A. B. (1962). [Topological sorting of large networks](https://doi.org/10.1145/368996.369025). *Communications of the ACM*, 5(11), 558-562. |
| **Elementary Circuit Enumeration** | `Johnson` | [`johnson_cycle.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/johnson_cycle.rs) | Johnson, D. B. (1975). [Finding all the elementary circuits of a directed graph](https://doi.org/10.1137/0204007). *SIAM Journal on Computing*, 4(1), 77-84. |
| **Strongly Connected Components** | `Tarjan` | [`tarjan.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/tarjan.rs) | Tarjan, R. E. (1972). [Depth-first search and linear graph algorithms](https://doi.org/10.1137/0201010). *SIAM Journal on Computing*, 1(2), 146-160. |
| **Cycle Detection (DFS)** | `CycleDetection` | - | Standard depth-first back-edge detection (no single canonical paper citation). |
| **Connected Components (Undirected)** | `ConnectedComponents` | - | Standard linear-time graph traversal (no single canonical paper citation). |
| **Community Detection** | `Louvain` | [`louvain.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/louvain.rs) | Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). [Fast unfolding of communities in large networks](https://doi.org/10.1088/1742-5468/2008/10/P10008). *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. |
| **Root Node Extraction** | `RootNodes` | [`root_nodes.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/root_nodes.rs) | Graph primitive (no specific paper citation). |
| **Sink Node Extraction** | `SinkNodes` | [`sink_nodes.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/sink_nodes.rs) | Graph primitive (no specific paper citation). |
| **Singleton Node Extraction** | `SingletonNodes` | - | Graph primitive (no specific paper citation). |
| **Simple Path Detection** | `SimplePath` | [`simple_path.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/simple_path.rs) | Graph property check (no specific paper citation). |
| **Information Content Propagation** | `InformationContent` | - | Resnik, P. (1995). [Using information content to evaluate semantic similarity in a taxonomy](https://arxiv.org/abs/cmp-lg/9511007). In *Proceedings of IJCAI-95*, 448-453. |
| **Resnik Semantic Similarity** | `Resnik` | - | Resnik, P. (1995). [Using information content to evaluate semantic similarity in a taxonomy](https://arxiv.org/abs/cmp-lg/9511007). In *Proceedings of IJCAI-95*, 448-453. |
| **Lin Semantic Similarity** | `Lin` | [`lin.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lin.rs) | Lin, D. (1998). [An Information-Theoretic Definition of Similarity](https://icml.cc/Conferences/1998/papers/paper37.html). In *Proceedings of ICML 1998*, 296-304. |
| **Wu-Palmer Semantic Similarity** | `WuPalmer` | [`wu_palmer.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/wu_palmer.rs) | Wu, Z., & Palmer, M. (1994). [Verb Semantics and Lexical Selection](https://doi.org/10.3115/981732.981751). In *Proceedings of ACL 1994*, 133-138. |
| **Random DAG Generation** | `RandomizedDAG` | - | Utility generator (requires `std` or `hashbrown` in addition to `alloc`). |

### Design Philosophy

* **Trait-Based**: Algorithms are implemented generic over traits such as `BipartiteGraph` and `MonopartiteGraph`, allowing them to be used with any backing data structure that implements the required interface (e.g., Matrices, CSR, Adjacency Lists).
* **Fuzzing & Correctness**: A significant focus is placed on correctness. Key algorithms are continuously fuzzed using `honggfuzz` to ensure robustness against edge cases and to verify invariants.
* **`no_std` Compatible**: The core traits and several implementations are designed to work in `no_std` environments. Feature flags allow enabling `std` or `alloc` only when necessary.
