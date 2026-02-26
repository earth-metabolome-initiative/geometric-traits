# Geometric traits

[![CI](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Rust%20CI/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![Security Audit](https://github.com/earth-metabolome-initiative/geometric-traits/workflows/Security%20Audit/badge.svg)](https://github.com/earth-metabolome-initiative/geometric-traits/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/geometric-traits)

Rust crate providing algebraic & graph algorithms, and basic structs.

## Main Features

This crate provides a collection of graph and algebraic algorithms designed with a trait-first approach. It supports `no_std` environments and offers optional `alloc` support for algorithms requiring dynamic memory allocation.

### Key Algorithms

The library implements several classical algorithms, verified against standard implementations or known properties.

| Algorithm | Trait | Fuzzing Harness | Reference |
|-----------|-------|-----------------|-----------|
| **Linear Assignment (LAPJV)** | `LAPJV`, `SparseLAPJV` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Jonker, R., & Volgenant, A. (1987). [A shortest augmenting path algorithm for dense and sparse linear assignment problems](https://doi.org/10.1007/BF02278710). Computing, 38, 325-340. |
| **Linear Assignment (LAPMOD)** | `LAPMOD`, `SparseLAPMOD` | [`lap.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lap.rs) | Volgenant, A. (1996). [Linear and semi-assignment problems: A core oriented approach](https://doi.org/10.1016/0305-0548(95)00099-2). Computers & Operations Research, 23(10), 917-932. |
| **Bipartite Matching** | `HopcroftKarp` | [`hopcroft_karp.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/hopcroft_karp.rs) | Hopcroft, J. E., & Karp, R. M. (1973). [An n^5/2 algorithm for maximum matchings in bipartite graphs](https://doi.org/10.1137/0202019). SIAM Journal on Computing, 2(4), 225-231. |
| **Strongly Connected Components** | `Tarjan` | [`tarjan.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/tarjan.rs) | Tarjan, R. (1972). [Depth-first search and linear graph algorithms](https://doi.org/10.1137/0201010). SIAM journal on computing, 1(2), 146-160. |
| **Topological Sorting** | `Kahn` | [`kahn.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/kahn.rs) | Kahn, A. B. (1962). [Topological sorting of large networks](https://doi.org/10.1145/368996.369025). Communications of the ACM, 5(11), 558-562. |
| **Cycle Detection** | `Johnson` | [`johnson_cycle.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/johnson_cycle.rs) | Johnson, D. B. (1975). [Finding all the elementary circuits of a directed graph](https://doi.org/10.1137/0204007). SIAM Journal on Computing, 4(1), 77-84. |
| **Lin Similarity** | `Lin` | [`lin.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/lin.rs) | Lin, D. (1998). [An information-theoretic definition of similarity](https://dl.acm.org/doi/10.5555/645527.657297). In Icml (Vol. 98, No. 1998, pp. 296-304). |
| **Wu-Palmer Similarity** | `WuPalmer` | [`wu_palmer.rs`](https://github.com/earth-metabolome-initiative/geometric-traits/blob/main/fuzz/fuzz_targets/wu_palmer.rs) | Wu, Z., & Palmer, M. (1994). [Verbs semantics and lexical selection](https://dl.acm.org/doi/10.3115/981732.981751). In Proceedings of the 32nd annual meeting on Association for Computational Linguistics (pp. 133-138). |
| **Resnik Similarity** | `Resnik` | - | Resnik, P. (1995). [Using information content to evaluate semantic similarity in a taxonomy](https://arxiv.org/pdf/cmp-lg/9511007). In Proceedings of the 14th international joint conference on artificial intelligence (IJCAI) (Vol. 1, pp. 448-453). |

### Design Philosophy

* **Trait-Based**: Algorithms are implemented generic over traits such as `BipartiteGraph` and `MonopartiteGraph`, allowing them to be used with any backing data structure that implements the required interface (e.g., Matrices, CSR, Adjacency Lists).
* **Fuzzing & Correctness**: A significant focus is placed on correctness. Key algorithms are continuously fuzzed using `honggfuzz` to ensure robustness against edge cases and to verify invariants.
* **`no_std` Compatible**: The core traits and several implementations are designed to work in `no_std` environments. Feature flags allow enabling `std` or `alloc` only when necessary.
