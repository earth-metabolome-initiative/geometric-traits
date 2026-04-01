# Graph fuzzing

Honggfuzz-based fuzzing for graph data structures.

If you have not installed honggfuzz, run `cargo install honggfuzz`.

## Harnesses

### RootNodes

```bash
cargo hfuzz run root_nodes
```

and to run the crash cases:

```bash
cargo hfuzz run-debug root_nodes hfuzz_workspace/*/*.fuzz
```

### SinkNodes

```bash
cargo hfuzz run sink_nodes
```

and to run the crash cases:

```bash
cargo hfuzz run-debug sink_nodes hfuzz_workspace/*/*.fuzz
```

### SimplePath

```bash
cargo hfuzz run sink_nodes
```

and to run the crash cases:

```bash
cargo hfuzz run-debug sink_nodes hfuzz_workspace/*/*.fuzz
```

### Lin

```bash
cargo hfuzz run lin
```

and to run the crash cases:

```bash
cargo hfuzz run-debug lin hfuzz_workspace/*/*.fuzz
```

### Wu-Palmer

```bash
cargo hfuzz run wu_palmer
```

and to run the crash cases:

```bash
cargo hfuzz run-debug wu_palmer hfuzz_workspaces/*/*.fuzz
```

### CSR2D

The CSR2D struct is one of the most commonly used sparse matrix representations. As such, it is worth fuzzing extensively.

```bash
cargo hfuzz run csr2d
```

and to run the crash cases:

```bash
cargo hfuzz run-debug csr2d hfuzz_workspace/*/*.fuzz
```

### Valued CSR2D

The Valued CSR2D struct is a variant of the CSR2D struct that allows for storing values in the sparse matrix. This is useful for applications where the values of the non-zero elements are important.

```bash
cargo hfuzz run valued_csr2d
```

and to run the crash cases:

```bash
cargo hfuzz run-debug valued_csr2d hfuzz_workspace/*/*.fuzz
```

### Padded Matrix2d

The Padded Matrix2d struct is wrapper struct which fills in all missing values in the underlying matrix with a provided lambda function.

```bash
cargo hfuzz run padded_matrix2d
```

and to run the crash cases:

```bash
cargo hfuzz run-debug padded_matrix2d hfuzz_workspace/*/*.fuzz
```

### Generic Matrix2D with padded diagonal

The `GenericMatrix2DWithPaddedDiagonal` struct is a generic matrix representation that allows implicitly squaring and padding the diagonal. This is useful for applications where the underlying matrix is not squared or may not have a fully populated diagonal, but some algorithms require a square matrix with a fully populated diagonal.

```bash
cargo hfuzz run generic_matrix2d_with_padded_diagonal
```

and to run the crash cases:

```bash
cargo hfuzz run-debug generic_matrix2d_with_padded_diagonal hfuzz_workspace/*/*.fuzz
```

### Hopcroft-Karp

The Hopcroft-Karp algorithm is a combinatorial algorithm for finding maximum cardinality matchings in bipartite graphs.
It is implemented for all structs implementing `SparseMatrix2D`.

```bash
cargo hfuzz run hopcroft_karp
```

### Gabow 1976

The Gabow 1976 harness validates the new paper-structured exact matcher
against `blossom()` and checks the returned matching for structural validity.

```bash
cargo hfuzz run gabow_1976
```

and to run the crash cases:

```bash
cargo hfuzz run-debug gabow_1976 hfuzz_workspace/*/*.fuzz
```

### LAP

Unified LAP harness validating sparse wrappers and core LAPMOD invariants.

```bash
cargo hfuzz run lap
```

and to run the crash cases:

```bash
cargo hfuzz run-debug lap hfuzz_workspace/*/*.fuzz
```

### Kahn's Algorithm

The Kahn algorithm is a topological sorting algorithm for directed acyclic graphs (DAGs).

```bash
cargo hfuzz run kahn
```

and to run the crash cases:

```bash
cargo hfuzz run-debug kahn hfuzz_workspace/*/*.fuzz
```

### Tarjan's Algorithm

The Tarjan algorithm is a strongly connected components algorithm for directed graphs.

```bash
cargo hfuzz run tarjan
```

and to run the crash cases:

```bash
cargo hfuzz run-debug tarjan hfuzz_workspace/*/*.fuzz
```

### Johnson's Algorithm for simple circuits

The Johnson algorithm is an algorithm for finding all simple circuits in a directed graph.

```bash
cargo hfuzz run johnson_cycle
```

and to run the crash cases:

```bash
cargo hfuzz run-debug johnson_cycle hfuzz_workspace/*/*.fuzz
```

### Blossom V

The Blossom V harness fuzzes minimum-cost perfect matching on valid even-order
undirected weighted graphs and checks that the solver never panics and any
successful result is a valid perfect matching. For small graphs it also
cross-checks the optimum against a brute-force oracle.

There are now two Blossom V targets:
- `blossom_v`: raw edge-bag mutations, which preserves compatibility with the
  saved crash corpus
- `blossom_v_structured`: seeded structured graph families plus richer weight
  regimes, which is better for coverage growth when the raw target plateaus

```bash
cargo hfuzz run blossom_v
cargo hfuzz run blossom_v_structured
```

and to run the crash cases:

```bash
cargo hfuzz run-debug blossom_v hfuzz_workspace/*/*.fuzz
```

### Floyd-Warshall

The Floyd-Warshall algorithm computes all-pairs shortest-path distances for a
weighted adjacency matrix.

```bash
cargo hfuzz run floyd_warshall
```

and to run the crash cases:

```bash
cargo hfuzz run-debug floyd_warshall hfuzz_workspace/*/*.fuzz
```

### Pairwise BFS

The PairwiseBFS harness computes all-pairs unweighted shortest-path
distances via repeated BFS and cross-checks them against Floyd-Warshall on the
same graph with implicit unit weights.

```bash
cargo hfuzz run pairwise_bfs
```

and to run the crash cases:

```bash
cargo hfuzz run-debug pairwise_bfs hfuzz_workspace/*/*.fuzz
```

### Pairwise Dijkstra

The PairwiseDijkstra harness computes all-pairs non-negative weighted
shortest-path distances via repeated Dijkstra and cross-checks them against
Floyd-Warshall on the subset of finite square inputs where the two algorithms
must agree.

```bash
cargo hfuzz run pairwise_dijkstra
```

and to run the crash cases:

```bash
cargo hfuzz run-debug pairwise_dijkstra hfuzz_workspace/*/*.fuzz
```

### GTH

The GTH harness fuzzes the dense stationary-distribution solver. It checks
that `gth()` never panics on arbitrary dense matrices, then projects square
inputs to finite nonnegative row-stochastic matrices and validates the
resulting stationary distribution:

- entries are finite and nonnegative
- entries sum to one
- the residual `||πP - π||₁` is small
- the solver is deterministic on identical input

```bash
cargo hfuzz run gth
```

and to run the crash cases:

```bash
cargo hfuzz run-debug gth hfuzz_workspace/*/*.fuzz
```
