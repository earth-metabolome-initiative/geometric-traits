//! Tests for TransposedMonoplexMonopartiteGraph::is_singleton and
//! SparseBiMatrix2D/SizedSparseBiMatrix2D column operations on
//! GenericBiMatrix2D.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SortedVec, SquareCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        EdgesBuilder, SizedSparseBiMatrix2D, SparseBiMatrix2D, TransposedMonoplexMonopartiteGraph,
        VocabularyBuilder,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestSquareBiMatrix = GenericBiMatrix2D<TestSquareCSR, TestSquareCSR>;
type TestBiMatrix = GenericBiMatrix2D<TestCSR, TestCSR>;
type TestGraph = GenericGraph<SortedVec<usize>, TestSquareBiMatrix>;

/// Helper to build a graph that supports transposed operations and
/// is_singleton.
fn build_graph(node_count: usize, edges: Vec<(usize, usize)>) -> TestGraph {
    let nodes: Vec<usize> = (0..node_count).collect();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_count = edges.len();
    let edges: TestSquareCSR = DiEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    let edges: TestSquareBiMatrix = GenericBiMatrix2D::new(edges);
    GenericGraph::from((nodes, edges))
}

// ============================================================================
// is_singleton tests
// ============================================================================

#[test]
fn test_is_singleton_isolated_node() {
    // Node 2 has no incoming or outgoing edges
    let graph = build_graph(3, vec![(0, 1)]);

    assert!(graph.is_singleton(2));
}

#[test]
fn test_is_singleton_source_node() {
    // Node 0 has outgoing but no incoming edges
    let graph = build_graph(3, vec![(0, 1), (0, 2)]);

    assert!(!graph.is_singleton(0));
}

#[test]
fn test_is_singleton_sink_node() {
    // Node 2 has incoming but no outgoing edges
    let graph = build_graph(3, vec![(0, 2), (1, 2)]);

    assert!(!graph.is_singleton(2));
}

#[test]
fn test_is_singleton_connected_node() {
    // Node 1 has both incoming and outgoing edges
    let graph = build_graph(3, vec![(0, 1), (1, 2)]);

    assert!(!graph.is_singleton(1));
}

#[test]
fn test_is_singleton_all_isolated() {
    let graph = build_graph(3, vec![]);

    assert!(graph.is_singleton(0));
    assert!(graph.is_singleton(1));
    assert!(graph.is_singleton(2));
}

#[test]
fn test_is_singleton_self_loop() {
    // Node 0 has a self-loop: it has both incoming and outgoing edges
    let graph = build_graph(2, vec![(0, 0)]);

    assert!(!graph.is_singleton(0));
    assert!(graph.is_singleton(1));
}

// ============================================================================
// SparseBiMatrix2D::sparse_column tests
// ============================================================================

fn build_bimatrix(node_count: usize, edges: Vec<(usize, usize)>) -> TestBiMatrix {
    let edge_count = edges.len();
    let inner: TestCSR = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edge_count)
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericBiMatrix2D::new(inner)
}

#[test]
fn test_sparse_column_basic() {
    // 0 -> 1, 0 -> 2, 1 -> 2
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);

    // Column 2: rows 0 and 1 have entries in column 2
    let col2: Vec<usize> = bm.sparse_column(2).collect();
    assert_eq!(col2, vec![0, 1]);

    // Column 1: only row 0 has an entry in column 1
    let col1: Vec<usize> = bm.sparse_column(1).collect();
    assert_eq!(col1, vec![0]);

    // Column 0: no entries in column 0
    let col0: Vec<usize> = bm.sparse_column(0).collect();
    assert!(col0.is_empty());
}

#[test]
fn test_sparse_column_empty_matrix() {
    let bm = build_bimatrix(3, vec![]);

    let col0: Vec<usize> = bm.sparse_column(0).collect();
    assert!(col0.is_empty());
}

#[test]
fn test_sparse_column_convergent() {
    // All nodes point to node 2
    let bm = build_bimatrix(4, vec![(0, 2), (1, 2), (3, 2)]);

    let col2: Vec<usize> = bm.sparse_column(2).collect();
    assert_eq!(col2, vec![0, 1, 3]);
}

// ============================================================================
// SizedSparseBiMatrix2D::number_of_defined_values_in_column tests
// ============================================================================

#[test]
fn test_number_of_defined_values_in_column() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert_eq!(bm.number_of_defined_values_in_column(0), 0);
    assert_eq!(bm.number_of_defined_values_in_column(1), 1);
    assert_eq!(bm.number_of_defined_values_in_column(2), 2);
}

#[test]
fn test_number_of_defined_values_in_column_empty() {
    let bm = build_bimatrix(3, vec![]);

    for col in 0..3 {
        assert_eq!(bm.number_of_defined_values_in_column(col), 0);
    }
}

// ============================================================================
// SizedSparseBiMatrix2D::sparse_column_sizes tests
// ============================================================================

#[test]
fn test_sparse_column_sizes() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);

    let col_sizes: Vec<usize> = bm.sparse_column_sizes().collect();
    assert_eq!(col_sizes, vec![0, 1, 2]);
}

#[test]
fn test_sparse_column_sizes_symmetric() {
    // Symmetric edges: 0 <-> 1
    let bm = build_bimatrix(2, vec![(0, 1), (1, 0)]);

    let col_sizes: Vec<usize> = bm.sparse_column_sizes().collect();
    assert_eq!(col_sizes, vec![1, 1]);
}

#[test]
fn test_sparse_column_sizes_star_out() {
    // Star: 0 -> 1, 0 -> 2, 0 -> 3
    let bm = build_bimatrix(4, vec![(0, 1), (0, 2), (0, 3)]);

    let col_sizes: Vec<usize> = bm.sparse_column_sizes().collect();
    assert_eq!(col_sizes, vec![0, 1, 1, 1]);
}

#[test]
fn test_sparse_column_sizes_star_in() {
    // Star: 1 -> 0, 2 -> 0, 3 -> 0
    let bm = build_bimatrix(4, vec![(1, 0), (2, 0), (3, 0)]);

    let col_sizes: Vec<usize> = bm.sparse_column_sizes().collect();
    assert_eq!(col_sizes, vec![3, 0, 0, 0]);
}
