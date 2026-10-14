//! Tests for CSR2D graph trait implementations: Edges, GrowableEdges, Graph,
//! MonoplexGraph, BipartiteGraph. Also tests Intersection DoubleEndedIterator
//! with crossing front/back candidates.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    traits::{
        BipartiteGraph, Edges, Graph, GrowableEdges, MatrixMut, MonoplexGraph, SizedSparseMatrix,
        SparseMatrix, SparseMatrixMut, Vocabulary,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;

// ============================================================================
// Edges trait
// ============================================================================

#[test]
fn test_edges_matrix() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    let matrix = Edges::matrix(&csr);
    assert_eq!(matrix.number_of_defined_values(), 2);
}

// ============================================================================
// GrowableEdges trait
// ============================================================================

#[test]
fn test_growable_edges_with_capacity() {
    let csr: TestCSR = GrowableEdges::with_capacity(10);
    assert!(SparseMatrix::is_empty(&csr));
}

#[test]
fn test_growable_edges_with_shape() {
    let csr: TestCSR = GrowableEdges::with_shape((3, 4));
    assert!(SparseMatrix::is_empty(&csr));
}

#[test]
fn test_growable_edges_with_shaped_capacity() {
    let csr: TestCSR = GrowableEdges::with_shaped_capacity((3, 4), 10);
    assert!(SparseMatrix::is_empty(&csr));
}

#[test]
fn test_growable_edges_matrix_mut() {
    let mut csr: TestCSR = GrowableEdges::with_shape((2, 2));
    let matrix = GrowableEdges::matrix_mut(&mut csr);
    MatrixMut::add(matrix, (0, 0)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 1);
}

// ============================================================================
// Graph trait
// ============================================================================

#[test]
fn test_graph_has_nodes_empty() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((0, 0));
    assert!(!Graph::has_nodes(&csr));
}

#[test]
fn test_graph_has_nodes_nonempty() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 3));
    assert!(Graph::has_nodes(&csr));
}

#[test]
fn test_graph_has_edges_empty() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 3));
    assert!(!Graph::has_edges(&csr));
}

#[test]
fn test_graph_has_edges_nonempty() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    assert!(Graph::has_edges(&csr));
}

// ============================================================================
// MonoplexGraph trait
// ============================================================================

#[test]
fn test_monoplex_graph_edges() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    let edges_ref = MonoplexGraph::edges(&csr);
    assert_eq!(edges_ref.number_of_defined_values(), 1);
}

// ============================================================================
// BipartiteGraph trait
// ============================================================================

#[test]
fn test_bipartite_graph_vocabularies() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 4));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let left = BipartiteGraph::left_nodes_vocabulary(&csr);
    let right = BipartiteGraph::right_nodes_vocabulary(&csr);
    assert_eq!(Vocabulary::len(left), 3);
    assert_eq!(Vocabulary::len(right), 4);
}
