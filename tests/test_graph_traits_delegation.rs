//! Tests for graph trait delegations on RaggedVector, ValuedCSR2D, and
//! GenericBiMatrix2D. Covers Edges, GrowableEdges, Graph, MonoplexGraph,
//! BipartiteGraph implementations.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, RaggedVector, SquareCSR2D, ValuedCSR2D},
    traits::{
        BipartiteGraph, Edges, Graph, GrowableEdges, Matrix2D, MatrixMut, MonoplexGraph,
        SparseMatrix, SparseMatrixMut,
    },
};

// ============================================================================
// RaggedVector graph traits (ragged_vec.rs — 0/20 coverage)
// ============================================================================

type TestRV = RaggedVector<usize, usize, usize>;

#[test]
fn test_rv_edges_matrix() {
    let rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    let m = Edges::matrix(&rv);
    assert_eq!(m.number_of_rows(), 3);
}

#[test]
fn test_rv_growable_edges_matrix_mut() {
    let mut rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    let m = GrowableEdges::matrix_mut(&mut rv);
    assert_eq!(m.number_of_rows(), 3);
}

#[test]
fn test_rv_growable_edges_with_capacity() {
    let rv: TestRV = GrowableEdges::with_capacity(10_usize);
    assert_eq!(rv.number_of_rows(), 0);
}

#[test]
fn test_rv_growable_edges_with_shape() {
    let rv: TestRV = GrowableEdges::with_shape((3_usize, 4_usize));
    assert_eq!(rv.number_of_rows(), 3);
    assert_eq!(rv.number_of_columns(), 4);
}

#[test]
fn test_rv_growable_edges_with_shaped_capacity() {
    let rv: TestRV = GrowableEdges::with_shaped_capacity((3_usize, 4_usize), 10_usize);
    assert_eq!(rv.number_of_rows(), 3);
}

#[test]
fn test_rv_graph_has_nodes_empty() {
    let rv: TestRV = RaggedVector::default();
    assert!(!Graph::has_nodes(&rv));
    assert!(!Graph::has_edges(&rv));
}

#[test]
fn test_rv_graph_has_nodes_populated() {
    let rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    assert!(Graph::has_nodes(&rv));
    assert!(!Graph::has_edges(&rv));
}

#[test]
fn test_rv_graph_has_edges() {
    let mut rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    MatrixMut::add(&mut rv, (0, 1)).unwrap();
    assert!(Graph::has_edges(&rv));
}

#[test]
fn test_rv_monoplex_graph_edges() {
    let rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    let _e = MonoplexGraph::edges(&rv);
}

#[test]
fn test_rv_bipartite_graph() {
    let rv: TestRV = SparseMatrixMut::with_sparse_shape((3, 4));
    let _left = BipartiteGraph::left_nodes_vocabulary(&rv);
    let _right = BipartiteGraph::right_nodes_vocabulary(&rv);
}

// ============================================================================
// ValuedCSR2D graph traits (valued_csr2d.rs — 4/20 coverage)
// ============================================================================

type TestVCSR = ValuedCSR2D<usize, usize, usize, f64>;

#[test]
fn test_vcsr_edges_matrix() {
    let vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    let m = Edges::matrix(&vcsr);
    assert_eq!(m.number_of_rows(), 3);
}

#[test]
fn test_vcsr_growable_edges_matrix_mut() {
    let mut vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    let m = GrowableEdges::matrix_mut(&mut vcsr);
    assert_eq!(m.number_of_rows(), 3);
}

#[test]
fn test_vcsr_growable_edges_with_capacity() {
    let vcsr: TestVCSR = GrowableEdges::with_capacity(10_usize);
    assert!(vcsr.is_empty());
}

#[test]
fn test_vcsr_growable_edges_with_shape() {
    let vcsr: TestVCSR = GrowableEdges::with_shape((3_usize, 4_usize));
    assert_eq!(vcsr.number_of_rows(), 3);
    assert_eq!(vcsr.number_of_columns(), 4);
}

#[test]
fn test_vcsr_growable_edges_with_shaped_capacity() {
    let vcsr: TestVCSR = GrowableEdges::with_shaped_capacity((3_usize, 4_usize), 10_usize);
    assert_eq!(vcsr.number_of_rows(), 3);
}

#[test]
fn test_vcsr_graph_has_nodes_empty() {
    let vcsr: TestVCSR = ValuedCSR2D::default();
    assert!(!Graph::has_nodes(&vcsr));
    assert!(!Graph::has_edges(&vcsr));
}

#[test]
fn test_vcsr_graph_has_nodes_populated() {
    let vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    assert!(Graph::has_nodes(&vcsr));
    assert!(!Graph::has_edges(&vcsr));
}

#[test]
fn test_vcsr_graph_has_edges() {
    let mut vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    MatrixMut::add(&mut vcsr, (0_usize, 1_usize, 1.5_f64)).unwrap();
    assert!(Graph::has_edges(&vcsr));
}

#[test]
fn test_vcsr_monoplex_graph_edges() {
    let vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    let _e = MonoplexGraph::edges(&vcsr);
}

#[test]
fn test_vcsr_bipartite_graph() {
    let vcsr: TestVCSR = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 5);
    let _left = BipartiteGraph::left_nodes_vocabulary(&vcsr);
    let _right = BipartiteGraph::right_nodes_vocabulary(&vcsr);
}

// ============================================================================
// GenericBiMatrix2D graph traits (generic_bimatrix.rs — 1/9 coverage)
// ============================================================================

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;

fn build_bimatrix(
    order: usize,
    entries: Vec<(usize, usize)>,
) -> GenericBiMatrix2D<TestSquareCSR, TestSquareCSR> {
    let mut m: TestSquareCSR = SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    m.extend(entries).unwrap();
    GenericBiMatrix2D::new(m)
}

#[test]
fn test_bimatrix_edges_matrix() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let m = Edges::matrix(&bm);
    assert_eq!(m.number_of_rows(), 3);
}

#[test]
fn test_bimatrix_graph_has_nodes_with_entries() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    assert!(Graph::has_nodes(&bm));
    assert!(Graph::has_edges(&bm));
}

#[test]
fn test_bimatrix_graph_no_edges() {
    let bm = build_bimatrix(3, vec![]);
    assert!(Graph::has_nodes(&bm));
    assert!(!Graph::has_edges(&bm));
}

#[test]
fn test_bimatrix_monoplex_graph_edges() {
    let bm = build_bimatrix(3, vec![(0, 1)]);
    let _e = MonoplexGraph::edges(&bm);
}
