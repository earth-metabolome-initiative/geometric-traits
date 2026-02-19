//! Tests for RangedCSR2D: construction, MatrixMut, SparseMatrix2D,
//! EmptyRows, SizedRowsSparseMatrix2D, TransposableMatrix2D, and graph traits.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ranged_csr::RangedCSR2D,
    traits::{
        BipartiteGraph, Edges, EmptyRows, Graph, GrowableEdges, Matrix, Matrix2D, Matrix2DRef,
        MatrixMut, MonoplexGraph, SizedRowsSparseMatrix2D, SizedSparseMatrix, SparseMatrix,
        SparseMatrix2D, SparseMatrixMut, TransposableMatrix2D,
    },
};
use multi_ranged::SimpleRange;

type TestRangedCSR = RangedCSR2D<usize, usize, SimpleRange<usize>>;

// ============================================================================
// Default + Debug
// ============================================================================

#[test]
fn test_default() {
    let csr: TestRangedCSR = RangedCSR2D::default();
    assert_eq!(csr.number_of_rows(), 0);
    assert_eq!(csr.number_of_columns(), 0);
    assert!(csr.is_empty());
    assert_eq!(csr.number_of_defined_values(), 0);
}

#[test]
fn test_debug() {
    let csr: TestRangedCSR = RangedCSR2D::default();
    let debug = format!("{csr:?}");
    assert!(debug.contains("RangedCSR2D"));
}

// ============================================================================
// SparseMatrixMut constructors
// ============================================================================

#[test]
fn test_with_sparse_capacity() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_capacity(10);
    assert_eq!(csr.number_of_rows(), 0);
    assert!(csr.is_empty());
}

#[test]
fn test_with_sparse_shape() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 5));
    assert_eq!(csr.number_of_rows(), 3);
    assert_eq!(csr.number_of_columns(), 5);
    assert!(csr.is_empty());
}

#[test]
fn test_with_sparse_shaped_capacity() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shaped_capacity((4, 6), 10);
    assert_eq!(csr.number_of_rows(), 4);
    assert_eq!(csr.number_of_columns(), 6);
    assert!(csr.is_empty());
}

// ============================================================================
// MatrixMut::add - valid entries
// ============================================================================

#[test]
fn test_add_single_entry() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    assert!(!csr.is_empty());
    assert_eq!(csr.number_of_defined_values(), 1);
    assert_eq!(csr.number_of_rows(), 1);
    assert_eq!(csr.number_of_columns(), 1);
}

#[test]
fn test_add_consecutive_in_row() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 3);
    assert_eq!(csr.number_of_columns(), 3);
}

#[test]
fn test_add_multiple_rows() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 3)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 4);
    assert_eq!(csr.number_of_rows(), 2);
    assert_eq!(csr.number_of_columns(), 4);
}

#[test]
fn test_add_with_gap_rows() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (3, 1)).unwrap();
    assert_eq!(csr.number_of_rows(), 4);
    assert_eq!(csr.number_of_defined_values(), 2);
}

// ============================================================================
// MatrixMut::add - error paths
// ============================================================================

#[test]
fn test_add_duplicate_entry() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_non_consecutive_in_row() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    // SimpleRange requires consecutive entries; skipping to column 5 should fail
    let result = MatrixMut::add(&mut csr, (0, 5));
    assert!(result.is_err());
}

// ============================================================================
// MatrixMut::increase_shape
// ============================================================================

#[test]
fn test_increase_shape_valid() {
    let mut csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((2, 3));
    csr.increase_shape((5, 10)).unwrap();
    assert_eq!(csr.number_of_rows(), 5);
    assert_eq!(csr.number_of_columns(), 10);
}

#[test]
fn test_increase_shape_incompatible() {
    let mut csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((5, 5));
    let result = csr.increase_shape((2, 2));
    assert!(result.is_err());
}

// ============================================================================
// Matrix / Matrix2D / Matrix2DRef
// ============================================================================

#[test]
fn test_shape() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 5));
    assert_eq!(csr.shape(), vec![3, 5]);
}

#[test]
fn test_matrix2d_ref() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 5));
    assert_eq!(*csr.number_of_rows_ref(), 3);
    assert_eq!(*csr.number_of_columns_ref(), 5);
}

// ============================================================================
// SparseMatrix: last_sparse_coordinates, is_empty
// ============================================================================

#[test]
fn test_last_sparse_coordinates_empty() {
    let csr: TestRangedCSR = RangedCSR2D::default();
    assert_eq!(csr.last_sparse_coordinates(), None);
}

#[test]
fn test_last_sparse_coordinates_populated() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    assert_eq!(csr.last_sparse_coordinates(), Some((2, 3)));
}

// ============================================================================
// SparseMatrix2D: sparse_row, has_entry, sparse_coordinates
// ============================================================================

#[test]
fn test_sparse_row() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    let row0: Vec<usize> = csr.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1, 2]);
}

#[test]
fn test_has_entry() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    assert!(csr.has_entry(0, 0));
    assert!(csr.has_entry(0, 1));
    assert!(!csr.has_entry(0, 2));
}

#[test]
fn test_sparse_coordinates() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    let coords: Vec<(usize, usize)> =
        geometric_traits::traits::SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 0), (0, 1), (1, 2)]);
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_empty_rows() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    csr.increase_shape((4, 4)).unwrap();
    assert_eq!(csr.number_of_non_empty_rows(), 2);
    assert_eq!(csr.number_of_empty_rows(), 2);
}

#[test]
fn test_empty_row_indices() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    csr.increase_shape((4, 4)).unwrap();
    let empty: Vec<usize> = csr.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 3]);
}

#[test]
fn test_non_empty_row_indices() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    csr.increase_shape((4, 4)).unwrap();
    let non_empty: Vec<usize> = csr.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 2]);
}

// ============================================================================
// SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_number_of_defined_values_in_row() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 0)).unwrap();
    assert_eq!(csr.number_of_defined_values_in_row(0), 3);
    assert_eq!(csr.number_of_defined_values_in_row(1), 1);
}

#[test]
fn test_sparse_row_sizes() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    csr.increase_shape((3, 5)).unwrap();
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 0, 1]);
}

// ============================================================================
// TransposableMatrix2D
// ============================================================================

#[test]
fn test_transpose() {
    let mut csr: TestRangedCSR = RangedCSR2D::default();
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    let transposed: TestRangedCSR = csr.transpose();
    assert_eq!(transposed.number_of_rows(), csr.number_of_columns());
    assert_eq!(transposed.number_of_columns(), csr.number_of_rows());
    assert_eq!(transposed.number_of_defined_values(), csr.number_of_defined_values());
    // (0,0) -> (0,0), (0,1) -> (1,0), (1,2) -> (2,1)
    assert!(transposed.has_entry(0, 0));
    assert!(transposed.has_entry(1, 0));
    assert!(transposed.has_entry(2, 1));
}

// ============================================================================
// Graph traits (from ranged_csr.rs)
// ============================================================================

#[test]
fn test_graph_has_nodes_empty() {
    let csr: TestRangedCSR = RangedCSR2D::default();
    assert!(!Graph::has_nodes(&csr));
    assert!(!Graph::has_edges(&csr));
}

#[test]
fn test_graph_has_nodes_populated() {
    let mut csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    assert!(Graph::has_nodes(&csr));
    assert!(!Graph::has_edges(&csr));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    assert!(Graph::has_edges(&csr));
}

#[test]
fn test_edges_matrix() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    let _matrix = Edges::matrix(&csr);
}

#[test]
fn test_growable_edges_matrix_mut() {
    let mut csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    let _matrix = GrowableEdges::matrix_mut(&mut csr);
}

#[test]
fn test_growable_edges_constructors() {
    let _csr: TestRangedCSR = GrowableEdges::with_capacity(10_usize);
    let _csr: TestRangedCSR = GrowableEdges::with_shape((3_usize, 5_usize));
    let _csr: TestRangedCSR = GrowableEdges::with_shaped_capacity((3_usize, 5_usize), 10_usize);
}

#[test]
fn test_monoplex_graph_edges() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    let _edges = MonoplexGraph::edges(&csr);
}

#[test]
fn test_bipartite_graph() {
    let csr: TestRangedCSR = SparseMatrixMut::with_sparse_shape((3, 5));
    let _left = BipartiteGraph::left_nodes_vocabulary(&csr);
    let _right = BipartiteGraph::right_nodes_vocabulary(&csr);
}
