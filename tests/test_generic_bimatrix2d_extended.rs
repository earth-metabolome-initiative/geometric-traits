//! Extended tests for GenericBiMatrix2D: additional trait delegations beyond
//! what test_generic_bimatrix2d.rs covers.
//! Covers: SparseMatrix, SizedSparseMatrix, RankSelectSparseMatrix,
//! SparseMatrix2D, EmptyRows, SizedRowsSparseMatrix2D, SizedSparseMatrix2D,
//! TransposableMatrix2D, BiMatrix2D, SquareMatrix, SparseSquareMatrix, Matrix
//! shape.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SquareCSR2D},
    traits::{
        BiMatrix2D, EmptyRows, Matrix, Matrix2D, MatrixMut, RankSelectSparseMatrix,
        SizedRowsSparseMatrix2D, SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrix,
        SparseMatrix2D, SparseMatrixMut, SparseSquareMatrix, SquareMatrix, TransposableMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestBiMatrix = GenericBiMatrix2D<TestSquareCSR, TestSquareCSR>;

fn build_bimatrix(order: usize, entries: Vec<(usize, usize)>) -> TestBiMatrix {
    let mut m: TestSquareCSR = SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    m.extend(entries).unwrap();
    GenericBiMatrix2D::new(m)
}

// ============================================================================
// Matrix / Matrix2D
// ============================================================================

#[test]
fn test_bimatrix_shape() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    assert_eq!(bm.shape(), vec![3, 3]);
    assert_eq!(bm.number_of_rows(), 3);
    assert_eq!(bm.number_of_columns(), 3);
}

// ============================================================================
// SquareMatrix / SparseSquareMatrix
// ============================================================================

#[test]
fn test_bimatrix_order() {
    let bm = build_bimatrix(4, vec![(0, 1)]);
    assert_eq!(bm.order(), 4);
}

#[test]
fn test_bimatrix_diagonal_values() {
    let bm = build_bimatrix(3, vec![(0, 0), (0, 1), (1, 1), (2, 2)]);
    assert_eq!(bm.number_of_defined_diagonal_values(), 3);
}

// ============================================================================
// SparseMatrix
// ============================================================================

#[test]
fn test_bimatrix_sparse_coordinates() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let coords: Vec<(usize, usize)> =
        <TestBiMatrix as SparseMatrix>::sparse_coordinates(&bm).collect();
    assert_eq!(coords, vec![(0, 1), (1, 2)]);
}

#[test]
fn test_bimatrix_last_sparse_coordinates() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    assert_eq!(bm.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_bimatrix_is_empty() {
    let bm = build_bimatrix(3, vec![]);
    assert!(bm.is_empty());
    let bm2 = build_bimatrix(3, vec![(0, 1)]);
    assert!(!bm2.is_empty());
}

// ============================================================================
// SizedSparseMatrix
// ============================================================================

#[test]
fn test_bimatrix_number_of_defined_values() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2), (2, 0)]);
    assert_eq!(bm.number_of_defined_values(), 3);
}

// ============================================================================
// RankSelectSparseMatrix
// ============================================================================

#[test]
fn test_bimatrix_rank_select() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    assert_eq!(bm.rank(&(0, 1)), 0);
    assert_eq!(bm.rank(&(1, 2)), 1);
    assert_eq!(bm.select(0), (0, 1));
    assert_eq!(bm.select(1), (1, 2));
}

// ============================================================================
// SparseMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_sparse_row() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    let row0: Vec<usize> = bm.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 2]);
    let row1: Vec<usize> = bm.sparse_row(1).collect();
    assert_eq!(row1, vec![2]);
}

#[test]
fn test_bimatrix_has_entry() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    assert!(!bm.has_entry(0, 0));
    assert!(bm.has_entry(0, 1));
    assert!(bm.has_entry(1, 2));
    assert!(!bm.has_entry(2, 0));
}

#[test]
fn test_bimatrix_sparse_columns() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    let cols: Vec<usize> = bm.sparse_columns().collect();
    assert_eq!(cols, vec![1, 2, 2]);
}

#[test]
fn test_bimatrix_sparse_rows() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    let rows: Vec<usize> = bm.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_bimatrix_empty_rows() {
    let bm = build_bimatrix(3, vec![(0, 1)]);
    assert_eq!(bm.number_of_empty_rows(), 2);
    assert_eq!(bm.number_of_non_empty_rows(), 1);
    let empty: Vec<usize> = bm.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 2]);
    let non_empty: Vec<usize> = bm.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
}

// ============================================================================
// SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_sized_rows() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(bm.number_of_defined_values_in_row(0), 2);
    assert_eq!(bm.number_of_defined_values_in_row(1), 1);
    assert_eq!(bm.number_of_defined_values_in_row(2), 0);
    let sizes: Vec<usize> = bm.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1, 0]);
}

// ============================================================================
// SizedSparseMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_sized_sparse_matrix2d() {
    let bm = build_bimatrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(bm.rank_row(0), 0);
    assert_eq!(bm.rank_row(1), 2);
    assert_eq!(bm.rank_row(2), 3);
    assert_eq!(bm.select_row(0), 0);
    assert_eq!(bm.select_row(2), 1);
    assert_eq!(bm.select_column(0), 1);
    assert_eq!(bm.select_column(1), 2);
}

// ============================================================================
// TransposableMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_transpose() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let transposed: TestSquareCSR = bm.transpose();
    assert!(transposed.has_entry(1, 0));
    assert!(transposed.has_entry(2, 1));
    assert!(!transposed.has_entry(0, 1));
}

// ============================================================================
// BiMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_matrix_accessor() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let m = BiMatrix2D::matrix(&bm);
    assert_eq!(m.number_of_rows(), 3);
    assert_eq!(m.number_of_defined_values(), 2);
}

#[test]
fn test_bimatrix_transposed_accessor() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let t = BiMatrix2D::transposed(&bm);
    assert_eq!(t.number_of_rows(), 3);
    // Transposed should have same number of entries
    assert_eq!(t.number_of_defined_values(), 2);
    assert!(t.has_entry(1, 0));
    assert!(t.has_entry(2, 1));
}

// ============================================================================
// Clone / Debug / PartialEq
// ============================================================================

#[test]
fn test_bimatrix_clone_eq() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let bm2 = bm.clone();
    assert_eq!(bm, bm2);
}

#[test]
fn test_bimatrix_debug() {
    let bm = build_bimatrix(2, vec![(0, 1)]);
    let debug = format!("{bm:?}");
    assert!(debug.contains("GenericBiMatrix2D"));
}
