//! Extended tests for UpperTriangularCSR2D trait delegations.
//! Covers: SparseMatrix, SizedSparseMatrix, RankSelectSparseMatrix,
//! SparseMatrix2D, EmptyRows, SizedRowsSparseMatrix2D, SizedSparseMatrix2D,
//! MatrixMut, TransposableMatrix2D, Symmetrize, AsRef, SquareMatrix,
//! SparseSquareMatrix.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, UpperTriangularCSR2D},
    traits::{
        EmptyRows, Matrix, MatrixMut, RankSelectSparseMatrix, SizedRowsSparseMatrix2D,
        SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, SparseMatrixMut,
        SparseSquareMatrix, SquareMatrix, Symmetrize, TransposableMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestUT = UpperTriangularCSR2D<TestCSR>;

fn build_ut(entries: Vec<(usize, usize)>) -> TestUT {
    let mut ut: TestUT = UpperTriangularCSR2D::default();
    for (r, c) in entries {
        MatrixMut::add(&mut ut, (r, c)).unwrap();
    }
    ut
}

// ============================================================================
// SparseMatrix delegation
// ============================================================================

#[test]
fn test_ut_sparse_coordinates() {
    let ut = build_ut(vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]);
    let coords: Vec<(usize, usize)> = <TestUT as SparseMatrix>::sparse_coordinates(&ut).collect();
    assert_eq!(coords, vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]);
}

#[test]
fn test_ut_is_empty() {
    let ut: TestUT = UpperTriangularCSR2D::default();
    assert!(ut.is_empty());
    let ut2 = build_ut(vec![(0, 0)]);
    assert!(!ut2.is_empty());
}

#[test]
fn test_ut_last_sparse_coordinates() {
    let ut: TestUT = UpperTriangularCSR2D::default();
    assert_eq!(ut.last_sparse_coordinates(), None);
    let ut2 = build_ut(vec![(0, 0), (0, 1), (1, 2)]);
    assert_eq!(ut2.last_sparse_coordinates(), Some((1, 2)));
}

// ============================================================================
// SizedSparseMatrix
// ============================================================================

#[test]
fn test_ut_number_of_defined_values() {
    let ut = build_ut(vec![(0, 0), (0, 1), (1, 1)]);
    assert_eq!(ut.number_of_defined_values(), 3);
}

// ============================================================================
// RankSelectSparseMatrix
// ============================================================================

#[test]
fn test_ut_rank_select() {
    // Use one entry per row so select aligns with row boundaries
    let ut = build_ut(vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(ut.rank(&(0, 0)), 0);
    assert_eq!(ut.rank(&(1, 1)), 1);
    assert_eq!(ut.rank(&(2, 2)), 2);
    assert_eq!(ut.select(0), (0, 0));
    assert_eq!(ut.select(1), (1, 1));
    assert_eq!(ut.select(2), (2, 2));
}

// ============================================================================
// SparseMatrix2D
// ============================================================================

#[test]
fn test_ut_sparse_row() {
    let ut = build_ut(vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]);
    let row0: Vec<usize> = ut.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1, 2]);
    let row1: Vec<usize> = ut.sparse_row(1).collect();
    assert_eq!(row1, vec![1, 2]);
}

#[test]
fn test_ut_has_entry() {
    let ut = build_ut(vec![(0, 1), (1, 2)]);
    assert!(!ut.has_entry(0, 0));
    assert!(ut.has_entry(0, 1));
    assert!(!ut.has_entry(1, 0));
    assert!(ut.has_entry(1, 2));
}

#[test]
fn test_ut_sparse_columns() {
    let ut = build_ut(vec![(0, 0), (0, 1), (1, 1)]);
    let cols: Vec<usize> = ut.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 1]);
}

#[test]
fn test_ut_sparse_rows() {
    let ut = build_ut(vec![(0, 0), (0, 1), (1, 1)]);
    let rows: Vec<usize> = ut.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_ut_empty_rows() {
    let mut ut = build_ut(vec![(0, 0), (0, 1)]);
    ut.increase_shape((0, 0)).unwrap_or(());
    // Row 0 has entries, row 1 is empty
    let empty: Vec<usize> = ut.empty_row_indices().collect();
    assert_eq!(empty, vec![1]);
    let non_empty: Vec<usize> = ut.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
    assert_eq!(ut.number_of_empty_rows(), 1);
    assert_eq!(ut.number_of_non_empty_rows(), 1);
}

// ============================================================================
// SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_ut_number_of_defined_values_in_row() {
    let ut = build_ut(vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]);
    assert_eq!(ut.number_of_defined_values_in_row(0), 3);
    assert_eq!(ut.number_of_defined_values_in_row(1), 2);
}

#[test]
fn test_ut_sparse_row_sizes() {
    let ut = build_ut(vec![(0, 0), (0, 1), (1, 1)]);
    let sizes: Vec<usize> = ut.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1]);
}

// ============================================================================
// SizedSparseMatrix2D
// ============================================================================

#[test]
fn test_ut_rank_row_select_row_col() {
    // Use one entry per row so select aligns with row boundaries
    let ut = build_ut(vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(ut.rank_row(0), 0);
    assert_eq!(ut.rank_row(1), 1);
    assert_eq!(ut.rank_row(2), 2);
    assert_eq!(ut.select_row(0), 0);
    assert_eq!(ut.select_row(1), 1);
    assert_eq!(ut.select_row(2), 2);
    assert_eq!(ut.select_column(0), 0);
    assert_eq!(ut.select_column(1), 1);
    assert_eq!(ut.select_column(2), 2);
}

// ============================================================================
// MatrixMut - error paths
// ============================================================================

#[test]
fn test_ut_add_lower_triangular_error() {
    let mut ut: TestUT = UpperTriangularCSR2D::default();
    let result = MatrixMut::add(&mut ut, (2, 0));
    assert!(result.is_err());
}

#[test]
fn test_ut_increase_shape() {
    let mut ut: TestUT = SparseMatrixMut::with_sparse_shape(2);
    assert_eq!(ut.order(), 2);
    ut.increase_shape((4, 4)).unwrap();
    assert_eq!(ut.order(), 4);
}

// ============================================================================
// TransposableMatrix2D
// ============================================================================

#[test]
fn test_ut_transpose() {
    let ut = build_ut(vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]);
    let transposed: SquareCSR2D<TestCSR> = ut.transpose();
    // (0,0) -> (0,0)
    assert!(transposed.has_entry(0, 0));
    // (0,1) -> (1,0)
    assert!(transposed.has_entry(1, 0));
    // (0,2) -> (2,0)
    assert!(transposed.has_entry(2, 0));
    // (1,1) -> (1,1)
    assert!(transposed.has_entry(1, 1));
    // (1,2) -> (2,1)
    assert!(transposed.has_entry(2, 1));
}

// ============================================================================
// Symmetrize
// ============================================================================

#[test]
fn test_ut_symmetrize_with_diagonal() {
    let ut = build_ut(vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
    let sym = ut.symmetrize();
    // All original entries should be present
    assert!(sym.has_entry(0, 0));
    assert!(sym.has_entry(0, 1));
    assert!(sym.has_entry(0, 2));
    assert!(sym.has_entry(1, 1));
    assert!(sym.has_entry(1, 2));
    assert!(sym.has_entry(2, 2));
    // Symmetric counterparts should be added
    assert!(sym.has_entry(1, 0));
    assert!(sym.has_entry(2, 0));
    assert!(sym.has_entry(2, 1));
}

#[test]
fn test_ut_symmetrize_no_diagonal() {
    let ut = build_ut(vec![(0, 1), (0, 2), (1, 2)]);
    let sym = ut.symmetrize();
    assert!(sym.has_entry(0, 1));
    assert!(sym.has_entry(1, 0));
    assert!(sym.has_entry(0, 2));
    assert!(sym.has_entry(2, 0));
    assert!(sym.has_entry(1, 2));
    assert!(sym.has_entry(2, 1));
    // 3 original + 3 symmetric = 6
    assert_eq!(sym.number_of_defined_values(), 6);
}

// ============================================================================
// SquareMatrix / SparseSquareMatrix / AsRef
// ============================================================================

#[test]
fn test_ut_order() {
    let ut: TestUT = SparseMatrixMut::with_sparse_shape(5);
    assert_eq!(ut.order(), 5);
}

#[test]
fn test_ut_sparse_square_matrix() {
    let ut = build_ut(vec![(0, 0), (0, 1), (1, 1)]);
    assert_eq!(ut.number_of_defined_diagonal_values(), 2);
}

#[test]
fn test_ut_as_ref() {
    let ut = build_ut(vec![(0, 0), (0, 1)]);
    let inner: &TestCSR = ut.as_ref();
    assert_eq!(inner.number_of_defined_values(), 2);
}

// ============================================================================
// SparseMatrixMut constructors
// ============================================================================

#[test]
fn test_ut_with_sparse_capacity() {
    let ut: TestUT = SparseMatrixMut::with_sparse_capacity(10);
    assert!(ut.is_empty());
}

#[test]
fn test_ut_with_sparse_shaped_capacity() {
    let ut: TestUT = SparseMatrixMut::with_sparse_shaped_capacity(4, 10);
    assert_eq!(ut.order(), 4);
    assert!(ut.is_empty());
}

// ============================================================================
// Matrix (shape)
// ============================================================================

#[test]
fn test_ut_shape() {
    let ut: TestUT = SparseMatrixMut::with_sparse_shape(3);
    assert_eq!(ut.shape(), vec![3, 3]);
}

// ============================================================================
// Clone / Debug
// ============================================================================

#[test]
fn test_ut_clone() {
    let ut = build_ut(vec![(0, 0), (0, 1)]);
    let ut2 = ut.clone();
    assert_eq!(ut2.number_of_defined_values(), 2);
    assert_eq!(ut2.order(), ut.order());
}
