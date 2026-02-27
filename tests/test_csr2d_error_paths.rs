//! Tests for CSR2D error and edge-case paths in MatrixMut::add,
//! using small index types to trigger MaxedOut errors.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    prelude::*,
    traits::{MatrixMut, SparseMatrix, SparseMatrix2D, SparseMatrixMut},
};

// Use u8 indices to easily trigger overflow
type SmallCSR = CSR2D<u8, u8, u8>;
type TestCSR = CSR2D<usize, usize, usize>;

// ============================================================================
// MatrixMut::add error paths
// ============================================================================

#[test]
fn test_add_maxed_out_column_index() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((1, 1));
    let result = MatrixMut::add(&mut csr, (0, 255));
    assert!(result.is_err());
}

#[test]
fn test_add_maxed_out_column_index_same_row() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 255));
    assert!(result.is_err());
}

#[test]
fn test_add_maxed_out_row_index_new_row() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((1, 1));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (255, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_maxed_out_sparse_index() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_capacity(255);
    for col in 0..254u8 {
        MatrixMut::add(&mut csr, (0, col)).unwrap();
    }
    MatrixMut::add(&mut csr, (0, 254)).unwrap();
    let result = MatrixMut::add(&mut csr, (1, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_duplicate_entry() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_unordered_column() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_unordered_row() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (1, 0));
    assert!(result.is_err());
}

#[test]
fn test_increase_shape_incompatible() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((5, 5), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = csr.increase_shape((3, 3));
    assert!(result.is_err());
}

#[test]
fn test_increase_shape_valid() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = csr.increase_shape((5, 5));
    assert!(result.is_ok());
    assert_eq!(csr.number_of_rows(), 5);
    assert_eq!(csr.number_of_columns(), 5);
}

// ============================================================================
// Error Display formatting
// ============================================================================

#[test]
fn test_error_display_debug() {
    use geometric_traits::impls::MutabilityError;

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let display = format!("{err}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let display = format!("{err}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let display = format!("{err}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((0, 0), (1, 1), "test");
    let display = format!("{err}");
    assert!(display.contains("out of expected bounds"));

    let debug = format!("{err:?}");
    assert!(debug.contains("out of expected bounds"));
}

#[test]
fn test_error_enum_display() {
    use geometric_traits::impls::{Error, MutabilityError};

    let inner: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let err: Error<TestCSR> = Error::Mutability(inner);
    let display = format!("{err}");
    assert!(display.contains("Mutability error"));

    let debug = format!("{err:?}");
    assert!(debug.contains("Mutability error"));
}

// ============================================================================
// RankSelectSparseMatrix methods
// ============================================================================

#[test]
fn test_rank_of_entries() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 4);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();

    assert_eq!(csr.rank(&(0, 0)), 0);
    assert_eq!(csr.rank(&(0, 2)), 1);
    assert_eq!(csr.rank(&(1, 1)), 2);
    assert_eq!(csr.rank(&(2, 0)), 3);
}

// ============================================================================
// last_sparse_coordinates
// ============================================================================

#[test]
fn test_last_sparse_coordinates_empty() {
    let csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 0);
    assert!(csr.last_sparse_coordinates().is_none());
}

#[test]
fn test_last_sparse_coordinates_nonempty() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    assert_eq!(csr.last_sparse_coordinates(), Some((2, 1)));
}

// ============================================================================
// sparse_rows with empty rows in between
// ============================================================================

#[test]
fn test_sparse_rows_consecutive() {
    // Use consecutive rows (no gaps) to test sparse_rows properly
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 4);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    let rows: Vec<usize> = <TestCSR as SparseMatrix2D>::sparse_rows(&csr).collect();
    assert_eq!(rows, vec![0, 0, 1, 2]);
    assert_eq!(<TestCSR as SparseMatrix2D>::sparse_rows(&csr).len(), 4);
}

// ============================================================================
// has_entry
// ============================================================================

#[test]
fn test_has_entry() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 3);
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    assert!(csr.has_entry(0, 1));
    assert!(!csr.has_entry(0, 0));
    assert!(csr.has_entry(1, 2));
    assert!(!csr.has_entry(1, 0));
}

// ============================================================================
// sparse_coordinates (CSR2DView)
// ============================================================================

#[test]
fn test_sparse_coordinates() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 4), 4);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 0), (0, 2), (1, 1), (2, 3)]);
}

#[test]
fn test_sparse_coordinates_len() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 4), 4);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    assert_eq!(SparseMatrix::sparse_coordinates(&csr).len(), 4);
}

#[test]
fn test_sparse_coordinates_forward() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 4), 4);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.next(), Some((0, 0)));
    assert_eq!(iter.next(), Some((0, 2)));
    assert_eq!(iter.next(), Some((1, 1)));
    assert_eq!(iter.next(), Some((2, 3)));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// Empty/non-empty row indices
// ============================================================================

#[test]
fn test_empty_row_indices() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((4, 4), 2);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    let empty: Vec<usize> = csr.empty_row_indices().collect();
    assert!(empty.contains(&1));
    assert!(empty.contains(&3));
    assert!(!empty.contains(&0));
    assert!(!empty.contains(&2));
}

#[test]
fn test_non_empty_row_indices() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((4, 4), 2);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1)).unwrap();
    let non_empty: Vec<usize> = csr.non_empty_row_indices().collect();
    assert!(non_empty.contains(&0));
    assert!(non_empty.contains(&2));
}

// ============================================================================
// Empty CSR edge cases
// ============================================================================

#[test]
fn test_empty_csr_sparse_coordinates() {
    let csr: SmallCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    assert!(csr.is_empty());
    let coords: Vec<(u8, u8)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert!(coords.is_empty());
}

#[test]
fn test_sparse_row_sizes() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 4), 5);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![3, 1, 1]);
}
