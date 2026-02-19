//! Tests for CSR2D error and edge-case paths in MatrixMut::add,
//! using small index types to trigger MaxedOut errors.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    traits::{MatrixMut, SparseMatrix, SparseMatrixMut},
};

// Use u8 indices to easily trigger overflow
type SmallCSR = CSR2D<u8, u8, u8>;

// ============================================================================
// MatrixMut::add error paths
// ============================================================================

#[test]
fn test_add_maxed_out_column_index() {
    // Try adding entry with column = u8::MAX (255) which should fail
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((1, 1));
    let result = MatrixMut::add(&mut csr, (0, 255));
    assert!(result.is_err());
}

#[test]
fn test_add_maxed_out_row_index_new_row() {
    // When adding a new row and number_of_non_empty_rows would exceed max
    // First fill up to a point, then try to add at row u8::MAX
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((1, 1));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    let result = MatrixMut::add(&mut csr, (255, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_maxed_out_sparse_index() {
    // Fill up the sparse index capacity with u8 (max 255)
    // We need 255 entries in one row to max out the sparse index
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_capacity(255);
    for col in 0..254u8 {
        MatrixMut::add(&mut csr, (0, col)).unwrap();
    }
    // At 254 entries, sparse index offset is 254. Adding one more should succeed
    MatrixMut::add(&mut csr, (0, 254)).unwrap();
    // Now the offset is 255 = u8::MAX. Next add should fail with
    // MaxedOutSparseIndex
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
    // Adding column 0 after column 1 in the same row is unordered
    let result = MatrixMut::add(&mut csr, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_add_unordered_row() {
    let mut csr: SmallCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    // Adding to row 1 after row 2 has been established is unordered
    let result = MatrixMut::add(&mut csr, (1, 0));
    assert!(result.is_err());
}

// ============================================================================
// Error Display formatting
// ============================================================================

#[test]
fn test_error_display_debug() {
    use geometric_traits::impls::MutabilityError;
    type TestCSR = CSR2D<usize, usize, usize>;

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

    // Debug should delegate to Display
    let debug = format!("{err:?}");
    assert!(debug.contains("out of expected bounds"));
}

#[test]
fn test_error_enum_display() {
    use geometric_traits::impls::{Error, MutabilityError};
    type TestCSR = CSR2D<usize, usize, usize>;

    let inner: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let err: Error<TestCSR> = Error::Mutability(inner);
    let display = format!("{err}");
    assert!(display.contains("Mutability error"));

    let debug = format!("{err:?}");
    assert!(debug.contains("Mutability error"));
}

// ============================================================================
// Empty CSR edge cases
// ============================================================================

#[test]
fn test_empty_csr_sparse_coordinates() {
    let csr: SmallCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    assert!(csr.is_empty());
    let coords: Vec<(u8, u8)> = csr.sparse_coordinates().collect();
    assert!(coords.is_empty());
}
