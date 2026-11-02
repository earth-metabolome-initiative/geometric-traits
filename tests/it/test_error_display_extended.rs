//! Tests for Error Display/Debug paths that remain uncovered:
//! - Error::Mutability Display (error.rs line 25-26)
//! - MutabilityError Display for OutOfBounds (line 78)
//! - MutabilityError Debug delegation to Display
//! - Error Debug delegation
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, MutabilityError},
    traits::{MatrixMut, SparseMatrixMut},
};

type TestCSR = CSR2D<usize, usize, usize>;

// ============================================================================
// MutabilityError Display variants coverage
// ============================================================================

#[test]
fn test_mutability_error_out_of_bounds_display() {
    let err: MutabilityError<TestCSR> =
        MutabilityError::OutOfBounds((5, 6), (10, 10), "test context");
    let display = format!("{err}");
    assert!(display.contains("out of"));
    assert!(display.contains("bounds"));
    assert!(display.contains("test context"));
}

#[test]
fn test_mutability_error_maxed_out_row_display() {
    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let display = format!("{err}");
    assert!(display.contains("Row index"));
}

#[test]
fn test_mutability_error_maxed_out_column_display() {
    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let display = format!("{err}");
    assert!(display.contains("Column index"));
}

#[test]
fn test_mutability_error_maxed_out_sparse_display() {
    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let display = format!("{err}");
    assert!(display.contains("Sparse index"));
}

#[test]
fn test_mutability_error_incompatible_shape_display() {
    let err: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let display = format!("{err}");
    assert!(display.contains("shape"));
}

#[test]
fn test_mutability_error_debug_delegates_to_display() {
    let err: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let debug = format!("{err:?}");
    let display = format!("{err}");
    assert_eq!(debug, display);
}

// ============================================================================
// Error Debug/Display
// ============================================================================

#[test]
fn test_error_display_mutability() {
    use geometric_traits::impls::Error;
    let err: Error<TestCSR> = Error::Mutability(MutabilityError::DuplicatedEntry((3, 4)));
    let display = format!("{err}");
    assert!(display.contains("Mutability error"));
    assert!(display.contains("Duplicated entry"));
}

#[test]
fn test_error_debug_delegates_to_display() {
    use geometric_traits::impls::Error;
    let err: Error<TestCSR> = Error::Mutability(MutabilityError::MaxedOutRowIndex);
    let debug = format!("{err:?}");
    let display = format!("{err}");
    assert_eq!(debug, display);
}

// ============================================================================
// Trigger actual MutabilityError through MatrixMut operations
// ============================================================================

#[test]
fn test_incompatible_shape_error() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    let result = csr.increase_shape((2, 3));
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = format!("{err}");
    assert!(display.contains("shape"));
}

#[test]
fn test_duplicate_entry_error_display() {
    let mut csr: TestCSR = CSR2D::default();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 1));
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = format!("{err}");
    assert!(display.contains("Duplicated entry"));
}

#[test]
fn test_unordered_coordinate_error_display() {
    let mut csr: TestCSR = CSR2D::default();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    let result = MatrixMut::add(&mut csr, (0, 1));
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = format!("{err}");
    assert!(display.contains("Unordered coordinate"));
}

// ============================================================================
// Error as std::error::Error trait coverage
// ============================================================================

#[test]
fn test_error_is_std_error() {
    use geometric_traits::impls::Error;
    let err: Error<TestCSR> = Error::Mutability(MutabilityError::MaxedOutRowIndex);
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_mutability_error_is_std_error() {
    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let _: &dyn std::error::Error = &err;
}
