//! Extended tests for the `PaddedMatrix2D` struct.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    traits::{Matrix2D, MatrixMut, SparseMatrix, SparseMatrixMut, SparseValuedMatrix2D},
};

fn build_valued_csr(rows: u8, cols: u8, entries: Vec<(u8, u8, u8)>) -> ValuedCSR2D<u8, u8, u8, u8> {
    let mut csr: ValuedCSR2D<u8, u8, u8, u8> = ValuedCSR2D::with_sparse_shaped_capacity(
        (rows, cols),
        u8::try_from(entries.len()).expect("test entries should fit in u8"),
    );
    for entry in entries {
        MatrixMut::add(&mut csr, entry).expect("Failed to add value");
    }
    csr
}

// ============================================================================
// Construction
// ============================================================================

#[test]
fn test_padded_new_success() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 10)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0);
    assert!(padded.is_ok());
}

// ============================================================================
// is_imputed
// ============================================================================

#[test]
fn test_padded_is_imputed() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 10), (1, 1, 20)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();

    // (0,0) has a real value, not imputed
    assert!(!padded.is_imputed((0, 0)));
    // (0,1) is not in the sparse matrix, should be imputed
    assert!(padded.is_imputed((0, 1)));
    // (1,0) is not in the sparse matrix, should be imputed
    assert!(padded.is_imputed((1, 0)));
    // (1,1) has a real value, not imputed
    assert!(!padded.is_imputed((1, 1)));
}

#[test]
fn test_padded_is_imputed_out_of_bounds() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 10)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();

    // Out of bounds coordinates are imputed
    assert!(padded.is_imputed((5, 5)));
    assert!(padded.is_imputed((0, 5)));
    assert!(padded.is_imputed((5, 0)));
}

// ============================================================================
// Sparse row values
// ============================================================================

#[test]
fn test_padded_sparse_row_values() {
    let csr = build_valued_csr(2, 3, vec![(0, 1, 42), (1, 2, 99)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 1).unwrap();

    // Row 0: columns 0(imputed=1), 1(real=42), 2(imputed=1)
    let row0_values: Vec<u8> = padded.sparse_row_values(0).collect();
    assert_eq!(row0_values, vec![1, 42, 1]);

    // Row 1: columns 0(imputed=1), 1(imputed=1), 2(real=99)
    let row1_values: Vec<u8> = padded.sparse_row_values(1).collect();
    assert_eq!(row1_values, vec![1, 1, 99]);
}

// ============================================================================
// Sparse coordinates
// ============================================================================

#[test]
fn test_padded_sparse_coordinates_nonempty() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 10), (1, 1, 20)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();

    let coords: Vec<(u8, u8)> = padded.sparse_coordinates().collect();
    // Padded matrix always has entries (it fills all cells)
    assert!(!coords.is_empty());
    // Must include the originally defined entries
    assert!(coords.contains(&(0, 0)));
    assert!(coords.contains(&(1, 1)));
}

// ============================================================================
// Debug impl
// ============================================================================

#[test]
fn test_padded_debug() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 5), (1, 1, 9)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();

    let debug_str = format!("{padded:?}");
    // Should contain I(0) for imputed values and real values
    assert!(debug_str.contains("I("), "Debug should show imputed values with I() prefix");
    assert!(debug_str.contains('5'), "Debug should show real value 5");
    assert!(debug_str.contains('9'), "Debug should show real value 9");
}

// ============================================================================
// Matrix2D trait
// ============================================================================

#[test]
fn test_padded_number_of_rows_cols() {
    let csr = build_valued_csr(3, 4, vec![(0, 0, 1)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();

    // PaddedMatrix2D uses max(rows, cols) to make it square
    assert_eq!(padded.number_of_rows(), 4);
    assert_eq!(padded.number_of_columns(), 4);
}

#[test]
fn test_padded_is_empty() {
    let csr = build_valued_csr(2, 2, vec![(0, 0, 1)]);
    let padded = PaddedMatrix2D::new(&csr, |_: (u8, u8)| 0).unwrap();
    // Padded matrix is never empty because it fills all cells
    assert!(!padded.is_empty());
}
