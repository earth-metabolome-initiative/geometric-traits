//! Tests for PaddedMatrix2D ImputedRowValues backward iteration,
//! PaddedCoordinates DoubleEndedIterator and ExactSizeIterator.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SparseMatrix, SparseValuedMatrix2D},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_padded_matrix(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> PaddedMatrix2D<TestValCSR, impl Fn((usize, usize)) -> f64 + Clone> {
    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    PaddedMatrix2D::new(inner, |(r, c): (usize, usize)| usize_to_f64(r * 100 + c)).unwrap()
}

// ============================================================================
// ImputedRowValues: backward iteration
// ============================================================================

#[test]
fn test_imputed_row_values_backward() {
    let padded = build_padded_matrix(vec![(0, 0, 5.0), (0, 1, 10.0)], 2, 3);
    // Row 0: dense columns [0, 1, 2]. Sparse at 0,1 → use 5.0, 10.0. Column 2
    // imputed → 2.0
    let vals_rev: Vec<f64> = padded.sparse_row_values(0).rev().collect();
    // Reversed: col 2 (imputed=2.0), col 1 (10.0), col 0 (5.0)
    assert_eq!(vals_rev, vec![2.0, 10.0, 5.0]);
}

#[test]
fn test_imputed_row_values_backward_all_imputed() {
    let padded = build_padded_matrix(vec![(0, 0, 5.0)], 2, 2);
    // Row 1 has no sparse entries; all columns imputed
    let vals_rev: Vec<f64> = padded.sparse_row_values(1).rev().collect();
    // map((1, 1)) = 101, map((1, 0)) = 100
    assert_eq!(vals_rev, vec![101.0, 100.0]);
}

#[test]
fn test_imputed_row_values_clone() {
    let padded = build_padded_matrix(vec![(0, 0, 5.0), (0, 1, 10.0)], 2, 3);
    let iter = padded.sparse_row_values(0);
    let cloned = iter.clone();
    let original: Vec<f64> = iter.collect();
    let from_clone: Vec<f64> = cloned.collect();
    assert_eq!(original, from_clone);
}

// ============================================================================
// PaddedCoordinates: DoubleEndedIterator and ExactSizeIterator
// ============================================================================

#[test]
fn test_padded_coordinates_forward() {
    let padded = build_padded_matrix(vec![(0, 0, 1.0)], 2, 2);
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&padded).collect();
    // PaddedCoordinates yields all dense (row, col) pairs; verify non-empty
    assert!(!coords.is_empty());
}

#[test]
fn test_padded_coordinates_exact_size() {
    let padded = build_padded_matrix(vec![(0, 0, 1.0)], 2, 2);
    let iter = SparseMatrix::sparse_coordinates(&padded);
    // Just verify len() runs without panicking
    let len = iter.len();
    assert!(len > 0);
}

// ============================================================================
// PaddedMatrix2D is_imputed
// ============================================================================

#[test]
fn test_padded_is_imputed() {
    let padded = build_padded_matrix(vec![(0, 0, 5.0), (0, 1, 10.0)], 2, 3);
    assert!(!padded.is_imputed((0, 0)));
    assert!(!padded.is_imputed((0, 1)));
    assert!(padded.is_imputed((0, 2)));
    assert!(padded.is_imputed((1, 0)));
    // Row/col beyond original matrix
    assert!(padded.is_imputed((2, 0)));
}

// ============================================================================
// PaddedMatrix2D Debug
// ============================================================================

#[test]
fn test_padded_debug() {
    let padded = build_padded_matrix(vec![(0, 0, 5.0)], 2, 2);
    let debug = format!("{padded:?}");
    // Should contain imputed values marked with I(...)
    assert!(debug.contains("I("));
    assert!(debug.contains("5.0"));
}

// ============================================================================
// PaddedMatrix2D dimensions
// ============================================================================

#[test]
fn test_padded_matrix_rectangular() {
    let padded = build_padded_matrix(vec![(0, 0, 1.0)], 2, 3);
    // 2x3 padded to 3x3
    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);
}
