//! Extended tests for GenericMatrix2DWithPaddedDiagonal: error paths,
//! is_diagonal_imputed, sparse_rows backward iteration,
//! SparseRowsWithPaddedDiagonal, and PaddedCoordinates
//! DoubleEndedIterator/ExactSizeIterator.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, EmptyRows, SparseMatrix, SparseMatrix2D, SparseValuedMatrix},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_padded(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> GenericMatrix2DWithPaddedDiagonal<TestValCSR, impl Fn(usize) -> f64 + Clone> {
    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericMatrix2DWithPaddedDiagonal::new(inner, |row: usize| usize_to_f64(row + 1) * 10.0)
        .unwrap()
}

// ============================================================================
// is_diagonal_imputed
// ============================================================================

#[test]
fn test_is_diagonal_imputed_when_present() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    assert!(!padded.is_diagonal_imputed(0));
}

#[test]
fn test_is_diagonal_imputed_when_absent() {
    let padded = build_padded(vec![(0, 1, 1.0)], 2, 2);
    assert!(padded.is_diagonal_imputed(0));
}

#[test]
fn test_is_diagonal_imputed_extra_row() {
    // Row 2 doesn't exist in the 2x3 matrix, so diagonal is always imputed
    let padded = build_padded(vec![(0, 1, 1.0)], 2, 3);
    assert!(padded.is_diagonal_imputed(2));
}

// ============================================================================
// matrix() accessor
// ============================================================================

#[test]
fn test_matrix_accessor() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 2);
    let inner = padded.matrix();
    assert_eq!(inner.number_of_defined_values(), 1);
}

// ============================================================================
// EmptyRows: no empty rows in padded diagonal
// ============================================================================

#[test]
fn test_padded_empty_rows() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 2);
    assert_eq!(padded.number_of_empty_rows(), 0);
    assert_eq!(padded.number_of_non_empty_rows(), 2);
    let empty: Vec<usize> = padded.empty_row_indices().collect();
    assert!(empty.is_empty());
    let non_empty: Vec<usize> = padded.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 1]);
}

// ============================================================================
// SparseMatrix: last_sparse_coordinates, is_empty, sparse_coordinates
// ============================================================================

#[test]
fn test_padded_last_sparse_coordinates() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 2);
    assert_eq!(padded.last_sparse_coordinates(), Some((1, 1)));
}

#[test]
fn test_padded_sparse_coordinates_forward() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let coords: Vec<(usize, usize)> = padded.sparse_coordinates().collect();
    // Row 0: [0, 1] (0 already present), Row 1: [1] (imputed diagonal)
    assert_eq!(coords, vec![(0, 0), (0, 1), (1, 1)]);
}

#[test]
fn test_padded_sparse_values() {
    let padded = build_padded(vec![(0, 1, 1.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_values().collect();
    // Row 0: imputed diagonal=10.0, then col 1=1.0. Row 1: imputed diagonal=20.0
    assert_eq!(vals, vec![10.0, 1.0, 20.0]);
}

// ============================================================================
// SparseRowsWithPaddedDiagonal: backward iteration
// ============================================================================

#[test]
fn test_sparse_rows_padded_forward() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let rows: Vec<usize> = padded.sparse_rows().collect();
    // Row 0 has 2 entries, row 1 has 1 (imputed diagonal)
    assert_eq!(rows, vec![0, 0, 1]);
}

#[test]
fn test_sparse_rows_padded_backward() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let rows: Vec<usize> = padded.sparse_rows().rev().collect();
    assert_eq!(rows, vec![1, 0, 0]);
}

#[test]
fn test_sparse_rows_padded_mixed() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let mut iter = padded.sparse_rows();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// Rectangular matrix padded to square
// ============================================================================

#[test]
fn test_rectangular_padded_to_square() {
    // 2x3 matrix, padded to 3x3
    let padded = build_padded(vec![(0, 1, 1.0), (1, 2, 2.0)], 2, 3);
    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);
    assert_eq!(padded.shape(), vec![3, 3]);
}

#[test]
fn test_tall_matrix_padded_to_square() {
    // 3x2 matrix, padded to 3x3
    let padded = build_padded(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 0, 3.0)], 3, 2);
    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);
}

// ============================================================================
// has_entry
// ============================================================================

#[test]
fn test_padded_has_entry() {
    let padded = build_padded(vec![(0, 1, 1.0)], 2, 2);
    // Original entry
    assert!(padded.has_entry(0, 1));
    // Imputed diagonal for row 0
    assert!(padded.has_entry(0, 0));
    // Imputed diagonal for row 1
    assert!(padded.has_entry(1, 1));
    // Not present
    assert!(!padded.has_entry(1, 0));
}

// ============================================================================
// sparse_columns on padded
// ============================================================================

#[test]
fn test_padded_sparse_columns() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let cols: Vec<usize> = padded.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 1]);
}
