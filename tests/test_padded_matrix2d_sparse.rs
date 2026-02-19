//! Tests for PaddedMatrix2D sparse matrix trait implementations:
//! SparseMatrix, SizedSparseMatrix, SparseMatrix2D, SizedRowsSparseMatrix2D,
//! EmptyRows.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, SizedRowsSparseMatrix2D, SizedSparseMatrix, SparseMatrix,
        SparseMatrix2D,
    },
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_padded(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> PaddedMatrix2D<TestValCSR, impl Fn((usize, usize)) -> f64> {
    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap()
}

// ============================================================================
// SparseMatrix: is_empty, last_sparse_coordinates, sparse_coordinates
// ============================================================================

#[test]
fn test_padded_is_empty_false() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    assert!(!padded.is_empty());
}

#[test]
fn test_padded_last_sparse_coordinates() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3, last coordinates = (2, 2)
    let last = padded.last_sparse_coordinates();
    assert_eq!(last, Some((2, 2)));
}

#[test]
fn test_padded_sparse_coordinates_yields_entries() {
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)], 2, 2);
    let coords: Vec<(usize, usize)> = padded.sparse_coordinates().collect();
    // Verify all expected entries appear in the output
    assert!(coords.contains(&(0, 0)));
    assert!(coords.contains(&(0, 1)));
    assert!(coords.contains(&(1, 0)));
    assert!(coords.contains(&(1, 1)));
}

// ============================================================================
// SizedSparseMatrix: number_of_defined_values
// ============================================================================

#[test]
fn test_padded_number_of_defined_values() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3, so 9 total
    assert_eq!(padded.number_of_defined_values(), 9);
}

// ============================================================================
// SparseMatrix2D: sparse_row, has_entry, sparse_columns, sparse_rows
// ============================================================================

#[test]
fn test_padded_sparse_row() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3, every row has all columns
    let row0: Vec<usize> = padded.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1, 2]);
    let row2: Vec<usize> = padded.sparse_row(2).collect();
    assert_eq!(row2, vec![0, 1, 2]);
}

#[test]
fn test_padded_has_entry_always_true() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded: all positions are "entries"
    assert!(padded.has_entry(0, 0));
    assert!(padded.has_entry(0, 2));
    assert!(padded.has_entry(2, 2));
}

#[test]
fn test_padded_sparse_columns() {
    let padded = build_padded(vec![(0, 0, 1.0)], 1, 2);
    // Padded to 2x2: row0=[0,1], row1=[0,1]
    let cols: Vec<usize> = padded.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 0, 1]);
}

#[test]
fn test_padded_sparse_rows() {
    let padded = build_padded(vec![(0, 0, 1.0)], 1, 2);
    // Padded to 2x2: row0 has 2 entries, row1 has 2 entries
    let rows: Vec<usize> = padded.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1, 1]);
}

// ============================================================================
// SizedRowsSparseMatrix2D: sparse_row_sizes, number_of_defined_values_in_row
// ============================================================================

#[test]
fn test_padded_number_of_defined_values_in_row() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3, every row has 3 columns
    assert_eq!(padded.number_of_defined_values_in_row(0), 3);
    assert_eq!(padded.number_of_defined_values_in_row(1), 3);
    assert_eq!(padded.number_of_defined_values_in_row(2), 3);
}

#[test]
fn test_padded_sparse_row_sizes() {
    let padded = build_padded(vec![(0, 0, 1.0)], 1, 2);
    // Padded to 2x2
    let sizes: Vec<usize> = padded.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 2]);
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_padded_empty_row_indices() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded: no empty rows
    let empty: Vec<usize> = padded.empty_row_indices().collect();
    assert!(empty.is_empty());
}

#[test]
fn test_padded_non_empty_row_indices() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3: all rows non-empty
    let non_empty: Vec<usize> = padded.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 1, 2]);
}

#[test]
fn test_padded_number_of_empty_rows() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    assert_eq!(padded.number_of_empty_rows(), 0);
    assert_eq!(padded.number_of_non_empty_rows(), 3);
}
