//! Tests for PaddedMatrix2D: Debug output, is_imputed, and SparseValuedMatrix2D
//! methods.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SparseValuedMatrix2D},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_valued_csr(edges: Vec<(usize, usize, f64)>, rows: usize, cols: usize) -> TestValCSR {
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Debug impl tests
// ============================================================================

#[test]
fn test_padded_debug_output() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    let debug = format!("{padded:?}");
    // Debug format shows I(value) for imputed values
    assert!(debug.contains("I("));
}

#[test]
fn test_padded_debug_no_imputed() {
    // All entries present: 2x2 with all 4 entries
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    let debug = format!("{padded:?}");
    assert!(!debug.is_empty());
}

// ============================================================================
// is_imputed tests
// ============================================================================

#[test]
fn test_is_imputed_within_sparse_entry() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    // (0,1) is present in sparse data
    assert!(!padded.is_imputed((0, 1)));
    // (0,0) is not present in sparse data
    assert!(padded.is_imputed((0, 0)));
    // (1,0) is not present
    assert!(padded.is_imputed((1, 0)));
    // (1,1) is not present
    assert!(padded.is_imputed((1, 1)));
}

#[test]
fn test_is_imputed_beyond_bounds() {
    let inner = build_valued_csr(vec![(0, 0, 1.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    // Beyond original matrix bounds
    assert!(padded.is_imputed((5, 5)));
}

// ============================================================================
// sparse_row_max/min_value tests (SparseValuedMatrix2D)
// ============================================================================

#[test]
fn test_sparse_row_max_value() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    // Row 0 has sparse values 1.0, 5.0, 3.0 — max is 5.0
    assert_eq!(padded.sparse_row_max_value(0), Some(5.0));
}

#[test]
fn test_sparse_row_max_value_imputed_row() {
    // 1x3 matrix padded to 3x3, rows 1 and 2 are fully imputed
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| -2.0).unwrap();

    // Row 1 is fully imputed with -2.0
    assert_eq!(padded.sparse_row_max_value(1), Some(-2.0));
}

#[test]
fn test_sparse_row_min_value() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    // Row 0 has sparse values 1.0, 5.0, 3.0 — min is 1.0
    assert_eq!(padded.sparse_row_min_value(0), Some(1.0));
}

#[test]
fn test_sparse_row_min_value_imputed_row() {
    let inner = build_valued_csr(vec![(0, 0, 1.0)], 1, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| -1.0).unwrap();

    // Row 1 (imputed row) has all values -1.0
    assert_eq!(padded.sparse_row_min_value(1), Some(-1.0));
}

#[test]
fn test_sparse_row_max_value_and_column() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0)], 1, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    let result = padded.sparse_row_max_value_and_column(0);
    assert_eq!(result, Some((5.0, 1)));
}

#[test]
fn test_sparse_row_min_value_and_column() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0)], 1, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    // Row 0 has values 1.0, 5.0 — min is 1.0 at column 0
    let result = padded.sparse_row_min_value_and_column(0);
    assert_eq!(result, Some((1.0, 0)));
}

// ============================================================================
// sparse_row_max_values / sparse_row_min_values iterators
// ============================================================================

#[test]
fn test_sparse_row_max_values_iterator() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    let maxes: Vec<Option<f64>> = padded.sparse_row_max_values().collect();
    assert_eq!(maxes, vec![Some(5.0), Some(3.0)]);
}

#[test]
fn test_sparse_row_min_values_iterator() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();

    let mins: Vec<Option<f64>> = padded.sparse_row_min_values().collect();
    // Row 0: values 1.0, 5.0 → min 1.0
    // Row 1: values 3.0, 0.0(imputed) → min 0.0
    assert_eq!(mins, vec![Some(1.0), Some(0.0)]);
}
