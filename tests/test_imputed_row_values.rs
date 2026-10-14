//! Tests for ImputedRowValues: DoubleEndedIterator, ExactSizeIterator, Clone.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{DenseValuedMatrix2D, EdgesBuilder, SparseValuedMatrix2D},
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
// DoubleEndedIterator for row_values / sparse_row_values
// ============================================================================

#[test]
fn test_row_values_rev() {
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0)], 1, 2);
    // Padded to 2x2. Row 0: [1.0, 2.0]
    let reversed: Vec<f64> = padded.row_values(0).rev().collect();
    assert_eq!(reversed, vec![2.0, 1.0]);
}

#[test]
fn test_row_values_rev_imputed_row() {
    let padded = build_padded(vec![(0, 0, 1.0)], 1, 2);
    // Padded to 2x2. Row 1: all imputed → [0.0, 0.0]
    let reversed: Vec<f64> = padded.row_values(1).rev().collect();
    assert_eq!(reversed, vec![0.0, 0.0]);
}

#[test]
fn test_sparse_row_values_rev() {
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 5.0)], 1, 2);
    let reversed: Vec<f64> = padded.sparse_row_values(0).rev().collect();
    assert_eq!(reversed, vec![5.0, 1.0]);
}

#[test]
fn test_sparse_row_values_rev_mixed() {
    // 1x3 padded to 3x3, row 0 has sparse at (0,0)=1.0 and (0,2)=3.0, column 1
    // imputed
    let padded = build_padded(vec![(0, 0, 1.0), (0, 2, 3.0)], 1, 3);
    let reversed: Vec<f64> = padded.sparse_row_values(0).rev().collect();
    // Row 0 in padded 3x3: columns [0, 1, 2] → [1.0, 0.0, 3.0] reversed → [3.0,
    // 0.0, 1.0]
    assert_eq!(reversed, vec![3.0, 0.0, 1.0]);
}

// ============================================================================
// ExactSizeIterator
// ============================================================================

#[test]
fn test_row_values_exact_size() {
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0)], 1, 2);
    // Padded to 2x2, so row 0 has 2 values
    let iter = padded.row_values(0);
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_row_values_exact_size_larger() {
    let padded = build_padded(vec![(0, 0, 1.0)], 2, 3);
    // Padded to 3x3, each row has 3 values
    let iter = padded.row_values(0);
    assert_eq!(iter.len(), 3);
}

// ============================================================================
// Clone
// ============================================================================

#[test]
fn test_row_values_clone() {
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0)], 1, 2);
    let iter = padded.row_values(0);
    let cloned = iter.clone();
    let vals1: Vec<f64> = iter.collect();
    let vals2: Vec<f64> = cloned.collect();
    assert_eq!(vals1, vals2);
}

// ============================================================================
// Large padded matrix edge cases
// ============================================================================

#[test]
fn test_row_values_padded_beyond_original() {
    // 1x4 matrix padded to 4x4
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0), (0, 3, 4.0)], 1, 4);
    // Row 3 (beyond original) - all imputed
    let vals: Vec<f64> = padded.row_values(3).collect();
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 0.0]);
}
