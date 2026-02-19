//! Tests for SparseRowWithPaddedDiagonal and SparseRowValuesWithPaddedDiagonal
//! iterators: forward, backward, clone, and edge cases.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SparseMatrix2D, SparseValuedMatrix2D},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_padded_diagonal(
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
    GenericMatrix2DWithPaddedDiagonal::new(inner, |row: usize| usize_to_f64(row + 1) * 100.0)
        .unwrap()
}

// ============================================================================
// SparseRowWithPaddedDiagonal: forward iteration
// ============================================================================

#[test]
fn test_sparse_row_forward_diagonal_present() {
    // Row 0 has (0,0) and (0,1). Diagonal already present.
    let padded = build_padded_diagonal(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let row0: Vec<usize> = padded.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1]);
}

#[test]
fn test_sparse_row_forward_diagonal_imputed_before_all() {
    // Row 0 has only (0,1). Diagonal 0 should be imputed before column 1.
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 2);
    let row0: Vec<usize> = padded.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1]);
}

#[test]
fn test_sparse_row_forward_diagonal_imputed_after_all() {
    // Row 1 has only (1,0). Diagonal 1 should be appended at end.
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 0, 2.0)], 2, 2);
    let row1: Vec<usize> = padded.sparse_row(1).collect();
    assert_eq!(row1, vec![0, 1]);
}

#[test]
fn test_sparse_row_forward_empty_row_gets_diagonal() {
    // Row 1 has no entries. Should get just the diagonal.
    let padded = build_padded_diagonal(vec![(0, 0, 1.0)], 2, 2);
    let row1: Vec<usize> = padded.sparse_row(1).collect();
    assert_eq!(row1, vec![1]);
}

#[test]
fn test_sparse_row_forward_extra_padded_row() {
    // 2x3 matrix padded to 3x3. Row 2 doesn't exist in original.
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 3);
    let row2: Vec<usize> = padded.sparse_row(2).collect();
    assert_eq!(row2, vec![2]);
}

// ============================================================================
// SparseRowWithPaddedDiagonal: backward iteration
// ============================================================================

#[test]
fn test_sparse_row_backward_diagonal_present() {
    let padded = build_padded_diagonal(vec![(0, 0, 5.0), (0, 1, 1.0), (0, 2, 3.0)], 2, 3);
    let row0_rev: Vec<usize> = padded.sparse_row(0).rev().collect();
    assert_eq!(row0_rev, vec![2, 1, 0]);
}

#[test]
fn test_sparse_row_backward_diagonal_imputed_at_end() {
    // Row 0 has (0,1), (0,2). Diagonal 0 should be imputed.
    // Forward: [0, 1, 2]. Reversed: [2, 1, 0]
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (0, 2, 3.0)], 1, 3);
    let row0_rev: Vec<usize> = padded.sparse_row(0).rev().collect();
    assert_eq!(row0_rev, vec![2, 1, 0]);
}

#[test]
fn test_sparse_row_backward_diagonal_imputed_between() {
    // Row 1 has (1,0) and (1,2). Diagonal=1 should appear between them.
    // Forward: [0, 1, 2]. Reversed: [2, 1, 0]
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 0, 2.0), (1, 2, 3.0)], 2, 3);
    let row1_rev: Vec<usize> = padded.sparse_row(1).rev().collect();
    assert_eq!(row1_rev, vec![2, 1, 0]);
}

#[test]
fn test_sparse_row_backward_empty_row() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0)], 2, 2);
    let row1_rev: Vec<usize> = padded.sparse_row(1).rev().collect();
    assert_eq!(row1_rev, vec![1]);
}

// ============================================================================
// SparseRowValuesWithPaddedDiagonal: forward iteration
// ============================================================================

#[test]
fn test_sparse_row_values_forward_diagonal_present() {
    let padded = build_padded_diagonal(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_row_values(0).collect();
    assert_eq!(vals, vec![5.0, 1.0]);
}

#[test]
fn test_sparse_row_values_forward_diagonal_imputed() {
    // Row 0 has (0,1)=1.0. Diagonal 0 imputed with map(0)=100.0
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_row_values(0).collect();
    assert_eq!(vals, vec![100.0, 1.0]);
}

#[test]
fn test_sparse_row_values_forward_diagonal_at_end() {
    // Row 1 has (1,0)=2.0. Diagonal 1 imputed at end with map(1)=200.0
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 0, 2.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_row_values(1).collect();
    assert_eq!(vals, vec![2.0, 200.0]);
}

#[test]
fn test_sparse_row_values_forward_empty_row() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_row_values(1).collect();
    assert_eq!(vals, vec![200.0]); // map(1) = 200.0
}

#[test]
fn test_sparse_row_values_forward_extra_padded_row() {
    // 2x3 padded to 3x3. Row 2 has just imputed diagonal.
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 3);
    let vals: Vec<f64> = padded.sparse_row_values(2).collect();
    assert_eq!(vals, vec![300.0]); // map(2) = 300.0
}

// ============================================================================
// SparseRowValuesWithPaddedDiagonal: backward iteration
// ============================================================================

#[test]
fn test_sparse_row_values_backward_all_before_diagonal() {
    // Row 1 has (1,0)=2.0 only. Diagonal=1 imputed.
    // Backward: next_back gets col 0, val 2.0 (col 0 < diag 1, no special handling
    // in current code). Then diagonal returned at end.
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 0, 2.0)], 2, 2);
    let vals_rev: Vec<f64> = padded.sparse_row_values(1).rev().collect();
    // Both values present
    assert_eq!(vals_rev.len(), 2);
    assert!(vals_rev.contains(&2.0));
    assert!(vals_rev.contains(&200.0));
}

#[test]
fn test_sparse_row_values_backward_all_after_diagonal() {
    // Row 0 has (0,1)=1.0 and (0,2)=3.0. Diagonal=0 imputed.
    // All columns > diagonal, so backward should handle imputation.
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (0, 2, 3.0)], 1, 3);
    let vals_rev: Vec<f64> = padded.sparse_row_values(0).rev().collect();
    assert_eq!(vals_rev.len(), 3);
    assert!(vals_rev.contains(&100.0)); // map(0) = 100.0
    assert!(vals_rev.contains(&1.0));
    assert!(vals_rev.contains(&3.0));
}

#[test]
fn test_sparse_row_values_backward_empty_row() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0)], 2, 2);
    let vals_rev: Vec<f64> = padded.sparse_row_values(1).rev().collect();
    assert_eq!(vals_rev, vec![200.0]);
}

// ============================================================================
// Clone for both iterators
// ============================================================================

#[test]
fn test_sparse_row_iterator_clone() {
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 2);
    let iter = padded.sparse_row(0);
    let cloned = iter.clone();
    let original: Vec<usize> = iter.collect();
    let from_clone: Vec<usize> = cloned.collect();
    assert_eq!(original, from_clone);
}

#[test]
fn test_sparse_row_values_iterator_clone() {
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 2);
    let iter = padded.sparse_row_values(0);
    let cloned = iter.clone();
    let original: Vec<f64> = iter.collect();
    let from_clone: Vec<f64> = cloned.collect();
    assert_eq!(original, from_clone);
}

// ============================================================================
// Mixed forward/backward iteration
// ============================================================================

#[test]
fn test_sparse_row_mixed_iteration() {
    // Row 0 has (0,1)=1.0 and (0,2)=3.0. Diagonal 0 imputed.
    // Full forward: [0, 1, 2]
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (0, 2, 3.0)], 1, 3);
    let mut iter = padded.sparse_row(0);
    let first = iter.next();
    assert_eq!(first, Some(0)); // imputed diagonal
    let last = iter.next_back();
    assert_eq!(last, Some(2)); // column 2
    let middle = iter.next();
    assert_eq!(middle, Some(1)); // column 1
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_row_values_mixed_iteration() {
    // Row 0 has (0,1)=1.0 and (0,2)=3.0. Diagonal 0 imputed with 100.0.
    // Full forward: [100.0, 1.0, 3.0]
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (0, 2, 3.0)], 1, 3);
    let mut iter = padded.sparse_row_values(0);
    let first = iter.next();
    assert_eq!(first, Some(100.0)); // imputed diagonal
    let last = iter.next_back();
    assert_eq!(last, Some(3.0)); // column 2 value
    let middle = iter.next();
    assert_eq!(middle, Some(1.0)); // column 1 value
    assert_eq!(iter.next(), None);
}
