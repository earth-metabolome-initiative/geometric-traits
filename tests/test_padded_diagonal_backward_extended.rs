//! Tests for backward iteration and crossing paths in padded diagonal
//! iterators: SparseRowWithPaddedDiagonal, SparseRowValuesWithPaddedDiagonal,
//! SparseRowsWithPaddedDiagonal (forward-back crossing).
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SparseMatrix, SparseMatrix2D, SparseValuedMatrix2D},
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
// SparseRowWithPaddedDiagonal: backward iteration where diagonal is in the
// sparse data
// ============================================================================

#[test]
fn test_padded_row_backward_diagonal_present() {
    // Row 0 has (0,0) and (0,1). The diagonal (0,0) IS present in the sparse data.
    // backward iteration should yield: col 1, col 0 (with diagonal value from
    // sparse)
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let row_rev: Vec<usize> = padded.sparse_row(0).rev().collect();
    // Row 0 should have columns [0, 1] forward. With padded diagonal (0), the
    // backward is [1, 0].
    assert_eq!(row_rev, vec![1, 0]);
}

#[test]
fn test_padded_row_backward_diagonal_missing() {
    // Row 0 has only (0,1). Diagonal (0,0) is imputed.
    // Forward: [0, 1] (diagonal first, then sparse). Backward: [1, 0]
    let padded = build_padded(vec![(0, 1, 1.0)], 2, 2);
    let row_fwd: Vec<usize> = padded.sparse_row(0).collect();
    // Diagonal 0 is imputed, then column 1 from sparse
    assert_eq!(row_fwd, vec![0, 1]);
    let row_rev: Vec<usize> = padded.sparse_row(0).rev().collect();
    assert_eq!(row_rev, vec![1, 0]);
}

#[test]
fn test_padded_row_backward_diagonal_after_sparse() {
    // Row 0 has only (0,0). Row 1 has only (1,0). Diagonal for row 1 is (1,1) which
    // is missing. For row 1: sparse has col 0, diagonal is col 1 (imputed).
    // Forward: [0, 1]. Backward: [1, 0].
    let padded = build_padded(vec![(0, 0, 5.0), (1, 0, 3.0)], 2, 2);
    let row1_rev: Vec<usize> = padded.sparse_row(1).rev().collect();
    assert_eq!(row1_rev, vec![1, 0]);
}

// ============================================================================
// SparseRowValuesWithPaddedDiagonal: backward where diagonal==column in
// next_back
// ============================================================================

#[test]
fn test_padded_values_backward_no_higher_cols() {
    // Row 0: only (0,0)=5.0. Diagonal at col 0 is present. No columns above
    // diagonal. Backward should just return 5.0.
    let padded = build_padded(vec![(0, 0, 5.0)], 2, 2);
    let vals_rev: Vec<f64> = padded.sparse_row_values(0).rev().collect();
    assert_eq!(vals_rev, vec![5.0]);
}

#[test]
fn test_padded_values_backward_diagonal_imputed() {
    // Row 1: no sparse entries, diagonal (1,1) imputed = map(1) = 20.0
    // Padded to 2x2, so row 1 just has the diagonal element
    let padded = build_padded(vec![(0, 0, 5.0)], 2, 2);
    let vals: Vec<f64> = padded.sparse_row_values(1).collect();
    assert_eq!(vals, vec![20.0]);
    let vals_rev: Vec<f64> = padded.sparse_row_values(1).rev().collect();
    assert_eq!(vals_rev, vec![20.0]);
}

// ============================================================================
// SparseRowsWithPaddedDiagonal: mixed forward-back crossing
// ============================================================================

#[test]
fn test_padded_sparse_rows_forward_back_crossing() {
    // 3x3 padded diagonal matrix. sparse_rows iterates using
    // SparseRowsWithPaddedDiagonal. We interleave forward and backward to
    // trigger crossing paths.
    let padded = build_padded(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (2, 2, 4.0)], 3, 3);
    let mut iter = padded.sparse_rows();
    // Forward: consume some from front
    let first = iter.next();
    assert!(first.is_some());
    // Backward: consume some from back
    let last = iter.next_back();
    assert!(last.is_some());
    // Continue consuming forward
    while iter.next().is_some() {}
}

#[test]
fn test_padded_sparse_rows_full_backward() {
    let padded = build_padded(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)], 3, 3);
    // Full backward iteration
    let rows_rev: Vec<usize> = padded.sparse_rows().rev().collect();
    assert!(!rows_rev.is_empty());
    // First element from back should be row 2
    assert_eq!(rows_rev[0], 2);
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal: matrix() accessor and is_diagonal_imputed
// ============================================================================

#[test]
fn test_padded_diagonal_matrix_accessor() {
    let padded = build_padded(vec![(0, 0, 5.0)], 2, 2);
    let inner = padded.matrix();
    assert_eq!(inner.number_of_defined_values(), 1);
}

#[test]
fn test_padded_diagonal_is_imputed() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    // Row 0 has diagonal at (0,0) in sparse data => not imputed
    assert!(!padded.is_diagonal_imputed(0));
    // Row 1 has no entries => diagonal is imputed
    assert!(padded.is_diagonal_imputed(1));
    // Row beyond matrix bounds => imputed
    assert!(padded.is_diagonal_imputed(5));
}

#[test]
fn test_padded_diagonal_is_empty() {
    let padded = build_padded(vec![(0, 0, 5.0)], 2, 2);
    assert!(!SparseMatrix::is_empty(&padded));
}

#[test]
fn test_padded_diagonal_last_sparse_coordinates() {
    let padded = build_padded(vec![(0, 0, 5.0)], 2, 2);
    let last = SparseMatrix::last_sparse_coordinates(&padded);
    assert!(last.is_some());
    // Should be (1, 1) since the padded diagonal extends to 2x2
    assert_eq!(last, Some((1, 1)));
}

// ============================================================================
// Clone for SparseRowWithPaddedDiagonal
// ============================================================================

#[test]
fn test_padded_row_clone() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let iter = padded.sparse_row(0);
    let cloned = iter.clone();
    let original: Vec<usize> = iter.collect();
    let from_clone: Vec<usize> = cloned.collect();
    assert_eq!(original, from_clone);
}

// ============================================================================
// Clone for SparseRowValuesWithPaddedDiagonal
// ============================================================================

#[test]
fn test_padded_values_clone() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let iter = padded.sparse_row_values(0);
    let cloned = iter.clone();
    let original: Vec<f64> = iter.collect();
    let from_clone: Vec<f64> = cloned.collect();
    assert_eq!(original, from_clone);
}
