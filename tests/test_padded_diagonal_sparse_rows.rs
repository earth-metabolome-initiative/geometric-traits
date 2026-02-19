//! Tests for SparseRowsWithPaddedDiagonal iterator: Iterator and
//! DoubleEndedIterator. Exercises coverage of
//! sparse_rows_with_padded_diagonal.rs (0/32 coverage).
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, Matrix2D, SparseMatrix2D},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_padded_diagonal(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> GenericMatrix2DWithPaddedDiagonal<TestValCSR, impl Fn(usize) -> f64> {
    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap()
}

// ============================================================================
// sparse_rows() forward iteration (exercises
// SparseRowsWithPaddedDiagonal::next)
// ============================================================================

#[test]
fn test_sparse_rows_forward_all_diagonal_present() {
    // 2x2 with diagonal entries at (0,0) and (1,1)
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 1, 2.0)], 2, 2);
    let rows: Vec<usize> = padded.sparse_rows().collect();
    // Row 0: [0] (diagonal present), Row 1: [1] (diagonal present)
    assert_eq!(rows, vec![0, 1]);
}

#[test]
fn test_sparse_rows_forward_diagonal_imputed() {
    // 2x2 with only off-diagonal entries: (0,1) and (1,0)
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let rows: Vec<usize> = padded.sparse_rows().collect();
    // Row 0: [0(imputed), 1] → 2 entries → [0, 0]
    // Row 1: [0, 1(imputed)] → 2 entries → [1, 1]
    assert_eq!(rows, vec![0, 0, 1, 1]);
}

#[test]
fn test_sparse_rows_forward_mixed() {
    // 2x2: row 0 has diagonal (0,0) + off-diag (0,1); row 1 has no entries (diag
    // imputed)
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (0, 1, 2.0)], 2, 2);
    let rows: Vec<usize> = padded.sparse_rows().collect();
    // Row 0: [0, 1] → 2 entries → [0, 0]
    // Row 1: [1(imputed)] → 1 entry → [1]
    assert_eq!(rows, vec![0, 0, 1]);
}

#[test]
fn test_sparse_rows_forward_3x3_rectangular() {
    // 2x3 padded to 3x3, all diagonal imputed
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (1, 2, 2.0)], 2, 3);
    assert_eq!(padded.number_of_rows(), 3);
    let rows: Vec<usize> = padded.sparse_rows().collect();
    // Row 0: [0(imputed), 1] → [0, 0]
    // Row 1: [1(imputed), 2] → [1, 1]
    // Row 2: [2(imputed)] → [2]
    assert_eq!(rows, vec![0, 0, 1, 1, 2]);
}

// ============================================================================
// sparse_rows() reverse iteration (exercises
// SparseRowsWithPaddedDiagonal::next_back)
// ============================================================================

#[test]
fn test_sparse_rows_rev_all_diagonal_present() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 1, 2.0)], 2, 2);
    let rows_rev: Vec<usize> = padded.sparse_rows().rev().collect();
    // Reversed: Row 1: [1], Row 0: [0]
    assert_eq!(rows_rev, vec![1, 0]);
}

#[test]
fn test_sparse_rows_rev_diagonal_imputed() {
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let rows_rev: Vec<usize> = padded.sparse_rows().rev().collect();
    // Row 1 (back): [0, 1(imputed)] → [1, 1]
    // Row 0 (forward): [0(imputed), 1] → [0, 0]
    assert_eq!(rows_rev, vec![1, 1, 0, 0]);
}

#[test]
fn test_sparse_rows_rev_3x3() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (0, 1, 2.0)], 2, 2);
    let rows_rev: Vec<usize> = padded.sparse_rows().rev().collect();
    // Row 1 (back): [1(imputed)] → [1]
    // Row 0 (forward): [0, 1] → [0, 0]
    assert_eq!(rows_rev, vec![1, 0, 0]);
}

// ============================================================================
// Mixed forward/backward iteration (exercises both paths interleaving)
// ============================================================================

#[test]
fn test_sparse_rows_mixed_iteration() {
    let padded = build_padded_diagonal(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let mut iter = padded.sparse_rows();
    // Forward: first element from row 0
    let first = iter.next();
    assert_eq!(first, Some(0));
    // Backward: last element from row 1
    let last = iter.next_back();
    assert_eq!(last, Some(1));
}

// ============================================================================
// sparse_columns() — also exercises CSR2DColumns via padded diagonal
// ============================================================================

#[test]
fn test_sparse_columns_forward() {
    let padded = build_padded_diagonal(vec![(0, 0, 1.0), (1, 1, 2.0)], 2, 2);
    let cols: Vec<usize> = padded.sparse_columns().collect();
    // Row 0: [0], Row 1: [1]
    assert_eq!(cols, vec![0, 1]);
}

#[test]
fn test_sparse_columns_with_imputed() {
    let padded = build_padded_diagonal(vec![(0, 1, 1.0)], 2, 2);
    let cols: Vec<usize> = padded.sparse_columns().collect();
    // Row 0: [0(imputed), 1] → [0, 1]
    // Row 1: [1(imputed)] → [1]
    assert_eq!(cols, vec![0, 1, 1]);
}
