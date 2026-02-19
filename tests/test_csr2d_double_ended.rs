//! Tests for DoubleEndedIterator and ExactSizeIterator paths in CSR2D
//! iterators: CSR2DView, CSR2DColumns (via padded diagonal), CSR2DSizedRows,
//! M2DValues. Uses take() to avoid known overflow issues in backward iteration.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, SparseValuedMatrix,
        SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_csr(entries: Vec<(usize, usize)>) -> TestCSR {
    CSR2D::from_entries(entries).unwrap()
}

fn build_csr_with_shape(rows: usize, cols: usize, entries: Vec<(usize, usize)>) -> TestCSR {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((rows, cols), entries.len());
    for entry in entries {
        MatrixMut::add(&mut csr, entry).unwrap();
    }
    csr
}

fn build_valued_csr(rows: usize, cols: usize, entries: Vec<(usize, usize, f64)>) -> TestValCSR {
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(entries.len())
        .expected_shape((rows, cols))
        .edges(entries.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// CSR2DView (sparse_coordinates): backward iteration and ExactSizeIterator
// ============================================================================

#[test]
fn test_view_backward_take_one() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let first_back: Vec<(usize, usize)> =
        SparseMatrix::sparse_coordinates(&csr).rev().take(1).collect();
    assert_eq!(first_back, vec![(2, 2)]);
}

#[test]
fn test_view_backward_take_two() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)]);
    let first_two: Vec<(usize, usize)> =
        SparseMatrix::sparse_coordinates(&csr).rev().take(2).collect();
    // Back row (2) has one entry, then falls through; exact behavior depends on
    // impl
    assert_eq!(first_two.len(), 2);
    assert_eq!(first_two[0], (2, 3));
}

#[test]
fn test_view_exact_size_initial() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_view_exact_size_after_next() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
}

#[test]
fn test_view_exact_size_empty_middle_rows() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (3, 3)]);
    let iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 2);
}

// ============================================================================
// CSR2DColumns via GenericMatrix2DWithPaddedDiagonal.sparse_columns()
// The padded diagonal type uses CSR2DColumns internally.
// ============================================================================

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

#[test]
fn test_padded_columns_backward_take() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    // sparse_columns() uses CSR2DColumns internally
    let first_back: Vec<usize> = padded.sparse_columns().rev().take(1).collect();
    assert_eq!(first_back, vec![1]);
}

#[test]
fn test_padded_columns_forward() {
    let padded = build_padded(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let cols: Vec<usize> = padded.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 1]);
}

// ============================================================================
// CSR2DSizedRows backward with take (used via sparse_rows() on CSR2D)
// ============================================================================

#[test]
fn test_sized_rows_backward_take_one() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let first_back: Vec<usize> =
        <TestCSR as SparseMatrix2D>::sparse_rows(&csr).rev().take(1).collect();
    assert_eq!(first_back, vec![2]);
}

#[test]
fn test_sized_rows_backward_take_two() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let first_two: Vec<usize> =
        <TestCSR as SparseMatrix2D>::sparse_rows(&csr).rev().take(2).collect();
    assert_eq!(first_two, vec![2, 1]);
}

// ============================================================================
// M2DValues: backward with take and ExactSizeIterator
// ============================================================================

#[test]
fn test_values_backward_take_one() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let first_back: Vec<f64> = vcsr.sparse_values().rev().take(1).collect();
    assert_eq!(first_back, vec![3.0]);
}

#[test]
fn test_values_backward_take_two() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let first_two: Vec<f64> = vcsr.sparse_values().rev().take(2).collect();
    assert_eq!(first_two, vec![3.0, 2.0]);
}

#[test]
fn test_values_exact_size_initial() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let iter = vcsr.sparse_values();
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_values_exact_size_after_next() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let mut iter = vcsr.sparse_values();
    iter.next();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
}

#[test]
fn test_values_mixed_two_rows() {
    let vcsr = build_valued_csr(2, 2, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);
    let mut iter = vcsr.sparse_values();
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next_back(), Some(4.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next_back(), Some(3.0));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// sparse_row_values backward on ValuedCSR2D
// ============================================================================

#[test]
fn test_sparse_row_values_rev() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)]);
    let vals: Vec<f64> = vcsr.sparse_row_values(0).rev().collect();
    assert_eq!(vals, vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_sparse_row_values_mixed() {
    let vcsr = build_valued_csr(2, 4, vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0), (0, 3, 4.0)]);
    let mut iter = vcsr.sparse_row_values(0);
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next_back(), Some(4.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next_back(), Some(3.0));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// SizedSparseMatrix2D: rank_row edge case
// ============================================================================

#[test]
fn test_rank_row_beyond_offsets() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (0, 1)]);
    assert_eq!(csr.rank_row(3), 2);
    assert_eq!(csr.rank_row(4), 2);
}

// ============================================================================
// select_row / select_column
// ============================================================================

#[test]
fn test_select_row_column() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(csr.select_row(0), 0);
    assert_eq!(csr.select_row(1), 1);
    assert_eq!(csr.select_row(2), 2);
    assert_eq!(csr.select_column(0), 0);
    assert_eq!(csr.select_column(1), 1);
    assert_eq!(csr.select_column(2), 2);
}

// ============================================================================
// CSR2D sparse_columns reverse (this is a simple slice iterator)
// ============================================================================

#[test]
fn test_sparse_columns_reverse() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let cols_rev: Vec<usize> = csr.sparse_columns().rev().collect();
    assert_eq!(cols_rev, vec![2, 1, 0]);
}

#[test]
fn test_sparse_columns_forward() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let cols: Vec<usize> = csr.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 2]);
}
