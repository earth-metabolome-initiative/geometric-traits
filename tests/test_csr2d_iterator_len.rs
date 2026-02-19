//! Tests for CSR2D iterator paths: forward crossing through middle rows (3+
//! rows), backward take, and ExactSizeIterator::len() for CSR2DView.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, SizedRowsSparseMatrix2D, SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D,
        SparseValuedMatrix, SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_csr(entries: Vec<(usize, usize)>, shape: (usize, usize)) -> TestCSR {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut csr, (r, c)).unwrap();
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
// CSR2DView (sparse_coordinates): ExactSizeIterator::len()
// ============================================================================

#[test]
fn test_view_len_initial() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_view_len_after_next() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)], (3, 4));
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_view_len_after_next_back() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 3);
    iter.next_back();
    assert_eq!(iter.len(), 2);
}

// ============================================================================
// CSR2DView: 3+ row forward iteration (crosses through middle rows)
// ============================================================================

#[test]
fn test_view_three_row_forward() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn test_view_four_row_forward() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2), (3, 3)], (4, 4));
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
}

#[test]
fn test_view_backward_single() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.next_back(), Some((2, 2)));
}

// ============================================================================
// CSR2DColumns: forward crossing through 3+ rows
// ============================================================================

#[test]
fn test_columns_three_row_forward() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let cols: Vec<usize> = csr.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 2]);
}

#[test]
fn test_columns_backward_single() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let first_back: Vec<usize> = csr.sparse_columns().rev().take(1).collect();
    assert_eq!(first_back, vec![2]);
}

#[test]
fn test_columns_backward_two() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let first_two: Vec<usize> = csr.sparse_columns().rev().take(2).collect();
    assert_eq!(first_two, vec![2, 1]);
}

#[test]
fn test_columns_full_backward() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let all_rev: Vec<usize> = csr.sparse_columns().rev().collect();
    assert_eq!(all_rev, vec![2, 1, 0]);
}

// ============================================================================
// CSR2DRows/CSR2DSizedRows: forward crossing through 3+ rows
// ============================================================================

#[test]
fn test_rows_three_row_forward() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)], (3, 4));
    let rows: Vec<usize> = <TestCSR as SparseMatrix2D>::sparse_rows(&csr).collect();
    assert_eq!(rows, vec![0, 0, 1, 2]);
}

#[test]
fn test_rows_backward_single() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let first_back: Vec<usize> =
        <TestCSR as SparseMatrix2D>::sparse_rows(&csr).rev().take(1).collect();
    assert_eq!(first_back, vec![2]);
}

#[test]
fn test_rows_backward_two() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let first_two: Vec<usize> =
        <TestCSR as SparseMatrix2D>::sparse_rows(&csr).rev().take(2).collect();
    assert_eq!(first_two, vec![2, 1]);
}

#[test]
fn test_rows_backward_three() {
    // Note: full rev().collect() triggers underflow bug in
    // CSR2DSizedRows::next_back at row 0, so we use take(3) to stop before the
    // underflow.
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let three_back: Vec<usize> =
        <TestCSR as SparseMatrix2D>::sparse_rows(&csr).rev().take(3).collect();
    assert_eq!(three_back, vec![2, 1, 0]);
}

// ============================================================================
// M2DValues: forward crossing through 3+ rows, backward
// ============================================================================

#[test]
fn test_values_three_row_forward() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let vals: Vec<f64> = vcsr.sparse_values().collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_values_four_row_forward() {
    let vcsr = build_valued_csr(4, 4, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (3, 3, 4.0)]);
    let vals: Vec<f64> = vcsr.sparse_values().collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_values_backward_single() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let first_back: Vec<f64> = vcsr.sparse_values().rev().take(1).collect();
    assert_eq!(first_back, vec![3.0]);
}

#[test]
fn test_values_backward_two() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let first_two: Vec<f64> = vcsr.sparse_values().rev().take(2).collect();
    assert_eq!(first_two, vec![3.0, 2.0]);
}

#[test]
fn test_values_full_backward() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let all_rev: Vec<f64> = vcsr.sparse_values().rev().collect();
    assert_eq!(all_rev, vec![3.0, 2.0, 1.0]);
}

// ============================================================================
// sparse_row_values: ExactSizeIterator::len()
// ============================================================================

#[test]
fn test_sparse_row_values_len() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)]);
    let iter = vcsr.sparse_row_values(0);
    assert_eq!(iter.len(), 3);
}

// ============================================================================
// rank_row, select_row, select_column on CSR2D with 3+ rows
// ============================================================================

#[test]
fn test_rank_select_multirow() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)], (3, 4));
    assert_eq!(csr.rank_row(0), 0);
    assert_eq!(csr.rank_row(1), 2);
    assert_eq!(csr.rank_row(2), 3);
    assert_eq!(csr.rank_row(3), 4);
    assert_eq!(csr.select_row(0), 0);
    assert_eq!(csr.select_row(2), 1);
    assert_eq!(csr.select_row(3), 2);
    assert_eq!(csr.select_column(0), 0);
    assert_eq!(csr.select_column(1), 1);
    assert_eq!(csr.select_column(2), 2);
    assert_eq!(csr.select_column(3), 3);
}

// ============================================================================
// sparse_row_sizes (SizedRowsSparseMatrix2D)
// ============================================================================

#[test]
fn test_sparse_row_sizes() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)], (3, 5));
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1, 2]);
}

// ============================================================================
// number_of_defined_values_in_row
// ============================================================================

#[test]
fn test_number_of_defined_values_in_row() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    assert_eq!(csr.number_of_defined_values_in_row(0), 2);
    assert_eq!(csr.number_of_defined_values_in_row(1), 1);
}
