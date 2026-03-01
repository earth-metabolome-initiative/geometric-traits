//! Tests for CSR2D iterator types: CSR2DSizedRows, CSR2DSizedRowsizes,
//! CSR2DEmptyRowIndices, CSR2DNonEmptyRowIndices, M2DValues, CSR2DView.
//! Exercises Iterator, DoubleEndedIterator and ExactSizeIterator
//! implementations.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, MatrixMut, SizedRowsSparseMatrix2D, SparseMatrix, SparseMatrix2D,
        SparseMatrixMut, SparseValuedMatrix, SparseValuedMatrix2D,
    },
};

type TestCSR2D = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_csr(entries: Vec<(usize, usize)>) -> TestCSR2D {
    CSR2D::from_entries(entries).unwrap()
}

fn build_valued_csr(rows: usize, cols: usize, entries: Vec<(usize, usize, f64)>) -> TestValCSR {
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(entries.len())
        .expected_shape((rows, cols))
        .edges(entries.into_iter())
        .build()
        .unwrap()
}

fn build_csr_with_shape(rows: usize, cols: usize, entries: Vec<(usize, usize)>) -> TestCSR2D {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((rows, cols), entries.len());
    for entry in entries {
        MatrixMut::add(&mut csr, entry).unwrap();
    }
    csr
}

// ============================================================================
// CSR2DSizedRows: Iterator, DoubleEndedIterator, ExactSizeIterator
// (accessed via CSR2D::sparse_rows())
// ============================================================================

#[test]
fn test_sized_rows_forward() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let rows: Vec<usize> = csr.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

#[test]
fn test_sized_rows_forward_three_rows() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let rows: Vec<usize> = csr.sparse_rows().collect();
    assert_eq!(rows, vec![0, 1, 2]);
}

#[test]
fn test_sized_rows_forward_multi_entry() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]);
    let rows: Vec<usize> = csr.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1, 1, 2]);
}

#[test]
fn test_sized_rows_rev_take() {
    // Use take to avoid overflow in DoubleEndedIterator
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let first_back: Vec<usize> = csr.sparse_rows().rev().take(2).collect();
    // Back row (2) yields 1 entry, then middle row (1) yields 1 entry
    assert_eq!(first_back, vec![2, 1]);
}

// ============================================================================
// CSR2DSizedRowsizes: Iterator, ExactSizeIterator
// (accessed via CSR2D::sparse_row_sizes())
// ============================================================================

#[test]
fn test_row_sizes_forward() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1]);
}

#[test]
fn test_row_sizes_exact_size() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let iter = csr.sparse_row_sizes();
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_row_sizes_with_empty_rows() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (0, 1), (3, 0)]);
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 0, 0, 1]);
}

#[test]
fn test_row_sizes_rev_take() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (0, 1), (3, 0)]);
    let first_back: Vec<usize> = csr.sparse_row_sizes().rev().take(2).collect();
    // Row sizes from back: row 3 has 1, then row 2 has 0
    assert_eq!(first_back, vec![1, 0]);
}

// ============================================================================
// CSR2DEmptyRowIndices: Iterator, ExactSizeIterator
// (accessed via CSR2D::empty_row_indices())
// ============================================================================

#[test]
fn test_empty_row_indices_forward() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (2, 1)]);
    let empty: Vec<usize> = csr.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 3]);
}

#[test]
fn test_empty_row_indices_exact_size() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (2, 1)]);
    let iter = csr.empty_row_indices();
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_empty_row_indices_rev_take() {
    let csr = build_csr_with_shape(5, 5, vec![(0, 0), (2, 1)]);
    let first_back: Vec<usize> = csr.empty_row_indices().rev().take(2).collect();
    assert_eq!(first_back, vec![4, 3]);
}

#[test]
fn test_empty_row_indices_none() {
    let csr = build_csr(vec![(0, 0), (1, 1)]);
    let empty: Vec<usize> = csr.empty_row_indices().collect();
    assert!(empty.is_empty());
}

// ============================================================================
// CSR2DNonEmptyRowIndices: Iterator, ExactSizeIterator
// (accessed via CSR2D::non_empty_row_indices())
// ============================================================================

#[test]
fn test_non_empty_row_indices_forward() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (2, 1)]);
    let non_empty: Vec<usize> = csr.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 2]);
}

#[test]
fn test_non_empty_row_indices_exact_size() {
    let csr = build_csr_with_shape(4, 4, vec![(0, 0), (2, 1)]);
    let iter = csr.non_empty_row_indices();
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_non_empty_row_indices_rev_take() {
    let csr = build_csr_with_shape(5, 5, vec![(0, 0), (2, 1), (4, 3)]);
    let first_back: Vec<usize> = csr.non_empty_row_indices().rev().take(2).collect();
    assert_eq!(first_back, vec![4, 2]);
}

#[test]
fn test_non_empty_row_indices_all() {
    let csr = build_csr(vec![(0, 0), (1, 1)]);
    let non_empty: Vec<usize> = csr.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 1]);
}

// ============================================================================
// M2DValues: Iterator, DoubleEndedIterator, ExactSizeIterator
// (accessed via ValuedCSR2D::sparse_values())
// ============================================================================

#[test]
fn test_m2d_values_forward() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)]);
    let vals: Vec<f64> = vcsr.sparse_values().collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_m2d_values_forward_multi_row() {
    let vcsr = build_valued_csr(
        3,
        3,
        vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 2, 4.0), (2, 1, 5.0)],
    );
    let vals: Vec<f64> = vcsr.sparse_values().collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_m2d_values_rev_take() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    let vals_rev: Vec<f64> = vcsr.sparse_values().rev().take(2).collect();
    // Back row (2): 3.0, then middle row (1): 2.0
    assert_eq!(vals_rev, vec![3.0, 2.0]);
}

// ============================================================================
// CSR2DView: Iterator, DoubleEndedIterator, ExactSizeIterator
// (accessed via CSR2D::sparse_coordinates())
// ============================================================================

#[test]
fn test_csr2d_view_forward() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)]);
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 1), (0, 2), (1, 0)]);
}

#[test]
fn test_csr2d_view_exact_size() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)]);
    let iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_csr2d_view_rev_take() {
    let csr = build_csr(vec![(0, 1), (1, 2)]);
    let first_back: Vec<(usize, usize)> =
        SparseMatrix::sparse_coordinates(&csr).rev().take(1).collect();
    assert_eq!(first_back, vec![(1, 2)]);
}

#[test]
fn test_csr2d_view_exact_size_after_partial() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0), (1, 1)]);
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    iter.next();
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_csr2d_view_three_rows() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
    assert_eq!(coords, vec![(0, 0), (1, 1), (2, 2)]);
}

// ============================================================================
// Additional: sparse_row_values on ValuedCSR2D
// ============================================================================

#[test]
fn test_valued_csr2d_sparse_row_values() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)]);
    let row0_vals: Vec<f64> = vcsr.sparse_row_values(0).collect();
    assert_eq!(row0_vals, vec![1.0, 2.0]);
    let row1_vals: Vec<f64> = vcsr.sparse_row_values(1).collect();
    assert_eq!(row1_vals, vec![3.0]);
}

#[test]
fn test_valued_csr2d_sparse_row_max_min() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0), (1, 0, 2.0)]);
    assert_eq!(vcsr.sparse_row_max_value(0), Some(5.0));
    assert_eq!(vcsr.sparse_row_min_value(0), Some(1.0));
    assert_eq!(vcsr.sparse_row_max_value(1), Some(2.0));
    assert_eq!(vcsr.sparse_row_min_value(1), Some(2.0));
}

// ============================================================================
// Additional coverage: CSR2DColumns forward multi-row, backward, len
// ============================================================================

fn make_3x4() -> TestCSR2D {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((3, 4), 6);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (0, 2)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    MatrixMut::add(&mut csr, (2, 3)).unwrap();
    csr
}

#[test]
fn test_sparse_columns_forward_multi_row() {
    let csr = make_3x4();
    let cols: Vec<usize> = csr.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 2, 1, 0, 3]);
}

#[test]
fn test_sparse_columns_backward_multi_row() {
    let csr = make_3x4();
    let cols: Vec<usize> = csr.sparse_columns().rev().collect();
    assert_eq!(cols, vec![3, 0, 1, 2, 1, 0]);
}

#[test]
fn test_sparse_columns_len_during_iteration() {
    let csr = make_3x4();
    let mut iter = csr.sparse_columns();
    assert_eq!(iter.len(), 6);
    iter.next();
    assert_eq!(iter.len(), 5);
    iter.next();
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
    iter.next();
    assert_eq!(iter.len(), 0);
}

#[test]
fn test_sparse_columns_mixed_direction() {
    let csr = make_3x4();
    let mut iter = csr.sparse_columns();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(3));
    assert_eq!(iter.next_back(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(1)); // falls through to back
    assert_eq!(iter.next(), None);
}

// ============================================================================
// CSR2DRows: forward multi-row, backward, len
// ============================================================================

#[test]
fn test_sparse_rows_forward_multi_row() {
    let csr = make_3x4();
    let rows: Vec<usize> = <TestCSR2D as SparseMatrix2D>::sparse_rows(&csr).collect();
    assert_eq!(rows, vec![0, 0, 0, 1, 2, 2]);
}

#[test]
fn test_sparse_rows_backward_multi_row() {
    let csr = make_3x4();
    let rows: Vec<usize> = <TestCSR2D as SparseMatrix2D>::sparse_rows(&csr).rev().collect();
    assert_eq!(rows, vec![2, 2, 1, 0, 0, 0]);
}

#[test]
fn test_sparse_rows_len_during_iteration() {
    let csr = make_3x4();
    let mut iter = <TestCSR2D as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.len(), 6);
    iter.next();
    assert_eq!(iter.len(), 5);
    iter.next_back();
    assert_eq!(iter.len(), 4);
}

#[test]
fn test_sparse_rows_mixed_direction() {
    let csr = make_3x4();
    let mut iter = <TestCSR2D as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(2));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(2));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// CSR2DView: double-ended coordinates iteration
// ============================================================================

#[test]
fn test_view_backward_multi_row() {
    let csr = make_3x4();
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).rev().collect();
    assert_eq!(coords, vec![(2, 3), (2, 0), (1, 1), (0, 2), (0, 1), (0, 0)]);
}

#[test]
fn test_view_len_mid_iteration() {
    let csr = make_3x4();
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.len(), 6);
    iter.next();
    assert_eq!(iter.len(), 5);
    iter.next_back();
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
}

#[test]
fn test_view_mixed_direction() {
    let csr = make_3x4();
    let mut iter = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(iter.next(), Some((0, 0)));
    assert_eq!(iter.next_back(), Some((2, 3)));
    assert_eq!(iter.next(), Some((0, 1)));
    assert_eq!(iter.next_back(), Some((2, 0)));
    assert_eq!(iter.next(), Some((0, 2)));
    assert_eq!(iter.next_back(), Some((1, 1)));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// M2DValues: multi-row forward, backward, len
// ============================================================================

#[test]
fn test_values_forward_3x3() {
    let m: TestValCSR =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    let vals: Vec<f64> = m.sparse_values().collect();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_values_backward_3x3() {
    let m: TestValCSR =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    let vals: Vec<f64> = m.sparse_values().rev().collect();
    assert_eq!(vals, vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_values_len_during_iteration() {
    let m: TestValCSR = ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
    let mut iter = m.sparse_values();
    assert_eq!(iter.len(), 6);
    iter.next();
    assert_eq!(iter.len(), 5);
    iter.next_back();
    assert_eq!(iter.len(), 4);
}

#[test]
fn test_values_mixed_direction() {
    let m: TestValCSR = ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
    let mut iter = m.sparse_values();
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next_back(), Some(6.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next_back(), Some(5.0));
    assert_eq!(iter.next(), Some(3.0));
    assert_eq!(iter.next_back(), Some(4.0));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// Two-row matrix: back-fallthrough paths
// ============================================================================

#[test]
fn test_two_row_columns_falls_to_back() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((2, 3), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();

    let mut iter = csr.sparse_columns();
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2)); // falls to back
    assert_eq!(iter.next(), None);
}

// ============================================================================
// Empty matrix safety across bidirectional CSR iterators
// ============================================================================

#[test]
fn test_empty_csr_iterators_are_safe_and_empty() {
    let csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((0, 0), 0);

    let mut sparse_rows = csr.sparse_rows();
    assert_eq!(sparse_rows.len(), 0);
    assert_eq!(sparse_rows.next(), None);
    assert_eq!(sparse_rows.next_back(), None);

    let mut sparse_columns = csr.sparse_columns();
    assert_eq!(sparse_columns.len(), 0);
    assert_eq!(sparse_columns.next(), None);
    assert_eq!(sparse_columns.next_back(), None);

    let mut sparse_row_sizes = csr.sparse_row_sizes();
    assert_eq!(sparse_row_sizes.len(), 0);
    assert_eq!(sparse_row_sizes.next(), None);
    assert_eq!(sparse_row_sizes.next_back(), None);

    let mut sparse_coords = SparseMatrix::sparse_coordinates(&csr);
    assert_eq!(sparse_coords.len(), 0);
    assert_eq!(sparse_coords.next(), None);
    assert_eq!(sparse_coords.next_back(), None);
}

#[test]
fn test_empty_valued_csr_sparse_values_are_safe_and_empty() {
    let vcsr: TestValCSR = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);

    let mut sparse_values = vcsr.sparse_values();
    assert_eq!(sparse_values.len(), 0);
    assert_eq!(sparse_values.next(), None);
    assert_eq!(sparse_values.next_back(), None);
}
