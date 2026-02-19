//! Tests for remaining uncovered delegations in SquareCSR2D and M2DValues.
//! Covers: SquareCSR2D EmptyRows, SizedRowsSparseMatrix2D,
//! MatrixMut::increase_shape, AsRef, and M2DValues (csr2d_values.rs).
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, MatrixMut, SizedRowsSparseMatrix2D, SparseMatrixMut,
        SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

// ============================================================================
// SquareCSR2D EmptyRows
// ============================================================================

#[test]
fn test_square_csr_empty_row_indices() {
    let mut sq: TestSquareCSR = SquareCSR2D::default();
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    // After adding (0,1), matrix becomes 2x2. Row 1 is empty.
    let empty: Vec<usize> = sq.empty_row_indices().collect();
    assert_eq!(empty, vec![1]);
}

#[test]
fn test_square_csr_non_empty_row_indices() {
    let mut sq: TestSquareCSR = SquareCSR2D::default();
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    let non_empty: Vec<usize> = sq.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
}

#[test]
fn test_square_csr_number_of_empty_rows() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    assert_eq!(sq.number_of_empty_rows(), 2);
    assert_eq!(sq.number_of_non_empty_rows(), 1);
}

// ============================================================================
// SquareCSR2D SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_square_csr_number_of_defined_values_in_row() {
    let mut sq: TestSquareCSR = SquareCSR2D::default();
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    MatrixMut::add(&mut sq, (1, 0)).unwrap();
    assert_eq!(sq.number_of_defined_values_in_row(0), 2);
    assert_eq!(sq.number_of_defined_values_in_row(1), 1);
}

#[test]
fn test_square_csr_sparse_row_sizes() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    MatrixMut::add(&mut sq, (1, 2)).unwrap();
    let sizes: Vec<usize> = sq.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1, 0]);
}

// ============================================================================
// SquareCSR2D AsRef
// ============================================================================

#[test]
fn test_square_csr_as_ref() {
    let mut sq: TestSquareCSR = SquareCSR2D::default();
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    let inner: &TestCSR = sq.as_ref();
    assert_eq!(inner.number_of_defined_values(), 1);
}

// ============================================================================
// SquareCSR2D SizedSparseMatrix2D
// ============================================================================

#[test]
fn test_square_csr_rank_row() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    MatrixMut::add(&mut sq, (1, 2)).unwrap();
    assert_eq!(sq.rank_row(0), 0);
    assert_eq!(sq.rank_row(1), 2);
    assert_eq!(sq.rank_row(2), 3);
}

#[test]
fn test_square_csr_select_row_col() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (1, 1)).unwrap();
    MatrixMut::add(&mut sq, (2, 2)).unwrap();
    assert_eq!(sq.select_row(0), 0);
    assert_eq!(sq.select_row(1), 1);
    assert_eq!(sq.select_row(2), 2);
    assert_eq!(sq.select_column(0), 0);
    assert_eq!(sq.select_column(1), 1);
    assert_eq!(sq.select_column(2), 2);
}

// ============================================================================
// SquareCSR2D MatrixMut::increase_shape
// ============================================================================

#[test]
fn test_square_csr_increase_shape_valid() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(2);
    sq.increase_shape((5, 5)).unwrap();
    assert_eq!(sq.number_of_rows(), 5);
    assert_eq!(sq.number_of_columns(), 5);
}

#[test]
fn test_square_csr_increase_shape_nonsquare_error() {
    let mut sq: TestSquareCSR = SparseMatrixMut::with_sparse_shape(2);
    assert!(sq.increase_shape((3, 5)).is_err());
}

// ============================================================================
// M2DValues (csr2d_values.rs) — sparse_values forward, backward,
// ExactSizeIterator
// ============================================================================

#[test]
fn test_sparse_values_forward() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)].into_iter())
        .build()
        .unwrap();
    let values: Vec<f64> = vcsr.sparse_values().collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_sparse_values_backward() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)].into_iter())
        .build()
        .unwrap();
    let values_rev: Vec<f64> = vcsr.sparse_values().rev().collect();
    assert_eq!(values_rev, vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_sparse_values_mixed() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(4)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)].into_iter())
        .build()
        .unwrap();
    let mut iter = vcsr.sparse_values();
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next_back(), Some(4.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next_back(), Some(3.0));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_sparse_values_exact_size() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(4)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)].into_iter())
        .build()
        .unwrap();
    let mut iter = vcsr.sparse_values();
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.len(), 2);
}

// ============================================================================
// sparse_row_values forward and backward
// ============================================================================

#[test]
fn test_sparse_row_values_forward() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 3))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)].into_iter())
        .build()
        .unwrap();
    let row0: Vec<f64> = vcsr.sparse_row_values(0).collect();
    assert_eq!(row0, vec![1.0, 2.0]);
    let row1: Vec<f64> = vcsr.sparse_row_values(1).collect();
    assert_eq!(row1, vec![3.0]);
}

#[test]
fn test_sparse_row_values_backward() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 3))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)].into_iter())
        .build()
        .unwrap();
    let row0_rev: Vec<f64> = vcsr.sparse_row_values(0).rev().collect();
    assert_eq!(row0_rev, vec![3.0, 2.0, 1.0]);
}
