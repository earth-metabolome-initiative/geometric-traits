//! Tests for ExactSizeIterator::len() on CSR2DSizedRows (returned by
//! CSR2D::sparse_rows()).
#![cfg(feature = "std")]

use geometric_traits::{impls::CSR2D, prelude::*, traits::SparseMatrix2D};

type TestCSR = CSR2D<usize, usize, usize>;

fn build_csr(entries: Vec<(usize, usize)>, shape: (usize, usize)) -> TestCSR {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut csr, (r, c)).unwrap();
    }
    csr
}

// ============================================================================
// CSR2DSizedRows::len() — CSR2D::sparse_rows() returns CSR2DSizedRows
// ============================================================================

#[test]
fn test_csr2d_sparse_rows_len_initial() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)], (3, 4));
    let iter = <TestCSR as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.len(), 4);
}

#[test]
fn test_csr2d_sparse_rows_len_after_next() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3)], (3, 4));
    let mut iter = <TestCSR as SparseMatrix2D>::sparse_rows(&csr);
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
fn test_csr2d_sparse_rows_len_after_next_back() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], (3, 3));
    let mut iter = <TestCSR as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.len(), 3);
    iter.next_back();
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_csr2d_sparse_rows_len_multi_entry() {
    let csr = build_csr(vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)], (3, 3));
    let mut iter = <TestCSR as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.len(), 6);
    // Consume entries from first row
    iter.next();
    assert_eq!(iter.len(), 5);
    iter.next();
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
    // Now in second row
    iter.next();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
    // Third row
    iter.next();
    assert_eq!(iter.len(), 0);
}

#[test]
fn test_csr2d_sparse_rows_len_mixed_next_and_next_back() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2), (2, 3), (2, 4)], (3, 5));
    let mut iter = <TestCSR as SparseMatrix2D>::sparse_rows(&csr);
    assert_eq!(iter.len(), 5);
    iter.next();
    assert_eq!(iter.len(), 4);
    iter.next_back();
    assert_eq!(iter.len(), 3);
}
