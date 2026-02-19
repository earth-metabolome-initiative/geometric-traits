//! Tests for ImplicitValuedSparseRowIterator (Clone, ExactSizeIterator,
//! DoubleEndedIterator) and ImplicitSparseValuedMatrix2D blanket impl.
//! Also covers ValuedCSR2D delegations through sparse_rows, sparse_columns,
//! has_entry.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericImplicitValuedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, ImplicitSparseValuedMatrix2D, RankSelectSparseMatrix, SizedSparseMatrix2D,
        SparseMatrix, SparseMatrix2D, SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_implicit(
    entries: Vec<(usize, usize)>,
    shape: (usize, usize),
) -> GenericImplicitValuedMatrix2D<TestCSR, impl Fn((usize, usize)) -> f64, f64> {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut csr, (r, c)).unwrap();
    }
    GenericImplicitValuedMatrix2D::new(csr, |(r, c)| usize_to_f64(r * 10 + c))
}

// ============================================================================
// ImplicitSparseValuedMatrix2D::sparse_row_implicit_values
// ============================================================================

#[test]
fn test_sparse_row_implicit_values_forward() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let vals: Vec<f64> = m.sparse_row_implicit_values(0).collect();
    assert_eq!(vals, vec![0.0, 1.0]);
}

#[test]
fn test_sparse_row_implicit_values_backward() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let vals: Vec<f64> = m.sparse_row_implicit_values(0).rev().collect();
    assert_eq!(vals, vec![1.0, 0.0]);
}

#[test]
fn test_sparse_row_implicit_values_exact_size() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let iter = m.sparse_row_implicit_values(0);
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_sparse_row_implicit_values_clone() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let iter = m.sparse_row_implicit_values(0);
    let cloned = iter.clone();
    let original: Vec<f64> = iter.collect();
    let from_clone: Vec<f64> = cloned.collect();
    assert_eq!(original, from_clone);
}

#[test]
fn test_sparse_row_implicit_values_mixed() {
    let m = build_implicit(vec![(0, 0), (0, 1), (0, 2)], (1, 3));
    let mut iter = m.sparse_row_implicit_values(0);
    assert_eq!(iter.next(), Some(0.0));
    assert_eq!(iter.next_back(), Some(2.0));
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next(), None);
}

// ============================================================================
// ValuedCSR2D delegations: sparse_rows, sparse_columns, has_entry
// ============================================================================

fn build_valued_csr(rows: usize, cols: usize, entries: Vec<(usize, usize, f64)>) -> TestValCSR {
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(entries.len())
        .expected_shape((rows, cols))
        .edges(entries.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_valued_csr_sparse_rows() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0), (2, 0, 4.0)]);
    let rows: Vec<usize> = vcsr.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1, 2]);
}

#[test]
fn test_valued_csr_sparse_columns() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)]);
    let cols: Vec<usize> = vcsr.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 2]);
}

#[test]
fn test_valued_csr_has_entry() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)]);
    assert!(vcsr.has_entry(0, 0));
    assert!(vcsr.has_entry(0, 1));
    assert!(!vcsr.has_entry(0, 2));
    assert!(vcsr.has_entry(1, 2));
}

#[test]
fn test_valued_csr_sparse_row() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)]);
    let row0: Vec<usize> = vcsr.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1]);
}

// ============================================================================
// ValuedCSR2D: SparseMatrix delegations (sparse_coordinates, last, is_empty)
// ============================================================================

#[test]
fn test_valued_csr_sparse_coordinates() {
    let vcsr = build_valued_csr(2, 2, vec![(0, 0, 1.0), (1, 1, 2.0)]);
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&vcsr).collect();
    assert_eq!(coords, vec![(0, 0), (1, 1)]);
}

#[test]
fn test_valued_csr_last_sparse_coordinates() {
    let vcsr = build_valued_csr(2, 3, vec![(0, 0, 1.0), (1, 2, 3.0)]);
    assert_eq!(SparseMatrix::last_sparse_coordinates(&vcsr), Some((1, 2)));
}

#[test]
fn test_valued_csr_is_empty() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_shape((2, 2))
        .edges(Vec::<(usize, usize, f64)>::new().into_iter())
        .build()
        .unwrap();
    assert!(SparseMatrix::is_empty(&vcsr));
}

// ============================================================================
// ValuedCSR2D: RankSelectSparseMatrix delegations
// ============================================================================

#[test]
fn test_valued_csr_rank_select() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    assert_eq!(RankSelectSparseMatrix::rank(&vcsr, &(0, 0)), 0);
    assert_eq!(RankSelectSparseMatrix::rank(&vcsr, &(1, 1)), 1);
    assert_eq!(RankSelectSparseMatrix::select(&vcsr, 0), (0, 0));
    assert_eq!(RankSelectSparseMatrix::select(&vcsr, 1), (1, 1));
}

// ============================================================================
// ValuedCSR2D: SizedSparseMatrix2D delegations
// ============================================================================

#[test]
fn test_valued_csr_rank_row_select_row_col() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    assert_eq!(SizedSparseMatrix2D::rank_row(&vcsr, 0), 0);
    assert_eq!(SizedSparseMatrix2D::rank_row(&vcsr, 1), 1);
    assert_eq!(SizedSparseMatrix2D::select_row(&vcsr, 0), 0);
    assert_eq!(SizedSparseMatrix2D::select_row(&vcsr, 1), 1);
    assert_eq!(SizedSparseMatrix2D::select_column(&vcsr, 0), 0);
    assert_eq!(SizedSparseMatrix2D::select_column(&vcsr, 1), 1);
}

// ============================================================================
// sparse_row_max_values / sparse_row_min_values
// ============================================================================

#[test]
fn test_sparse_row_max_values() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0), (2, 2, 2.0)]);
    let maxes: Vec<Option<f64>> = vcsr.sparse_row_max_values().collect();
    assert_eq!(maxes, vec![Some(5.0), Some(3.0), Some(2.0)]);
}

#[test]
fn test_sparse_row_min_values() {
    let vcsr = build_valued_csr(3, 3, vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0), (2, 2, 2.0)]);
    let mins: Vec<Option<f64>> = vcsr.sparse_row_min_values().collect();
    assert_eq!(mins, vec![Some(1.0), Some(3.0), Some(2.0)]);
}
