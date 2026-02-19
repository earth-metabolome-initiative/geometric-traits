//! Tests for blanket impls: SparseMatrix2D for &M and EmptyRows for &M,
//! using UFCS to ensure the blanket impl is exercised rather than auto-deref.
//! Also covers DenseValuedMatrix for &M and SparseValuedMatrix for &M.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, MatrixMut, SparseMatrix2D, SparseMatrixMut, SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_csr(entries: Vec<(usize, usize)>, shape: (usize, usize)) -> TestCSR {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut csr, (r, c)).unwrap();
    }
    csr
}

// ============================================================================
// SparseMatrix2D for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_sparse_row_ufcs() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let r = &csr;
    let row0: Vec<usize> = <&TestCSR as SparseMatrix2D>::sparse_row(&r, 0).collect();
    assert_eq!(row0, vec![0, 1]);
}

#[test]
fn test_ref_has_entry_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 2)], (2, 3));
    let r = &csr;
    assert!(<&TestCSR as SparseMatrix2D>::has_entry(&r, 0, 0));
    assert!(!<&TestCSR as SparseMatrix2D>::has_entry(&r, 0, 1));
    assert!(<&TestCSR as SparseMatrix2D>::has_entry(&r, 1, 2));
}

#[test]
fn test_ref_sparse_columns_ufcs() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let r = &csr;
    let cols: Vec<usize> = <&TestCSR as SparseMatrix2D>::sparse_columns(&r).collect();
    assert_eq!(cols, vec![0, 1, 2]);
}

#[test]
fn test_ref_sparse_rows_ufcs() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let r = &csr;
    let rows: Vec<usize> = <&TestCSR as SparseMatrix2D>::sparse_rows(&r).collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

// ============================================================================
// EmptyRows for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_number_of_empty_rows_ufcs() {
    let csr = build_csr(vec![(0, 0), (2, 1)], (3, 3));
    let r = &csr;
    assert_eq!(<&TestCSR as EmptyRows>::number_of_empty_rows(&r), 1);
}

#[test]
fn test_ref_number_of_non_empty_rows_ufcs() {
    let csr = build_csr(vec![(0, 0), (2, 1)], (3, 3));
    let r = &csr;
    assert_eq!(<&TestCSR as EmptyRows>::number_of_non_empty_rows(&r), 2);
}

#[test]
fn test_ref_empty_row_indices_ufcs() {
    let csr = build_csr(vec![(0, 0), (2, 1)], (3, 3));
    let r = &csr;
    let empty: Vec<usize> = <&TestCSR as EmptyRows>::empty_row_indices(&r).collect();
    assert_eq!(empty, vec![1]);
}

#[test]
fn test_ref_non_empty_row_indices_ufcs() {
    let csr = build_csr(vec![(0, 0), (2, 1)], (3, 3));
    let r = &csr;
    let non_empty: Vec<usize> = <&TestCSR as EmptyRows>::non_empty_row_indices(&r).collect();
    assert_eq!(non_empty, vec![0, 2]);
}

// ============================================================================
// SparseValuedMatrix2D for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_sparse_row_values_ufcs() {
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 3))
        .edges(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0)].into_iter())
        .build()
        .unwrap();
    let r = &vcsr;
    let vals: Vec<f64> = <&TestValCSR as SparseValuedMatrix2D>::sparse_row_values(&r, 0).collect();
    assert_eq!(vals, vec![1.0, 2.0]);
}

// ============================================================================
// DenseValuedMatrix for &M (UFCS)  — covers matrix.rs lines 231-237
// ============================================================================

#[test]
fn test_ref_dense_valued_matrix_value_ufcs() {
    use geometric_traits::{impls::PaddedMatrix2D, traits::DenseValuedMatrix};
    type TestPadded = PaddedMatrix2D<TestValCSR, Box<dyn Fn((usize, usize)) -> f64>>;

    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(1)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 5.0)].into_iter())
        .build()
        .unwrap();
    let padded = PaddedMatrix2D::new(
        inner,
        Box::new(|(r, c): (usize, usize)| usize_to_f64(r * 10 + c))
            as Box<dyn Fn((usize, usize)) -> f64>,
    )
    .unwrap();
    let r = &padded;
    // Value at (0,0) should be 5.0 (sparse), (0,1) should be imputed 1.0
    assert!((<&TestPadded as DenseValuedMatrix>::value(&r, (0, 0)) - 5.0).abs() < f64::EPSILON);
    assert!((<&TestPadded as DenseValuedMatrix>::value(&r, (0, 1)) - 1.0).abs() < f64::EPSILON);
}
