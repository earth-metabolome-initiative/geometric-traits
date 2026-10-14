//! Tests for reference wrapper impls of matrix traits:
//! &M impls for Matrix, ValuedMatrix, SparseMatrix, SizedSparseMatrix,
//! SparseMatrix2D, SparseValuedMatrix, DenseValuedMatrix, DenseValuedMatrix2D,
//! EmptyRows, SizedRowsSparseMatrix2D.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{
        DenseValuedMatrix, DenseValuedMatrix2D, EdgesBuilder, EmptyRows, Matrix2D,
        SizedRowsSparseMatrix2D, SizedSparseMatrix, SparseMatrix2D, SparseValuedMatrix,
        SparseValuedMatrix2D,
    },
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

/// Helper to build a ValuedCSR2D.
fn build_valued_csr(edges: Vec<(usize, usize, f64)>, rows: usize, cols: usize) -> TestValCSR {
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// ValuedMatrix reference impl
// ============================================================================

#[test]
fn test_valued_matrix_ref() {
    let csr = build_valued_csr(vec![(0, 1, 5.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    // ValuedMatrix via reference should have same Value type
    let vals: Vec<f64> = SparseValuedMatrix::sparse_values(csr_ref).collect();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 5.0).abs() < f64::EPSILON);
}

// ============================================================================
// SparseValuedMatrix reference impl
// ============================================================================

#[test]
fn test_sparse_valued_matrix_ref_max_min() {
    let csr = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    assert_eq!(SparseValuedMatrix::max_sparse_value(csr_ref), Some(5.0));
    assert_eq!(SparseValuedMatrix::min_sparse_value(csr_ref), Some(1.0));
}

// ============================================================================
// SparseValuedMatrix2D reference impl (sparse_row_values)
// ============================================================================

#[test]
fn test_sparse_valued_matrix2d_ref() {
    let csr = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    let row0_vals: Vec<f64> = csr_ref.sparse_row_values(0).collect();
    assert_eq!(row0_vals, vec![1.0, 2.0]);

    let row1_vals: Vec<f64> = csr_ref.sparse_row_values(1).collect();
    assert_eq!(row1_vals, vec![3.0]);
}

// ============================================================================
// SparseMatrix2D reference impl
// ============================================================================

#[test]
fn test_sparse_matrix2d_ref_sparse_row() {
    let csr = build_valued_csr(vec![(0, 1, 1.0), (0, 2, 2.0), (1, 0, 3.0)], 2, 3);
    let csr_ref: &TestValCSR = &csr;

    let row0: Vec<usize> = csr_ref.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 2]);

    assert!(csr_ref.has_entry(0, 1));
    assert!(!csr_ref.has_entry(0, 0));
}

#[test]
fn test_sparse_matrix2d_ref_columns_rows() {
    let csr = build_valued_csr(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    let cols: Vec<usize> = csr_ref.sparse_columns().collect();
    assert_eq!(cols, vec![1, 0]);

    let rows: Vec<usize> = csr_ref.sparse_rows().collect();
    assert_eq!(rows, vec![0, 1]);
}

// ============================================================================
// EmptyRows reference impl
// ============================================================================

#[test]
fn test_empty_rows_ref() {
    let csr = build_valued_csr(vec![(0, 1, 1.0)], 3, 3);
    let csr_ref: &TestValCSR = &csr;

    assert_eq!(csr_ref.number_of_empty_rows(), 2);
    assert_eq!(csr_ref.number_of_non_empty_rows(), 1);

    let empty: Vec<usize> = csr_ref.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 2]);

    let non_empty: Vec<usize> = csr_ref.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
}

// ============================================================================
// SizedRowsSparseMatrix2D reference impl
// ============================================================================

#[test]
fn test_sized_rows_ref() {
    let csr = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    let sizes: Vec<usize> = csr_ref.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1]);

    assert_eq!(csr_ref.number_of_defined_values_in_row(0), 2);
    assert_eq!(csr_ref.number_of_defined_values_in_row(1), 1);
}

// ============================================================================
// DenseValuedMatrix reference impl via PaddedMatrix2D
// ============================================================================

#[test]
fn test_dense_valued_matrix_ref() {
    // PaddedMatrix2D makes it square: 2x2 -> 2x2
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 2.0)], 2, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();
    let padded_ref: &PaddedMatrix2D<_, _> = &padded;

    assert!((padded_ref.value((0, 0)) - 1.0).abs() < f64::EPSILON);
    assert!((padded_ref.value((0, 1)) - 2.0).abs() < f64::EPSILON);

    let values: Vec<f64> = padded_ref.values().collect();
    // 2x2 = 4 values: (0,0)=1.0, (0,1)=2.0, (1,0)=0.0, (1,1)=0.0
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < f64::EPSILON);
    assert!((values[1] - 2.0).abs() < f64::EPSILON);
}

// ============================================================================
// DenseValuedMatrix2D reference impl via PaddedMatrix2D
// ============================================================================

#[test]
fn test_dense_valued_matrix2d_ref() {
    let inner = build_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0)], 1, 2);
    let padded = PaddedMatrix2D::new(inner, |_: (usize, usize)| 0.0).unwrap();
    let padded_ref: &PaddedMatrix2D<_, _> = &padded;

    let row0: Vec<f64> = padded_ref.row_values(0).collect();
    assert_eq!(row0, vec![1.0, 5.0]);
}

// ============================================================================
// Matrix2D reference impl
// ============================================================================

#[test]
fn test_matrix2d_ref_shape() {
    let csr = build_valued_csr(vec![(0, 1, 1.0)], 2, 3);
    let csr_ref: &TestValCSR = &csr;

    assert_eq!(csr_ref.number_of_rows(), 2);
    assert_eq!(csr_ref.number_of_columns(), 3);
}

// ============================================================================
// SizedSparseMatrix::density reference impl
// ============================================================================

#[test]
fn test_density_via_reference() {
    let csr = build_valued_csr(vec![(0, 0, 1.0), (1, 1, 2.0)], 2, 2);
    let csr_ref: &TestValCSR = &csr;

    // 2 values out of 4 total = 0.5
    assert!((csr_ref.density() - 0.5).abs() < f64::EPSILON);
}
