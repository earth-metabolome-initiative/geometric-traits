//! Tests for GenericImplicitValuedMatrix2D.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericImplicitValuedMatrix2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, ImplicitValuedMatrix, Matrix, Matrix2D, SizedSparseMatrix,
        SizedSparseValuedMatrix, SparseMatrix, SparseMatrix2D, SparseValuedMatrix,
        SparseValuedMatrix2D,
    },
};

fn implicit_val(row: usize, col: usize) -> f64 {
    let value = row * 10 + col;
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

/// Helper to build a CSR2D from edges.
fn build_csr(edges: Vec<(usize, usize)>, rows: usize, cols: usize) -> CSR2D<usize, usize, usize> {
    GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Basic matrix property tests
// ============================================================================

#[test]
fn test_shape() {
    let csr = build_csr(vec![(0, 1), (1, 0)], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert_eq!(m.shape(), vec![3, 3]);
}

#[test]
fn test_dimensions() {
    let csr = build_csr(vec![(0, 1)], 2, 5);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert_eq!(m.number_of_rows(), 2);
    assert_eq!(m.number_of_columns(), 5);
}

#[test]
fn test_number_of_defined_values() {
    let csr = build_csr(vec![(0, 1), (1, 0), (1, 2)], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert_eq!(m.number_of_defined_values(), 3);
}

#[test]
fn test_is_empty() {
    let csr = build_csr(vec![], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert!(m.is_empty());

    let csr = build_csr(vec![(0, 0)], 1, 1);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert!(!m.is_empty());
}

// ============================================================================
// Sparse structure tests
// ============================================================================

#[test]
fn test_sparse_coordinates() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).collect();
    assert_eq!(coords, vec![(0, 1), (0, 2), (1, 0)]);
}

#[test]
fn test_sparse_row() {
    let csr = build_csr(vec![(0, 1), (0, 3), (1, 2)], 2, 4);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    let row0: Vec<usize> = m.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 3]);

    let row1: Vec<usize> = m.sparse_row(1).collect();
    assert_eq!(row1, vec![2]);
}

#[test]
fn test_has_entry() {
    let csr = build_csr(vec![(0, 1), (1, 0)], 2, 2);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    assert!(m.has_entry(0, 1));
    assert!(m.has_entry(1, 0));
    assert!(!m.has_entry(0, 0));
    assert!(!m.has_entry(1, 1));
}

#[test]
fn test_last_sparse_coordinates() {
    let csr = build_csr(vec![(0, 1), (2, 3)], 3, 4);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert_eq!(SparseMatrix::last_sparse_coordinates(&m), Some((2, 3)));

    let csr = build_csr(vec![], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });
    assert_eq!(SparseMatrix::last_sparse_coordinates(&m), None);
}

// ============================================================================
// Implicit value tests
// ============================================================================

#[test]
fn test_implicit_value() {
    let csr = build_csr(vec![(0, 1), (2, 3)], 3, 4);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    // Map is |(row, col)| row * 10 + col
    assert!((m.implicit_value(&(0, 1)) - 1.0).abs() < f64::EPSILON);
    assert!((m.implicit_value(&(2, 3)) - 23.0).abs() < f64::EPSILON);
    // Also works for non-sparse coordinates
    assert!((m.implicit_value(&(1, 1)) - 11.0).abs() < f64::EPSILON);
}

// ============================================================================
// Sparse values tests
// ============================================================================

#[test]
fn test_sparse_values() {
    let csr = build_csr(vec![(0, 1), (1, 2), (2, 0)], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    let values: Vec<f64> = m.sparse_values().collect();
    assert_eq!(values.len(), 3);
    // (0,1) -> 1.0, (1,2) -> 12.0, (2,0) -> 20.0
    assert!((values[0] - 1.0).abs() < f64::EPSILON);
    assert!((values[1] - 12.0).abs() < f64::EPSILON);
    assert!((values[2] - 20.0).abs() < f64::EPSILON);
}

#[test]
fn test_sparse_row_values() {
    let csr = build_csr(vec![(0, 1), (0, 3), (1, 2)], 2, 4);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    let row0_values: Vec<f64> = m.sparse_row_values(0).collect();
    assert_eq!(row0_values.len(), 2);
    assert!((row0_values[0] - 1.0).abs() < f64::EPSILON); // (0,1)
    assert!((row0_values[1] - 3.0).abs() < f64::EPSILON); // (0,3)

    let row1_values: Vec<f64> = m.sparse_row_values(1).collect();
    assert_eq!(row1_values.len(), 1);
    assert!((row1_values[0] - 12.0).abs() < f64::EPSILON); // (1,2)
}

#[test]
fn test_select_value() {
    let csr = build_csr(vec![(0, 1), (1, 2), (2, 0)], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    // select_value(0) -> value at sparse index 0 -> (0,1) -> 1.0
    assert!((m.select_value(0) - 1.0).abs() < f64::EPSILON);
    // select_value(1) -> (1,2) -> 12.0
    assert!((m.select_value(1) - 12.0).abs() < f64::EPSILON);
    // select_value(2) -> (2,0) -> 20.0
    assert!((m.select_value(2) - 20.0).abs() < f64::EPSILON);
}

// ============================================================================
// EmptyRows tests
// ============================================================================

#[test]
fn test_empty_rows() {
    let csr = build_csr(vec![(0, 1), (2, 0)], 3, 3);
    let m = GenericImplicitValuedMatrix2D::new(csr, |(row, col): (usize, usize)| {
        implicit_val(row, col)
    });

    assert_eq!(m.number_of_empty_rows(), 1);
    assert_eq!(m.number_of_non_empty_rows(), 2);

    let empty: Vec<usize> = m.empty_row_indices().collect();
    assert_eq!(empty, vec![1]);

    let non_empty: Vec<usize> = m.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 2]);
}
