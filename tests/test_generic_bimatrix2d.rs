//! Tests for the `GenericBiMatrix2D` struct.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SquareCSR2D},
    traits::{
        BiMatrix2D, EmptyRows, Matrix, Matrix2D, MatrixMut, SizedRowsSparseMatrix2D,
        SizedSparseMatrix, SparseMatrix, SparseMatrix2D, SparseMatrixMut, SquareMatrix,
        TransposableMatrix2D,
    },
};

type TestCSR2D = CSR2D<usize, usize, usize>;
type TestSquareCSR2D = SquareCSR2D<TestCSR2D>;
type TestBiMatrix = GenericBiMatrix2D<TestSquareCSR2D, TestSquareCSR2D>;

/// Helper to build a square CSR matrix.
fn build_square_matrix(order: usize, entries: Vec<(usize, usize)>) -> TestSquareCSR2D {
    let mut matrix: TestSquareCSR2D =
        SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    for entry in entries {
        MatrixMut::add(&mut matrix, entry).expect("Failed to add entry");
    }
    matrix
}

// ============================================================================
// Construction and basic properties
// ============================================================================

#[test]
fn test_bimatrix_new() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    assert_eq!(bi.number_of_rows(), 3);
    assert_eq!(bi.number_of_columns(), 3);
    assert_eq!(bi.number_of_defined_values(), 2);
}

#[test]
fn test_bimatrix_shape() {
    let inner = build_square_matrix(4, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);
    assert_eq!(bi.shape(), vec![4, 4]);
}

#[test]
fn test_bimatrix_is_empty() {
    let inner = build_square_matrix(2, vec![]);
    let bi = TestBiMatrix::new(inner);
    assert!(bi.is_empty());
}

#[test]
fn test_bimatrix_not_empty() {
    let inner = build_square_matrix(2, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);
    assert!(!bi.is_empty());
}

#[test]
fn test_bimatrix_order() {
    let inner = build_square_matrix(5, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);
    assert_eq!(bi.order(), 5);
}

// ============================================================================
// Sparse operations
// ============================================================================

#[test]
fn test_bimatrix_sparse_row() {
    let inner = build_square_matrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    let row0: Vec<usize> = bi.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 2]);

    let row1: Vec<usize> = bi.sparse_row(1).collect();
    assert_eq!(row1, vec![2]);

    let row2: Vec<usize> = bi.sparse_row(2).collect();
    assert!(row2.is_empty());
}

#[test]
fn test_bimatrix_has_entry() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    assert!(bi.has_entry(0, 1));
    assert!(bi.has_entry(1, 2));
    assert!(!bi.has_entry(0, 2));
    assert!(!bi.has_entry(2, 0));
}

#[test]
fn test_bimatrix_sparse_coordinates() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    let coords: Vec<(usize, usize)> =
        geometric_traits::traits::SparseMatrix::sparse_coordinates(&bi).collect();
    assert_eq!(coords, vec![(0, 1), (1, 2)]);
}

#[test]
fn test_bimatrix_last_sparse_coordinates() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);
    assert_eq!(bi.last_sparse_coordinates(), Some((1, 2)));
}

// ============================================================================
// Transpose via BiMatrix2D
// ============================================================================

#[test]
fn test_bimatrix_transpose() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    let transposed: TestSquareCSR2D = bi.transpose();
    assert!(transposed.has_entry(1, 0));
    assert!(transposed.has_entry(2, 1));
}

#[test]
fn test_bimatrix_matrix_and_transposed() {
    let inner = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    let matrix: &TestSquareCSR2D = bi.matrix();
    assert!(matrix.has_entry(0, 1));

    let transposed: &TestSquareCSR2D = bi.transposed();
    assert!(transposed.has_entry(1, 0));
}

// ============================================================================
// Debug / Clone / PartialEq
// ============================================================================

#[test]
fn test_bimatrix_debug() {
    let inner = build_square_matrix(2, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);
    let debug = format!("{bi:?}");
    assert!(debug.contains("GenericBiMatrix2D"));
}

#[test]
fn test_bimatrix_clone() {
    let inner = build_square_matrix(2, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);
    let cloned = bi.clone();
    assert_eq!(bi, cloned);
}

// ============================================================================
// Empty rows
// ============================================================================

#[test]
fn test_bimatrix_empty_rows() {
    let inner = build_square_matrix(3, vec![(0, 1)]);
    let bi = TestBiMatrix::new(inner);

    let empty: Vec<usize> = bi.empty_row_indices().collect();
    assert!(empty.contains(&1));
    assert!(empty.contains(&2));

    let non_empty: Vec<usize> = bi.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
}

// ============================================================================
// Row sizes
// ============================================================================

#[test]
fn test_bimatrix_row_sizes() {
    let inner = build_square_matrix(3, vec![(0, 1), (0, 2), (1, 2)]);
    let bi = TestBiMatrix::new(inner);

    assert_eq!(bi.number_of_defined_values_in_row(0), 2);
    assert_eq!(bi.number_of_defined_values_in_row(1), 1);
    assert_eq!(bi.number_of_defined_values_in_row(2), 0);

    let sizes: Vec<usize> = bi.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1, 0]);
}
