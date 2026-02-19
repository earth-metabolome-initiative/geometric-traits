//! Tests for Matrix trait methods: dimensions, total_values, density.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    traits::{
        Matrix, Matrix2D, MatrixMut, SizedSparseMatrix, SparseMatrix, SparseMatrixMut, SquareMatrix,
    },
};

type TestCSR2D = CSR2D<usize, usize, usize>;
type TestSquareCSR2D = SquareCSR2D<TestCSR2D>;

// ============================================================================
// Matrix::dimensions
// ============================================================================

#[test]
fn test_csr2d_dimensions() {
    assert_eq!(TestCSR2D::dimensions(), 2);
}

#[test]
fn test_square_csr2d_dimensions() {
    assert_eq!(TestSquareCSR2D::dimensions(), 2);
}

// ============================================================================
// Matrix::total_values
// ============================================================================

#[test]
fn test_total_values_square() {
    let mut matrix: TestSquareCSR2D = SquareCSR2D::with_sparse_shaped_capacity(3, 1);
    matrix.extend(vec![(0, 1)]).unwrap();
    // 3x3 = 9 total values
    assert_eq!(matrix.total_values(), 9);
}

#[test]
fn test_total_values_rect() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((2, 4), 1);
    csr.add((0, 0)).unwrap();
    // 2x4 = 8 total values
    assert_eq!(csr.total_values(), 8);
}

// ============================================================================
// SizedSparseMatrix::density
// ============================================================================

#[test]
fn test_density_full() {
    let csr: TestCSR2D = CSR2D::from_entries(vec![(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap();
    let density = csr.density();
    assert!((density - 1.0).abs() < 1e-10, "Full matrix density should be 1.0, got {density}");
}

#[test]
fn test_density_half() {
    let csr: TestCSR2D = CSR2D::from_entries(vec![(0, 0), (1, 1)]).unwrap();
    let density = csr.density();
    assert!((density - 0.5).abs() < 1e-10, "Half-filled density should be 0.5, got {density}");
}

#[test]
fn test_density_quarter() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((2, 2), 1);
    csr.add((0, 0)).unwrap();
    let density = csr.density();
    assert!((density - 0.25).abs() < 1e-10, "Quarter density should be 0.25, got {density}");
}

// ============================================================================
// SquareMatrix::order
// ============================================================================

#[test]
fn test_square_matrix_order() {
    let matrix: TestSquareCSR2D = SquareCSR2D::with_sparse_shape(5);
    assert_eq!(matrix.order(), 5);
}

// ============================================================================
// SparseMatrix: is_empty, last_sparse_coordinates
// ============================================================================

#[test]
fn test_sparse_matrix_empty() {
    let csr: TestCSR2D = CSR2D::with_sparse_shape((3, 3));
    assert!(csr.is_empty());
    assert_eq!(csr.last_sparse_coordinates(), None);
}

#[test]
fn test_sparse_matrix_not_empty() {
    let csr: TestCSR2D = CSR2D::from_entries(vec![(0, 1)]).unwrap();
    assert!(!csr.is_empty());
    assert_eq!(csr.last_sparse_coordinates(), Some((0, 1)));
}

// ============================================================================
// Matrix2D: number_of_rows, number_of_columns
// ============================================================================

#[test]
fn test_matrix2d_rows_cols() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((5, 3), 0);
    assert_eq!(csr.number_of_rows(), 5);
    assert_eq!(csr.number_of_columns(), 3);
    // No additions, should be empty
    assert!(csr.is_empty());
    // Add a value within bounds
    csr.add((4, 2)).unwrap();
    assert!(!csr.is_empty());
}

// ============================================================================
// Reference wrapper tests (Matrix for &M)
// ============================================================================

#[test]
fn test_matrix_ref_wrapper() {
    let csr: TestCSR2D = CSR2D::from_entries(vec![(0, 0), (1, 1)]).unwrap();
    let csr_ref = &csr;

    // Matrix trait via reference
    assert_eq!(csr_ref.shape(), csr.shape());
    assert_eq!(csr_ref.total_values(), csr.total_values());

    // SparseMatrix trait via reference
    assert_eq!(csr_ref.is_empty(), csr.is_empty());
    assert_eq!(csr_ref.last_sparse_coordinates(), csr.last_sparse_coordinates());

    // SizedSparseMatrix trait via reference
    assert_eq!(csr_ref.number_of_defined_values(), csr.number_of_defined_values());
}
