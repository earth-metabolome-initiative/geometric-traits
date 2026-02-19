//! Integration tests for RaggedVector sparse matrix.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::RaggedVector,
    traits::{
        EmptyRows, Matrix, Matrix2D, MatrixMut, SizedRowsSparseMatrix2D, SizedSparseMatrix,
        SparseMatrix, SparseMatrix2D, SparseMatrixMut, TransposableMatrix2D,
    },
};

type RV = RaggedVector<usize, usize, usize>;

// ============================================================================
// Construction tests
// ============================================================================

#[test]
fn test_default() {
    let rv: RV = RaggedVector::default();
    assert_eq!(rv.number_of_rows(), 0);
    assert_eq!(rv.number_of_columns(), 0);
    assert_eq!(rv.number_of_defined_values(), 0);
    assert!(rv.is_empty());
}

#[test]
fn test_with_sparse_shape() {
    let rv: RV = SparseMatrixMut::with_sparse_shape((5, 10));
    assert_eq!(rv.number_of_rows(), 5);
    assert_eq!(rv.number_of_columns(), 10);
    assert!(rv.is_empty());
}

#[test]
fn test_with_sparse_capacity() {
    let rv: RV = SparseMatrixMut::with_sparse_capacity(100);
    assert_eq!(rv.number_of_rows(), 0);
    assert_eq!(rv.number_of_columns(), 0);
}

#[test]
fn test_with_sparse_shaped_capacity() {
    let rv: RV = SparseMatrixMut::with_sparse_shaped_capacity((3, 4), 10);
    assert_eq!(rv.number_of_rows(), 3);
    assert_eq!(rv.number_of_columns(), 4);
}

// ============================================================================
// Add and query tests
// ============================================================================

#[test]
fn test_add_entries_updates_dimensions() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 0)).unwrap();
    assert_eq!(rv.number_of_rows(), 1);
    assert_eq!(rv.number_of_columns(), 1);

    rv.add((2, 5)).unwrap();
    assert_eq!(rv.number_of_rows(), 3);
    assert_eq!(rv.number_of_columns(), 6);
}

#[test]
fn test_add_duplicate_error() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    assert!(rv.add((0, 1)).is_err());
}

#[test]
fn test_add_unordered_error() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 3)).unwrap();
    assert!(rv.add((0, 1)).is_err());
}

#[test]
fn test_sparse_row() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    rv.add((0, 4)).unwrap();
    rv.add((1, 2)).unwrap();
    rv.add((1, 3)).unwrap();

    let row0: Vec<usize> = rv.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 4]);

    let row1: Vec<usize> = rv.sparse_row(1).collect();
    assert_eq!(row1, vec![2, 3]);

    // Row beyond stored data
    let row5: Vec<usize> = rv.sparse_row(5).collect();
    assert!(row5.is_empty());
}

#[test]
fn test_has_entry() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    rv.add((0, 3)).unwrap();
    rv.add((2, 2)).unwrap();

    assert!(rv.has_entry(0, 1));
    assert!(rv.has_entry(0, 3));
    assert!(!rv.has_entry(0, 0));
    assert!(!rv.has_entry(0, 2));
    assert!(rv.has_entry(2, 2));
    // Out of bounds row
    assert!(!rv.has_entry(5, 0));
}

// ============================================================================
// Shape and dimensions tests
// ============================================================================

#[test]
fn test_shape() {
    let mut rv: RV = RaggedVector::default();
    rv.add((1, 3)).unwrap();
    assert_eq!(rv.shape(), vec![2, 4]);
}

#[test]
fn test_total_values() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((3, 4));
    assert_eq!(rv.total_values(), 12);
    rv.add((0, 0)).unwrap();
    assert_eq!(rv.total_values(), 12); // total_values is rows * cols
}

// ============================================================================
// Empty rows tests
// ============================================================================

#[test]
fn test_empty_and_non_empty_rows() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((4, 4));
    rv.add((0, 1)).unwrap();
    rv.add((2, 3)).unwrap();

    assert_eq!(rv.number_of_non_empty_rows(), 2);
    assert_eq!(rv.number_of_empty_rows(), 2);

    let empty: Vec<usize> = rv.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 3]);

    let non_empty: Vec<usize> = rv.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 2]);
}

// ============================================================================
// Row sizes tests
// ============================================================================

#[test]
fn test_sparse_row_sizes() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((3, 5));
    rv.add((0, 0)).unwrap();
    rv.add((0, 2)).unwrap();
    rv.add((0, 4)).unwrap();
    rv.add((2, 1)).unwrap();

    let sizes: Vec<usize> = rv.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![3, 0, 1]);
}

#[test]
fn test_number_of_defined_values_in_row() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((3, 5));
    rv.add((0, 0)).unwrap();
    rv.add((0, 2)).unwrap();
    rv.add((1, 1)).unwrap();

    assert_eq!(rv.number_of_defined_values_in_row(0), 2);
    assert_eq!(rv.number_of_defined_values_in_row(1), 1);
    assert_eq!(rv.number_of_defined_values_in_row(2), 0);
}

// ============================================================================
// Sparse coordinates tests
// ============================================================================

#[test]
fn test_sparse_coordinates() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    rv.add((0, 2)).unwrap();
    rv.add((1, 0)).unwrap();

    let coords: Vec<(usize, usize)> = rv.sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 1), (0, 2), (1, 0)]);
}

#[test]
fn test_last_sparse_coordinates() {
    let mut rv: RV = RaggedVector::default();
    assert_eq!(rv.last_sparse_coordinates(), None);

    rv.add((0, 1)).unwrap();
    assert_eq!(rv.last_sparse_coordinates(), Some((0, 1)));

    rv.add((0, 3)).unwrap();
    assert_eq!(rv.last_sparse_coordinates(), Some((0, 3)));

    rv.add((2, 0)).unwrap();
    assert_eq!(rv.last_sparse_coordinates(), Some((2, 0)));
}

// ============================================================================
// Increase shape tests
// ============================================================================

#[test]
fn test_increase_shape() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((2, 3));
    rv.increase_shape((5, 5)).unwrap();
    assert_eq!(rv.number_of_rows(), 5);
    assert_eq!(rv.number_of_columns(), 5);
}

#[test]
fn test_increase_shape_error_smaller_rows() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((5, 5));
    assert!(rv.increase_shape((3, 5)).is_err());
}

#[test]
fn test_increase_shape_error_smaller_cols() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((5, 5));
    assert!(rv.increase_shape((5, 3)).is_err());
}

// ============================================================================
// Transpose tests
// ============================================================================

#[test]
fn test_transpose() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    rv.add((0, 2)).unwrap();
    rv.add((1, 0)).unwrap();

    let transposed: RV = rv.transpose();

    assert!(transposed.has_entry(1, 0));
    assert!(transposed.has_entry(2, 0));
    assert!(transposed.has_entry(0, 1));
    assert!(!transposed.has_entry(0, 0));

    assert_eq!(transposed.number_of_rows(), rv.number_of_columns());
    assert_eq!(transposed.number_of_columns(), rv.number_of_rows());
}

#[test]
fn test_transpose_empty() {
    let rv: RV = SparseMatrixMut::with_sparse_shape((3, 4));
    let transposed: RV = rv.transpose();

    assert_eq!(transposed.number_of_rows(), 4);
    assert_eq!(transposed.number_of_columns(), 3);
    assert!(transposed.is_empty());
}

// ============================================================================
// Debug and Clone tests
// ============================================================================

#[test]
fn test_debug() {
    let rv: RV = RaggedVector::default();
    let debug = format!("{rv:?}");
    assert!(debug.contains("RaggedVector"));
}

#[test]
fn test_clone() {
    let mut rv: RV = RaggedVector::default();
    rv.add((0, 1)).unwrap();

    let cloned = rv.clone();
    assert_eq!(cloned.number_of_defined_values(), 1);
    assert!(cloned.has_entry(0, 1));
}

// ============================================================================
// Density tests
// ============================================================================

#[test]
fn test_density() {
    let mut rv: RV = SparseMatrixMut::with_sparse_shape((2, 4));
    rv.add((0, 0)).unwrap();
    rv.add((0, 1)).unwrap();
    rv.add((1, 2)).unwrap();
    rv.add((1, 3)).unwrap();

    // 4 values out of 8 total = 0.5
    assert!((rv.density() - 0.5).abs() < f64::EPSILON);
}
