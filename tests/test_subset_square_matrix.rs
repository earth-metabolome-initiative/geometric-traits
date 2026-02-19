//! Test submodule for the `SubsetSquareMatrix` struct.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SubsetSquareMatrix},
    traits::{Matrix2D, MatrixMut, SparseMatrix, SparseMatrix2D, SparseMatrixMut},
};

type TestCSR2D = CSR2D<usize, usize, usize>;
type TestSquareCSR2D = SquareCSR2D<TestCSR2D>;

/// Helper to build a square CSR matrix from edges.
fn build_square_matrix(order: usize, entries: Vec<(usize, usize)>) -> TestSquareCSR2D {
    let mut matrix: TestSquareCSR2D =
        SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    matrix.extend(entries).expect("Failed to extend matrix");
    matrix
}

#[test]
fn test_subset_with_sorted_indices() {
    let matrix = build_square_matrix(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 2]);

    // Columns are filtered by subset {0, 2}: only column indices in the subset
    // appear.
    assert!(subset.has_entry(0, 2));
    assert!(!subset.has_entry(0, 1)); // column 1 not in subset
    // Row 1 still exists; its column 2 is in the subset
    assert!(subset.has_entry(1, 2));
    assert!(!subset.has_entry(1, 3)); // column 3 not in subset
}

#[test]
fn test_subset_with_unsorted_indices() {
    let matrix = build_square_matrix(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    let subset = SubsetSquareMatrix::with_unsorted_indices(matrix, vec![2, 0].into_iter())
        .expect("Should succeed with valid indices");

    // Column filtering with subset {0, 2}
    assert!(subset.has_entry(0, 2));
    assert!(!subset.has_entry(0, 1)); // column 1 not in subset
}

#[test]
fn test_subset_out_of_bounds_error() {
    let matrix = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let result = SubsetSquareMatrix::with_unsorted_indices(matrix, vec![0, 5].into_iter());
    assert!(result.is_err(), "Should fail with out-of-bounds index");
}

#[test]
fn test_subset_empty_indices() {
    let matrix = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, Vec::<usize>::new());

    // No indices means no entries visible
    assert!(subset.is_empty());
}

#[test]
fn test_subset_all_indices() {
    let entries = vec![(0, 1), (1, 2)];
    let matrix = build_square_matrix(3, entries.clone());
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 1, 2]);

    // All entries should be visible when all indices are included
    assert!(subset.has_entry(0, 1));
    assert!(subset.has_entry(1, 2));
    assert!(!subset.has_entry(0, 2));
}

#[test]
fn test_subset_sparse_row() {
    let matrix = build_square_matrix(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 2, 4]);

    let row0: Vec<usize> = subset.sparse_row(0).collect();
    // Only columns in subset {0, 2, 4} should appear
    assert_eq!(row0, vec![2, 4]);
}

#[test]
fn test_subset_sparse_coordinates() {
    let matrix = build_square_matrix(4, vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 3]);

    let coords: Vec<(usize, usize)> = subset.sparse_coordinates().collect();
    // Columns are filtered to {0, 3}: (0,3) and (2,3) survive
    assert_eq!(coords, vec![(0, 3), (2, 3)]);
}

#[test]
fn test_subset_default() {
    let subset: SubsetSquareMatrix<TestSquareCSR2D, Vec<usize>> = SubsetSquareMatrix::default();
    assert!(subset.is_empty());
}

#[test]
fn test_subset_preserves_shape() {
    let matrix = build_square_matrix(5, vec![(0, 1), (1, 2)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 1]);

    // Shape comes from the underlying matrix, not the subset
    assert_eq!(subset.number_of_rows(), 5);
    assert_eq!(subset.number_of_columns(), 5);
}

// ============================================================================
// sparse_columns() → exercises CSR2DColumns via SubsetSquareMatrix
// ============================================================================

#[test]
fn test_subset_sparse_columns() {
    // 3x3 matrix, all entries at rows 0 and 1. Subset = all indices.
    let matrix = build_square_matrix(3, vec![(0, 0), (0, 1), (1, 0), (1, 2)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 1, 2]);

    let cols: Vec<usize> = subset.sparse_columns().collect();
    // Row 0: intersection of [0,1] with [0,1,2] → [0, 1]
    // Row 1: intersection of [0,2] with [0,1,2] → [0, 2]
    // Row 2 (back): no entries → []
    assert_eq!(cols, vec![0, 1, 0, 2]);
}

#[test]
fn test_subset_sparse_columns_filtered() {
    let matrix = build_square_matrix(4, vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 2]);

    let cols: Vec<usize> = subset.sparse_columns().collect();
    // Row 0: intersection of [0,1,2,3] with [0,2] → [0, 2]
    assert_eq!(cols, vec![0, 2]);
}

// ============================================================================
// sparse_rows() → exercises CSR2DRows via SubsetSquareMatrix
// ============================================================================

#[test]
fn test_subset_sparse_rows() {
    let matrix = build_square_matrix(3, vec![(0, 0), (0, 1), (1, 0), (1, 2)]);
    let subset = SubsetSquareMatrix::with_sorted_indices(matrix, vec![0, 1, 2]);

    let rows: Vec<usize> = subset.sparse_rows().collect();
    // Row 0 has 2 filtered entries → [0, 0], Row 1 has 2 filtered entries → [1, 1]
    assert_eq!(rows, vec![0, 0, 1, 1]);
}
