//! Extended tests for the CSR2D sparse matrix struct.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    prelude::{MatrixMut, SparseMatrixMut},
    traits::{
        EmptyRows, Matrix, Matrix2D, RankSelectSparseMatrix, SizedRowsSparseMatrix2D,
        SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, TransposableMatrix2D,
    },
};

type TestCSR2D = CSR2D<usize, usize, usize>;

/// Helper to build from entries.
fn build_csr(entries: Vec<(usize, usize)>) -> TestCSR2D {
    CSR2D::from_entries(entries).unwrap()
}

// ============================================================================
// Basic construction tests
// ============================================================================

#[test]
fn test_csr2d_empty() {
    let csr: TestCSR2D = CSR2D::with_sparse_shape((0, 0));
    assert!(csr.is_empty());
    assert_eq!(csr.number_of_rows(), 0);
    assert_eq!(csr.number_of_columns(), 0);
    assert_eq!(csr.number_of_defined_values(), 0);
}

#[test]
fn test_csr2d_shape() {
    let csr = build_csr(vec![(0, 0), (1, 1)]);
    let shape = csr.shape();
    assert_eq!(shape.len(), 2);
    assert!(shape[0] >= 2);
    assert!(shape[1] >= 2);
}

#[test]
fn test_csr2d_total_values() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 0)]);
    let total = csr.total_values();
    // total_values is product of shape dimensions
    assert!(total >= 4);
}

#[test]
fn test_csr2d_dimensions() {
    assert_eq!(TestCSR2D::dimensions(), 2);
}

// ============================================================================
// Sparse row tests
// ============================================================================

#[test]
fn test_csr2d_sparse_row_empty() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0)).unwrap();
    csr.add((0, 1)).unwrap();
    // Row 1 exists but has no entries
    let row1: Vec<usize> = csr.sparse_row(1).collect();
    assert!(row1.is_empty());
}

#[test]
fn test_csr2d_sparse_row_populated() {
    let csr = build_csr(vec![(0, 0), (0, 1), (0, 2)]);
    let row0: Vec<usize> = csr.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 1, 2]);
}

#[test]
fn test_csr2d_has_entry() {
    let csr = build_csr(vec![(0, 1), (1, 2)]);
    assert!(csr.has_entry(0, 1));
    assert!(csr.has_entry(1, 2));
    assert!(!csr.has_entry(0, 0));
    assert!(!csr.has_entry(0, 2));
    assert!(!csr.has_entry(1, 0));
}

// ============================================================================
// Sparse coordinates / iteration
// ============================================================================

#[test]
fn test_csr2d_sparse_coordinates() {
    let csr = build_csr(vec![(0, 1), (1, 0), (1, 2)]);
    let coords: Vec<(usize, usize)> = csr.sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 1), (1, 0), (1, 2)]);
}

#[test]
fn test_csr2d_sparse_coordinates_reverse() {
    let csr = build_csr(vec![(0, 1), (1, 0), (1, 2)]);
    let coords: Vec<(usize, usize)> = csr.sparse_coordinates().rev().collect();
    // Rev iterates backwards: last entry first
    assert_eq!(coords, vec![(1, 0), (1, 2), (0, 1)]);
}

#[test]
fn test_csr2d_last_sparse_coordinates() {
    let csr = build_csr(vec![(0, 1), (1, 2)]);
    assert_eq!(csr.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_csr2d_last_sparse_coordinates_empty() {
    let csr: TestCSR2D = CSR2D::with_sparse_shape((0, 0));
    assert_eq!(csr.last_sparse_coordinates(), None);
}

// ============================================================================
// Row sizes
// ============================================================================

#[test]
fn test_csr2d_sparse_row_sizes() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let sizes: Vec<usize> = csr.sparse_row_sizes().collect();
    assert_eq!(sizes[0], 2);
    assert_eq!(sizes[1], 1);
}

#[test]
fn test_csr2d_number_of_defined_values_in_row() {
    let csr = build_csr(vec![(0, 0), (0, 1), (0, 2), (1, 0)]);
    assert_eq!(csr.number_of_defined_values_in_row(0), 3);
    assert_eq!(csr.number_of_defined_values_in_row(1), 1);
}

// ============================================================================
// Empty rows
// ============================================================================

#[test]
fn test_csr2d_empty_row_indices() {
    let csr = build_csr(vec![(0, 1), (2, 3)]);
    let empty: Vec<usize> = csr.empty_row_indices().collect();
    assert!(empty.contains(&1));
}

#[test]
fn test_csr2d_non_empty_row_indices() {
    let csr = build_csr(vec![(0, 1), (2, 3)]);
    let non_empty: Vec<usize> = csr.non_empty_row_indices().collect();
    assert!(non_empty.contains(&0));
    assert!(non_empty.contains(&2));
}

#[test]
fn test_csr2d_number_of_empty_rows() {
    let csr = build_csr(vec![(0, 1), (2, 3)]);
    let total_rows = csr.number_of_rows();
    let non_empty = csr.number_of_non_empty_rows();
    let empty = csr.number_of_empty_rows();
    assert_eq!(empty + non_empty, total_rows);
}

// ============================================================================
// Rank/Select
// ============================================================================

#[test]
fn test_csr2d_rank_select() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)]);
    let rank = csr.rank(&(0, 1));
    assert_eq!(rank, 0);
    let rank2 = csr.rank(&(0, 2));
    assert_eq!(rank2, 1);

    let coords = csr.select(0);
    assert_eq!(coords, (0, 1));
    let coords2 = csr.select(2);
    assert_eq!(coords2, (1, 0));
}

#[test]
fn test_csr2d_rank_row_select_column() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)]);
    // rank_row gives the sparse index of the first entry in that row
    assert_eq!(csr.rank_row(0), 0);
    assert_eq!(csr.rank_row(1), 2);

    // select_column maps sparse index -> column
    assert_eq!(csr.select_column(0), 1);
    assert_eq!(csr.select_column(1), 2);
    assert_eq!(csr.select_column(2), 0);
}

// ============================================================================
// Transpose
// ============================================================================

#[test]
fn test_csr2d_transpose() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 2)]);
    let transposed: TestCSR2D = csr.transpose();

    // (0,1) becomes (1,0)
    assert!(transposed.has_entry(1, 0));
    // (0,2) becomes (2,0)
    assert!(transposed.has_entry(2, 0));
    // (1,2) becomes (2,1)
    assert!(transposed.has_entry(2, 1));

    assert!(!transposed.has_entry(0, 1));
}

#[test]
fn test_csr2d_transpose_empty_non_empty_row_invariants_with_empty_rows() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 2)]);
    let transposed: TestCSR2D = csr.transpose();

    // Transposed rows:
    // row 0 -> empty
    // row 1 -> [0]
    // row 2 -> [0, 1]
    assert_eq!(transposed.number_of_rows(), 3);
    assert_eq!(transposed.number_of_non_empty_rows(), 2);
    assert_eq!(transposed.number_of_empty_rows(), 1);
    assert_eq!(
        transposed.number_of_empty_rows() + transposed.number_of_non_empty_rows(),
        transposed.number_of_rows()
    );
}

#[test]
fn test_csr2d_transpose_empty_non_empty_row_invariants_fully_populated() {
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)]);
    let transposed: TestCSR2D = csr.transpose();

    assert_eq!(transposed.number_of_rows(), 3);
    assert_eq!(transposed.number_of_non_empty_rows(), 3);
    assert_eq!(transposed.number_of_empty_rows(), 0);
    assert_eq!(
        transposed.number_of_empty_rows() + transposed.number_of_non_empty_rows(),
        transposed.number_of_rows()
    );
}

// ============================================================================
// Sparse rows/columns iterators
// ============================================================================

#[test]
fn test_csr2d_rows_via_row_indices() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 2)]);
    let row0: Vec<usize> = csr.sparse_row(0).collect();
    let row1: Vec<usize> = csr.sparse_row(1).collect();
    assert_eq!(row0, vec![0, 1]);
    assert_eq!(row1, vec![2]);
}

#[test]
fn test_csr2d_sparse_columns() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 0)]);
    let columns: Vec<usize> = csr.sparse_columns().collect();
    assert_eq!(columns, vec![0, 1, 0]);
}

// ============================================================================
// Mutation / extend
// ============================================================================

#[test]
fn test_csr2d_add_entries() {
    let mut csr: TestCSR2D = CSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 1)).unwrap();
    csr.add((1, 2)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 2);
    assert!(csr.has_entry(0, 1));
    assert!(csr.has_entry(1, 2));
}

// ============================================================================
// Density
// ============================================================================

#[test]
fn test_csr2d_density() {
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    let density = csr.density();
    // 4 values out of 2x2 = 4 total → density = 1.0
    assert!((density - 1.0).abs() < 1e-10);
}

#[test]
fn test_csr2d_density_sparse() {
    let csr = build_csr(vec![(0, 1)]);
    let density = csr.density();
    // 1 value out of at least 2x2 = 4 total → density <= 0.5
    assert!(density > 0.0 && density <= 0.5);
}
