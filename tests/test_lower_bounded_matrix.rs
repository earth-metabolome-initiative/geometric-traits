//! Tests for LowerBoundedSquareMatrix: construction, sparse_columns
//! (CSR2DColumns), sparse_rows (CSR2DRows), and related iterators.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, LowerBoundedSquareMatrix, SquareCSR2D},
    traits::{Matrix2D, MatrixMut, SparseMatrix, SparseMatrix2D, SparseMatrixMut, SquareMatrix},
};

type TestCSR2D = CSR2D<usize, usize, usize>;
type TestSquareCSR2D = SquareCSR2D<TestCSR2D>;

fn build_square(order: usize, entries: Vec<(usize, usize)>) -> TestSquareCSR2D {
    let mut m: TestSquareCSR2D = SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    m.extend(entries).unwrap();
    m
}

// ============================================================================
// Construction
// ============================================================================

#[test]
fn test_lower_bounded_new() {
    let m = build_square(4, vec![(0, 1), (1, 2), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 1).unwrap();
    assert_eq!(lb.order(), 4);
    assert_eq!(lb.number_of_rows(), 4);
    assert_eq!(lb.number_of_columns(), 4);
}

#[test]
fn test_lower_bounded_out_of_bounds() {
    let m = build_square(3, vec![(0, 1)]);
    let result = LowerBoundedSquareMatrix::new(m, 3);
    assert!(result.is_err());
}

#[test]
fn test_lower_bounded_debug() {
    let m = build_square(3, vec![(0, 1), (1, 2)]);
    let lb = LowerBoundedSquareMatrix::new(m, 1).unwrap();
    let debug = format!("{lb:?}");
    assert!(debug.contains("LowerBoundedSquareMatrix"));
}

// ============================================================================
// SparseMatrix / SparseMatrix2D
// ============================================================================

#[test]
fn test_lower_bounded_has_entry() {
    let m = build_square(4, vec![(0, 1), (1, 2), (1, 3), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();

    assert!(!lb.has_entry(0, 1));
    assert!(!lb.has_entry(1, 2));
    assert!(!lb.has_entry(1, 3));
    assert!(lb.has_entry(2, 3));
}

#[test]
fn test_lower_bounded_sparse_row() {
    let m = build_square(4, vec![(0, 1), (1, 2), (1, 3), (2, 1), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();

    let row2: Vec<usize> = lb.sparse_row(2).collect();
    assert_eq!(row2, vec![3]);

    let row0: Vec<usize> = lb.sparse_row(0).collect();
    assert!(row0.is_empty());
}

#[test]
fn test_lower_bounded_sparse_row_rev() {
    let m = build_square(5, vec![(2, 2), (2, 3), (2, 4)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();

    let row2_rev: Vec<usize> = lb.sparse_row(2).rev().collect();
    assert_eq!(row2_rev, vec![4, 3, 2]);
}

#[test]
fn test_lower_bounded_sparse_row_clone() {
    let m = build_square(5, vec![(2, 2), (2, 3), (2, 4)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();

    let iter = lb.sparse_row(2);
    let cloned = iter.clone();
    let vals1: Vec<usize> = iter.collect();
    let vals2: Vec<usize> = cloned.collect();
    assert_eq!(vals1, vals2);
}

#[test]
fn test_lower_bounded_is_empty() {
    let m = build_square(4, vec![(0, 1), (1, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();
    assert!(lb.is_empty());
}

#[test]
fn test_lower_bounded_not_empty() {
    let m = build_square(4, vec![(0, 1), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();
    assert!(!lb.is_empty());
}

#[test]
fn test_lower_bounded_last_sparse_coordinates() {
    let m = build_square(4, vec![(0, 1), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();
    assert_eq!(lb.last_sparse_coordinates(), Some((2, 3)));
}

#[test]
fn test_lower_bounded_last_sparse_coordinates_none() {
    let m = build_square(4, vec![(0, 1), (1, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 2).unwrap();
    assert_eq!(lb.last_sparse_coordinates(), None);
}

// ============================================================================
// sparse_columns() → exercises CSR2DColumns Iterator + From
// Use bound=0 so rows aren't filtered, ensuring consecutive non-empty rows.
// ============================================================================

#[test]
fn test_lower_bounded_sparse_columns_forward() {
    // bound=0 means all entries visible. 3x3 with entries at rows 0 and 1.
    let m = build_square(3, vec![(0, 1), (0, 2), (1, 0), (1, 2)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let cols: Vec<usize> = lb.sparse_columns().collect();
    // next_row=0: [1, 2], next_row=1 < back_row=2: [0, 2], next_row=2 → back(row
    // 2): []
    assert_eq!(cols, vec![1, 2, 0, 2]);
}

#[test]
fn test_lower_bounded_sparse_columns_all_rows_filled() {
    // 3x3 with entries at every row including the back row
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let cols: Vec<usize> = lb.sparse_columns().collect();
    // next_row=0: [1], next_row=1 < back_row=2: [2], next_row=2 → back(row 2): [0]
    assert_eq!(cols, vec![1, 2, 0]);
}

#[test]
fn test_lower_bounded_sparse_columns_rev_partial() {
    // 3x3 with entries at back row (2) and row 1. Use .take() to avoid overflow.
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0), (2, 1)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    // Take exactly the number of entries (4) from reverse
    let cols_rev: Vec<usize> = lb.sparse_columns().rev().take(4).collect();
    // back(row 2) = [0, 1], then row 1: [2], then falls to next row 0: [1]
    assert_eq!(cols_rev, vec![0, 1, 2, 1]);
}

#[test]
fn test_lower_bounded_sparse_columns_multi_entry_rows() {
    // 4x4, bound=0, consecutive rows with multiple entries
    let m = build_square(4, vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let cols: Vec<usize> = lb.sparse_columns().collect();
    // Row 0: [0,1], Row 1: [1,2], Row 2: [2,3], Row 3 (back): []
    assert_eq!(cols, vec![0, 1, 1, 2, 2, 3]);
}

// ============================================================================
// sparse_rows() → exercises CSR2DRows Iterator + From
// ============================================================================

#[test]
fn test_lower_bounded_sparse_rows_forward() {
    let m = build_square(3, vec![(0, 1), (0, 2), (1, 0), (1, 2)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let rows: Vec<usize> = lb.sparse_rows().collect();
    // Row 0 has 2 entries → [0, 0], Row 1 has 2 entries → [1, 1]
    assert_eq!(rows, vec![0, 0, 1, 1]);
}

#[test]
fn test_lower_bounded_sparse_rows_all_filled() {
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let rows: Vec<usize> = lb.sparse_rows().collect();
    assert_eq!(rows, vec![0, 1, 2]);
}

#[test]
fn test_lower_bounded_sparse_rows_rev_partial() {
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0), (2, 1)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    // Take exactly the number of entries (4) from reverse
    let rows_rev: Vec<usize> = lb.sparse_rows().rev().take(4).collect();
    // back(row 2) has 2 entries → [2, 2], then row 1: [1], then falls to next row
    // 0: [0]
    assert_eq!(rows_rev, vec![2, 2, 1, 0]);
}

// ============================================================================
// sparse_coordinates() → exercises CSR2DView on LowerBoundedSquareMatrix
// ============================================================================

#[test]
fn test_lower_bounded_sparse_coordinates_forward() {
    // bound=0 with entries at rows 0,1. Row 2 (back) is empty.
    let m = build_square(3, vec![(0, 1), (1, 2)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let coords: Vec<(usize, usize)> = lb.sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 1), (1, 2)]);
}

#[test]
fn test_lower_bounded_sparse_coordinates_all_filled() {
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    let coords: Vec<(usize, usize)> = lb.sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 1), (1, 2), (2, 0)]);
}

#[test]
fn test_lower_bounded_sparse_coordinates_count() {
    let m = build_square(3, vec![(0, 1), (1, 2), (2, 0)]);
    let lb = LowerBoundedSquareMatrix::new(m, 0).unwrap();

    assert_eq!(lb.sparse_coordinates().count(), 3);
}
