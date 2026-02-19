//! Tests for RankSelectSparseMatrix and SizedSparseMatrix2D:
//! rank, select, rank_row, select_row, select_column.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    prelude::*,
    traits::{
        EdgesBuilder, Matrix, RankSelectSparseMatrix, SizedSparseMatrix, SizedSparseMatrix2D,
        SparseMatrix, SparseMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;

/// Helper to build a CSR2D from edges.
fn build_csr(edges: Vec<(usize, usize)>, rows: usize, cols: usize) -> TestCSR {
    GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// RankSelectSparseMatrix::rank tests
// ============================================================================

#[test]
fn test_rank_first_entry() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    assert_eq!(csr.rank(&(0, 1)), 0);
}

#[test]
fn test_rank_second_entry() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    assert_eq!(csr.rank(&(0, 2)), 1);
}

#[test]
fn test_rank_third_entry() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    assert_eq!(csr.rank(&(1, 0)), 2);
}

// ============================================================================
// RankSelectSparseMatrix::select tests
// ============================================================================

#[test]
fn test_select_first() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    assert_eq!(csr.select(0), (0, 1));
}

#[test]
fn test_select_second() {
    // Each row has exactly one entry so select_row behaves predictably
    let csr = build_csr(vec![(0, 1), (1, 0), (2, 2)], 3, 3);
    assert_eq!(csr.select(0), (0, 1));
    assert_eq!(csr.select(1), (1, 0));
    assert_eq!(csr.select(2), (2, 2));
}

#[test]
fn test_select_third() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);
    assert_eq!(csr.select(2), (1, 0));
}

// ============================================================================
// Rank-select roundtrip
// ============================================================================

#[test]
fn test_rank_select_roundtrip() {
    // One entry per row to avoid select_row ambiguity
    let csr = build_csr(vec![(0, 0), (1, 1), (2, 2)], 3, 3);

    for i in 0..csr.number_of_defined_values() {
        let coords = csr.select(i);
        let rank = csr.rank(&coords);
        assert_eq!(rank, i, "roundtrip failed for sparse_index {i}");
    }
}

#[test]
fn test_rank_of_each_entry() {
    // Test rank for known entries
    let csr = build_csr(vec![(0, 0), (0, 1), (1, 0), (1, 1)], 2, 2);

    assert_eq!(csr.rank(&(0, 0)), 0);
    assert_eq!(csr.rank(&(0, 1)), 1);
    assert_eq!(csr.rank(&(1, 0)), 2);
    assert_eq!(csr.rank(&(1, 1)), 3);
}

// ============================================================================
// SizedSparseMatrix2D::rank_row tests
// ============================================================================

#[test]
fn test_rank_row() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0), (2, 1)], 3, 3);

    // Row 0 starts at sparse index 0
    assert_eq!(csr.rank_row(0), 0);
    // Row 1 starts at sparse index 2 (after 2 entries in row 0)
    assert_eq!(csr.rank_row(1), 2);
    // Row 2 starts at sparse index 3 (after 2 + 1 entries)
    assert_eq!(csr.rank_row(2), 3);
}

#[test]
fn test_rank_row_empty_row() {
    let csr = build_csr(vec![(0, 0), (2, 1)], 3, 3);

    assert_eq!(csr.rank_row(0), 0);
    // Row 1 is empty, but its rank is 1 (same as row 2's start)
    assert_eq!(csr.rank_row(1), 1);
    assert_eq!(csr.rank_row(2), 1);
}

// ============================================================================
// SizedSparseMatrix2D::select_row tests
// ============================================================================

#[test]
fn test_select_row() {
    // One entry per row: offsets = [0, 1, 2] for 3 entries
    let csr = build_csr(vec![(0, 1), (1, 0), (2, 2)], 3, 3);

    assert_eq!(csr.select_row(0), 0); // sparse index 0 is in row 0
    assert_eq!(csr.select_row(1), 1); // sparse index 1 is in row 1
    assert_eq!(csr.select_row(2), 2); // sparse index 2 is in row 2
}

#[test]
fn test_select_row_via_select() {
    // Verify select_row is consistent with select
    let csr = build_csr(vec![(0, 1), (1, 0), (2, 2)], 3, 3);

    for i in 0..csr.number_of_defined_values() {
        let (row, _col) = csr.select(i);
        assert_eq!(csr.select_row(i), row);
    }
}

// ============================================================================
// SizedSparseMatrix2D::select_column tests
// ============================================================================

#[test]
fn test_select_column() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0), (2, 1)], 3, 3);

    assert_eq!(csr.select_column(0), 1); // sparse index 0 -> column 1
    assert_eq!(csr.select_column(1), 2); // sparse index 1 -> column 2
    assert_eq!(csr.select_column(2), 0); // sparse index 2 -> column 0
    assert_eq!(csr.select_column(3), 1); // sparse index 3 -> column 1
}

// ============================================================================
// sparse_columns and sparse_rows iterators
// ============================================================================

#[test]
fn test_sparse_columns_iterator() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);

    let cols: Vec<usize> = csr.sparse_columns().collect();
    assert_eq!(cols, vec![1, 2, 0]);
}

#[test]
fn test_sparse_rows_iterator() {
    let csr = build_csr(vec![(0, 1), (0, 2), (1, 0)], 2, 3);

    let rows: Vec<usize> = csr.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

#[test]
fn test_sparse_columns_empty() {
    let csr = build_csr(vec![], 2, 3);

    let cols: Vec<usize> = csr.sparse_columns().collect();
    assert!(cols.is_empty());
}

#[test]
fn test_sparse_rows_empty() {
    let csr = build_csr(vec![], 2, 3);

    let rows: Vec<usize> = csr.sparse_rows().collect();
    assert!(rows.is_empty());
}

// ============================================================================
// Reference impls for RankSelectSparseMatrix
// ============================================================================

#[test]
fn test_rank_via_reference() {
    let csr = build_csr(vec![(0, 1), (1, 0)], 2, 2);
    let csr_ref: &TestCSR = &csr;

    assert_eq!(csr_ref.rank(&(0, 1)), 0);
    assert_eq!(csr_ref.rank(&(1, 0)), 1);
}

#[test]
fn test_select_via_reference() {
    let csr = build_csr(vec![(0, 1), (1, 0)], 2, 2);
    let csr_ref: &TestCSR = &csr;

    assert_eq!(csr_ref.select(0), (0, 1));
    assert_eq!(csr_ref.select(1), (1, 0));
}

// ============================================================================
// SparseMatrix reference impls
// ============================================================================

#[test]
fn test_sparse_matrix_ref_impls() {
    let csr = build_csr(vec![(0, 1), (1, 0)], 2, 2);
    let csr_ref: &TestCSR = &csr;

    // SparseMatrix methods via reference
    assert_eq!(csr_ref.last_sparse_coordinates(), Some((1, 0)));
    assert!(!csr_ref.is_empty());

    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(csr_ref).collect();
    assert_eq!(coords, vec![(0, 1), (1, 0)]);

    // SizedSparseMatrix via reference
    assert_eq!(csr_ref.number_of_defined_values(), 2);
}

// ============================================================================
// Matrix trait reference impls
// ============================================================================

#[test]
fn test_matrix_dimensions_via_reference() {
    let csr = build_csr(vec![(0, 1)], 2, 3);
    let csr_ref: &TestCSR = &csr;

    assert_eq!(<&TestCSR>::dimensions(), 2);
    assert_eq!(csr_ref.shape(), vec![2, 3]);
    assert_eq!(csr_ref.total_values(), 6);
}
