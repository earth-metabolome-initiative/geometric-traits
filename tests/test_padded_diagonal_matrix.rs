//! Tests for GenericMatrix2DWithPaddedDiagonal.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, MutabilityError, ValuedCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, EmptyRows, Matrix, Matrix2D, SparseMatrix, SparseMatrix2D,
        SparseMatrixMut, SparseValuedMatrix, SparseValuedMatrix2D,
    },
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

/// Helper to build a ValuedCSR2D from triples.
fn build_valued_csr(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> ValuedCSR2D<usize, usize, usize, f64> {
    GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Construction tests
// ============================================================================

#[test]
fn test_padded_diagonal_construction() {
    let inner = build_valued_csr(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let padded =
        GenericMatrix2DWithPaddedDiagonal::new(inner, |row: usize| usize_to_f64(row) * 10.0);
    assert!(padded.is_ok());
}

#[test]
fn test_padded_diagonal_square_padding() {
    // Rectangular 2x3 matrix should become 3x3
    let inner = build_valued_csr(vec![(0, 1, 1.0), (1, 2, 2.0)], 2, 3);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);
}

#[test]
fn test_padded_diagonal_shape() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 3);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    assert_eq!(padded.shape(), vec![3, 3]);
}

#[test]
fn test_padded_diagonal_capacity_validation_allows_asymmetric_indices_when_square_fits() {
    let inner: ValuedCSR2D<usize, u16, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10u16, 200u8), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: u16| 0.0);
    assert!(padded.is_ok(), "Square size 200 fits both u16 rows and u8 columns");
}

#[test]
fn test_padded_diagonal_capacity_rejects_column_count_too_large_for_row_index() {
    let inner: ValuedCSR2D<usize, u8, u16, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10u8, 256u16), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: u8| 0.0);
    assert!(matches!(padded, Err(MutabilityError::MaxedOutColumnIndex)));
}

#[test]
fn test_padded_diagonal_capacity_rejects_row_count_too_large_for_column_index() {
    let inner: ValuedCSR2D<usize, u16, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((256u16, 10u8), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: u16| 0.0);
    assert!(matches!(padded, Err(MutabilityError::MaxedOutRowIndex)));
}

// ============================================================================
// Diagonal imputation tests
// ============================================================================

#[test]
fn test_is_diagonal_imputed_missing() {
    // Matrix with no diagonal entries
    let inner = build_valued_csr(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // Diagonal (0,0) is not in the original matrix, so it should be imputed
    assert!(padded.is_diagonal_imputed(0));
    // Diagonal (1,1) is not in the original matrix either
    assert!(padded.is_diagonal_imputed(1));
}

#[test]
fn test_is_diagonal_imputed_present() {
    // Matrix with a diagonal entry at (0,0)
    let inner = build_valued_csr(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // (0,0) is present in original
    assert!(!padded.is_diagonal_imputed(0));
    // (1,1) is missing
    assert!(padded.is_diagonal_imputed(1));
}

#[test]
fn test_is_diagonal_imputed_out_of_bounds() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // Row beyond original matrix bounds is imputed
    assert!(padded.is_diagonal_imputed(5));
}

// ============================================================================
// Sparse row with padding tests
// ============================================================================

#[test]
fn test_sparse_row_includes_diagonal() {
    // 2x2 matrix with only off-diagonal entries
    let inner = build_valued_csr(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // Row 0 should contain both column 0 (padded diagonal) and column 1 (original)
    let row0: Vec<usize> = padded.sparse_row(0).collect();
    assert!(row0.contains(&0), "Diagonal column 0 should be in row 0");
    assert!(row0.contains(&1), "Original column 1 should be in row 0");

    // Row 1 should contain column 0 (original) and column 1 (padded diagonal)
    let row1: Vec<usize> = padded.sparse_row(1).collect();
    assert!(row1.contains(&0), "Original column 0 should be in row 1");
    assert!(row1.contains(&1), "Diagonal column 1 should be in row 1");
}

#[test]
fn test_sparse_row_no_duplicate_diagonal() {
    // Matrix that already has diagonal at (0,0)
    let inner = build_valued_csr(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    let row0: Vec<usize> = padded.sparse_row(0).collect();
    // Should not have duplicate column 0
    let count_zeros = row0.iter().filter(|&&c| c == 0).count();
    assert_eq!(count_zeros, 1, "Diagonal should not be duplicated");
}

// ============================================================================
// has_entry tests
// ============================================================================

#[test]
fn test_has_entry_includes_padded_diagonal() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // Padded diagonal entries should exist
    assert!(padded.has_entry(0, 0));
    assert!(padded.has_entry(1, 1));

    // Original entry should still exist
    assert!(padded.has_entry(0, 1));
}

// ============================================================================
// EmptyRows tests
// ============================================================================

#[test]
fn test_no_empty_rows_after_padding() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 3, 3);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // After padding, every row has at least its diagonal entry
    assert_eq!(padded.number_of_empty_rows(), 0);
    assert_eq!(padded.number_of_non_empty_rows(), 3);

    let empty_rows: Vec<usize> = padded.empty_row_indices().collect();
    assert!(empty_rows.is_empty());

    let non_empty: Vec<usize> = padded.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 1, 2]);
}

// ============================================================================
// is_empty tests
// ============================================================================

#[test]
fn test_not_empty_with_padding() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    assert!(!padded.is_empty());
}

// ============================================================================
// last_sparse_coordinates tests
// ============================================================================

#[test]
fn test_last_sparse_coordinates() {
    let inner = build_valued_csr(vec![(0, 0, 1.0)], 3, 3);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    let last = padded.last_sparse_coordinates();
    assert_eq!(last, Some((2, 2)));
}

// ============================================================================
// sparse_row_values tests
// ============================================================================

#[test]
fn test_sparse_row_values_with_imputed_diagonal() {
    let inner = build_valued_csr(vec![(0, 1, 1.0), (1, 0, 2.0)], 2, 2);
    let padded =
        GenericMatrix2DWithPaddedDiagonal::new(inner, |row: usize| usize_to_f64(row + 1) * 100.0)
            .unwrap();

    // Row 0: diagonal imputed with map(0) = 100.0, plus original (0,1) = 1.0
    let row0_values: Vec<f64> = padded.sparse_row_values(0).collect();
    assert_eq!(row0_values.len(), 2);
    assert!(row0_values.contains(&100.0));
    assert!(row0_values.contains(&1.0));

    // Row 1: original (1,0) = 2.0, diagonal imputed with map(1) = 200.0
    let row1_values: Vec<f64> = padded.sparse_row_values(1).collect();
    assert_eq!(row1_values.len(), 2);
    assert!(row1_values.contains(&2.0));
    assert!(row1_values.contains(&200.0));
}

#[test]
fn test_sparse_row_values_no_imputation_needed() {
    // Diagonal already present
    let inner = build_valued_csr(vec![(0, 0, 5.0), (0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 999.0).unwrap();

    let row0_values: Vec<f64> = padded.sparse_row_values(0).collect();
    // Should use the original value 5.0, NOT the map value 999.0
    assert!(row0_values.contains(&5.0));
    assert!(row0_values.contains(&1.0));
    assert!(!row0_values.contains(&999.0));
}

// ============================================================================
// sparse_values (all values) tests
// ============================================================================

#[test]
fn test_sparse_values_includes_imputed() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    let all_values: Vec<f64> = padded.sparse_values().collect();
    // Should have original entry + 2 imputed diagonal entries
    assert_eq!(all_values.len(), 3);
}

// ============================================================================
// matrix() accessor test
// ============================================================================

#[test]
fn test_matrix_accessor() {
    let inner = build_valued_csr(vec![(0, 1, 1.0)], 2, 2);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: usize| 0.0).unwrap();

    // The inner matrix should maintain its original properties
    assert_eq!(padded.matrix().number_of_rows(), 2);
    assert_eq!(padded.matrix().number_of_columns(), 2);
    assert_eq!(padded.matrix().number_of_defined_values(), 1);
}
