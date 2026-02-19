//! Extended tests for SymmetricCSR2D: covers additional trait delegations
//! (SquareMatrix, SparseSquareMatrix, AsRef, Default, EmptyRows,
//! SizedRowsSparseMatrix2D, TransposableMatrix2D, Symmetrize, BiMatrix2D).
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{
        BiMatrix2D, EmptyRows, Matrix, Matrix2D, RankSelectSparseMatrix, SizedRowsSparseMatrix2D,
        SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, SparseSquareMatrix,
        SquareMatrix, Symmetrize, TransposableMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSym = SymmetricCSR2D<TestCSR>;

fn build_symmetric(edges: Vec<(usize, usize)>) -> TestSym {
    UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Default
// ============================================================================

#[test]
fn test_default() {
    let sym: TestSym = SymmetricCSR2D::default();
    assert!(sym.is_empty());
    assert_eq!(sym.number_of_rows(), 0);
    assert_eq!(sym.number_of_columns(), 0);
}

// ============================================================================
// Debug
// ============================================================================

#[test]
fn test_debug() {
    let sym = build_symmetric(vec![(0, 1)]);
    let debug = format!("{sym:?}");
    assert!(debug.contains("SymmetricCSR2D"));
}

// ============================================================================
// SquareMatrix
// ============================================================================

#[test]
fn test_order() {
    let sym = build_symmetric(vec![(1, 2), (2, 3)]);
    assert_eq!(sym.order(), 4);
}

// ============================================================================
// SparseSquareMatrix
// ============================================================================

#[test]
fn test_number_of_defined_diagonal_values_none() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    assert_eq!(sym.number_of_defined_diagonal_values(), 0);
}

#[test]
fn test_number_of_defined_diagonal_values_some() {
    let sym = build_symmetric(vec![(0, 1), (2, 2)]);
    assert_eq!(sym.number_of_defined_diagonal_values(), 1);
}

// ============================================================================
// Matrix::shape
// ============================================================================

#[test]
fn test_shape() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    assert_eq!(sym.shape(), vec![3, 3]);
}

// ============================================================================
// AsRef<SquareCSR2D>
// ============================================================================

#[test]
fn test_as_ref() {
    let sym = build_symmetric(vec![(0, 1)]);
    let inner = sym.as_ref();
    assert_eq!(inner.order(), sym.order());
}

// ============================================================================
// SparseMatrix delegation
// ============================================================================

#[test]
fn test_sparse_coordinates() {
    let sym = build_symmetric(vec![(0, 1)]);
    let coords: Vec<(usize, usize)> =
        geometric_traits::traits::SparseMatrix::sparse_coordinates(&sym).collect();
    assert!(coords.contains(&(0, 1)));
    assert!(coords.contains(&(1, 0)));
}

#[test]
fn test_is_empty() {
    let sym = build_symmetric(vec![(0, 1)]);
    assert!(!sym.is_empty());
}

#[test]
fn test_last_sparse_coordinates() {
    let sym = build_symmetric(vec![(0, 1)]);
    let last = sym.last_sparse_coordinates();
    assert!(last.is_some());
    assert_eq!(last.unwrap(), (1, 0));
}

// ============================================================================
// SizedSparseMatrix
// ============================================================================

#[test]
fn test_number_of_defined_values() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    assert_eq!(sym.number_of_defined_values(), 4);
}

// ============================================================================
// SparseMatrix2D delegation
// ============================================================================

#[test]
fn test_sparse_row() {
    let sym = build_symmetric(vec![(0, 1), (0, 2)]);
    let row0: Vec<usize> = sym.sparse_row(0).collect();
    assert_eq!(row0, vec![1, 2]);
}

#[test]
fn test_has_entry() {
    let sym = build_symmetric(vec![(0, 1)]);
    assert!(sym.has_entry(0, 1));
    assert!(sym.has_entry(1, 0));
    assert!(!sym.has_entry(0, 0));
}

// ============================================================================
// RankSelectSparseMatrix
// ============================================================================

#[test]
fn test_rank_select() {
    let sym = build_symmetric(vec![(0, 1)]);
    let rank = sym.rank(&(0, 1));
    let coords = sym.select(rank);
    assert_eq!(coords, (0, 1));
}

// ============================================================================
// SizedSparseMatrix2D
// ============================================================================

#[test]
fn test_rank_row() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    // Row 0 starts at rank 0
    assert_eq!(sym.rank_row(0), 0);
}

#[test]
fn test_select_column() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    assert_eq!(sym.select_column(0), 1); // first entry is (0,1), column=1
}

#[test]
fn test_select_row() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    assert_eq!(sym.select_row(0), 0); // first entry row=0
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_empty_row_indices() {
    let sym = build_symmetric(vec![(0, 2)]);
    let empty: Vec<usize> = sym.empty_row_indices().collect();
    assert_eq!(empty, vec![1]);
}

#[test]
fn test_non_empty_row_indices() {
    let sym = build_symmetric(vec![(0, 2)]);
    let non_empty: Vec<usize> = sym.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0, 2]);
}

#[test]
fn test_number_of_empty_rows() {
    let sym = build_symmetric(vec![(0, 2)]);
    assert_eq!(sym.number_of_empty_rows(), 1);
    assert_eq!(sym.number_of_non_empty_rows(), 2);
}

// ============================================================================
// SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_number_of_defined_values_in_row() {
    let sym = build_symmetric(vec![(0, 1), (0, 2)]);
    assert_eq!(sym.number_of_defined_values_in_row(0), 2);
}

#[test]
fn test_sparse_row_sizes() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    let sizes: Vec<usize> = sym.sparse_row_sizes().collect();
    // Row 0: [1], Row 1: [0, 2], Row 2: [1]
    assert_eq!(sizes, vec![1, 2, 1]);
}

// ============================================================================
// TransposableMatrix2D (symmetric = own transpose)
// ============================================================================

#[test]
fn test_transpose_is_clone() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    let transposed = sym.transpose();
    assert_eq!(transposed, sym);
}

// ============================================================================
// Symmetrize (symmetric = already symmetric)
// ============================================================================

#[test]
fn test_symmetrize_is_clone() {
    let sym = build_symmetric(vec![(0, 1), (1, 2)]);
    let symmetrized = sym.symmetrize();
    assert_eq!(symmetrized, sym);
}

// ============================================================================
// BiMatrix2D
// ============================================================================

#[test]
fn test_bimatrix2d_matrix() {
    let sym = build_symmetric(vec![(0, 1)]);
    let matrix = BiMatrix2D::matrix(&sym);
    assert_eq!(matrix.order(), sym.order());
}

#[test]
fn test_bimatrix2d_transposed() {
    let sym = build_symmetric(vec![(0, 1)]);
    let transposed_ref = BiMatrix2D::transposed(&sym);
    assert_eq!(transposed_ref.order(), sym.order());
}
