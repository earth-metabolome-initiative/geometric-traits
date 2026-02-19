//! Tests for SparseBiMatrix2D and SizedSparseBiMatrix2D through reference
//! wrappers, SquareMatrix reference impl, and SquareMatrix::order.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SquareCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, SizedSparseBiMatrix2D, SparseBiMatrix2D, SparseSquareMatrix, SquareMatrix,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestBiMatrix = GenericBiMatrix2D<TestCSR, TestCSR>;
type TestSquareBiMatrix = GenericBiMatrix2D<TestSquareCSR, TestSquareCSR>;

fn build_bimatrix(n: usize, edges: Vec<(usize, usize)>) -> TestBiMatrix {
    let inner: TestCSR = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericBiMatrix2D::new(inner)
}

fn build_square_bimatrix(n: usize, edges: Vec<(usize, usize)>) -> TestSquareBiMatrix {
    let inner: TestSquareCSR = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericBiMatrix2D::new(inner)
}

// ============================================================================
// SparseBiMatrix2D via reference
// ============================================================================

#[test]
fn test_sparse_column_via_reference() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 1), (2, 1)]);
    let bm_ref: &TestBiMatrix = &bm;

    let col1: Vec<usize> = bm_ref.sparse_column(1).collect();
    assert_eq!(col1, vec![0, 1, 2]);
}

// ============================================================================
// SizedSparseBiMatrix2D via reference
// ============================================================================

#[test]
fn test_number_of_defined_values_in_column_via_ref() {
    let bm = build_bimatrix(3, vec![(0, 2), (1, 2)]);
    let bm_ref: &TestBiMatrix = &bm;

    assert_eq!(bm_ref.number_of_defined_values_in_column(0), 0);
    assert_eq!(bm_ref.number_of_defined_values_in_column(2), 2);
}

#[test]
fn test_sparse_column_sizes_via_ref() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2), (2, 2)]);
    let bm_ref: &TestBiMatrix = &bm;

    let sizes: Vec<usize> = bm_ref.sparse_column_sizes().collect();
    assert_eq!(sizes, vec![0, 1, 2]);
}

// ============================================================================
// SquareMatrix::order tests
// ============================================================================

#[test]
fn test_square_matrix_order() {
    let sq = build_square_bimatrix(4, vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(sq.order(), 4);
}

#[test]
fn test_square_matrix_order_via_ref() {
    let sq = build_square_bimatrix(3, vec![(0, 1)]);
    let sq_ref: &TestSquareBiMatrix = &sq;
    assert_eq!(sq_ref.order(), 3);
}

// ============================================================================
// SparseSquareMatrix::number_of_defined_diagonal_values tests
// ============================================================================

#[test]
fn test_number_of_defined_diagonal_values() {
    let sq = build_square_bimatrix(3, vec![(0, 0), (0, 1), (1, 1), (2, 0)]);
    // Diagonal entries: (0,0) and (1,1)
    assert_eq!(sq.number_of_defined_diagonal_values(), 2);
}

#[test]
fn test_number_of_defined_diagonal_values_none() {
    let sq = build_square_bimatrix(3, vec![(0, 1), (1, 2), (2, 0)]);
    assert_eq!(sq.number_of_defined_diagonal_values(), 0);
}

#[test]
fn test_number_of_defined_diagonal_values_all() {
    let sq = build_square_bimatrix(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(sq.number_of_defined_diagonal_values(), 2);
}

#[test]
fn test_number_of_defined_diagonal_values_via_ref() {
    let sq = build_square_bimatrix(2, vec![(0, 0), (1, 1)]);
    let sq_ref: &TestSquareBiMatrix = &sq;
    assert_eq!(sq_ref.number_of_defined_diagonal_values(), 2);
}
