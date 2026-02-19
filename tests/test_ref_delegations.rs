//! Tests for trait delegation impls on references (&M).
//! Covers: Matrix for &M, SparseMatrix for &M, SizedSparseMatrix for &M,
//! RankSelectSparseMatrix for &M, SparseValuedMatrix for &M,
//! Vocabulary for &V, VocabularyRef for &V,
//! BidirectionalVocabulary for &V, BidirectionalVocabularyRef for &V.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    traits::{
        BidirectionalVocabulary, Matrix, MatrixMut, RankSelectSparseMatrix, SizedSparseMatrix,
        SparseMatrix, SparseMatrixMut, Vocabulary,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

// ============================================================================
// Matrix for &M
// ============================================================================

#[test]
fn test_ref_matrix_shape() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 4));
    let r = &csr;
    assert_eq!(r.shape(), vec![3, 4]);
}

#[test]
fn test_ref_matrix_total_values() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 4));
    let r = &csr;
    assert_eq!(r.total_values(), 12);
}

// ============================================================================
// SparseMatrix for &M
// ============================================================================

#[test]
fn test_ref_sparse_matrix_sparse_coordinates() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    let r = &csr;
    let coords: Vec<(usize, usize)> = r.sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 0), (1, 1)]);
}

#[test]
fn test_ref_sparse_matrix_last_sparse_coordinates() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    let r = &csr;
    assert_eq!(r.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_ref_sparse_matrix_is_empty() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    let r = &csr;
    assert!(r.is_empty());
}

// ============================================================================
// SizedSparseMatrix for &M
// ============================================================================

#[test]
fn test_ref_sized_sparse_matrix_number_of_defined_values() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    let r = &csr;
    assert_eq!(r.number_of_defined_values(), 2);
}

// ============================================================================
// RankSelectSparseMatrix for &M
// ============================================================================

#[test]
fn test_ref_rank_select_sparse_matrix() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((3, 3));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    let r = &csr;
    assert_eq!(r.rank(&(0, 0)), 0);
    assert_eq!(r.rank(&(1, 1)), 1);
    assert_eq!(r.select(0), (0, 0));
    assert_eq!(r.select(1), (1, 1));
}

// ============================================================================
// SparseValuedMatrix for &M
// ============================================================================

#[test]
fn test_ref_sparse_valued_matrix() {
    use geometric_traits::prelude::*;
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(2)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (1, 1, 2.0)].into_iter())
        .build()
        .unwrap();
    let r = &vcsr;
    let values: Vec<f64> = r.sparse_values().collect();
    assert_eq!(values, vec![1.0, 2.0]);
}

// ============================================================================
// Vocabulary for &V (using Vec<T> which implements Vocabulary)
// ============================================================================

#[test]
fn test_ref_vocabulary_convert() {
    let vocab: Vec<&str> = vec!["apple", "banana", "cherry"];
    let r = &vocab;
    assert_eq!(Vocabulary::convert(r, &0), Some("apple"));
    assert_eq!(Vocabulary::convert(r, &1), Some("banana"));
    assert_eq!(Vocabulary::convert(r, &3), None);
}

#[test]
fn test_ref_vocabulary_len() {
    let vocab: Vec<&str> = vec!["apple", "banana"];
    let r = &vocab;
    assert_eq!(Vocabulary::len(r), 2);
}

#[test]
fn test_ref_vocabulary_sources() {
    let vocab: Vec<&str> = vec!["a", "b", "c"];
    let r = &vocab;
    let sources: Vec<usize> = Vocabulary::sources(r).collect();
    assert_eq!(sources, vec![0, 1, 2]);
}

#[test]
fn test_ref_vocabulary_destinations() {
    let vocab: Vec<&str> = vec!["a", "b"];
    let r = &vocab;
    let destinations: Vec<&str> = Vocabulary::destinations(r).collect();
    assert_eq!(destinations, vec!["a", "b"]);
}

// ============================================================================
// BidirectionalVocabulary for &V (using usize which implements it)
// ============================================================================

#[test]
fn test_ref_bidirectional_vocabulary() {
    let vocab: usize = 5;
    let r = &vocab;
    // usize implements BidirectionalVocabulary where invert returns the same value
    assert_eq!(BidirectionalVocabulary::invert(r, &3), Some(3_usize));
}

// ============================================================================
// SizedSparseMatrix density
// ============================================================================

#[test]
fn test_density() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    // 2 defined values out of 4 total = 0.5
    assert!((csr.density() - 0.5).abs() < 1e-10);
}

// ============================================================================
// SparseValuedMatrix max/min
// ============================================================================

#[test]
fn test_max_sparse_value() {
    use geometric_traits::prelude::*;
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 3))
        .edges(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 2, 3.0)].into_iter())
        .build()
        .unwrap();
    assert_eq!(vcsr.max_sparse_value(), Some(5.0));
    assert_eq!(vcsr.min_sparse_value(), Some(1.0));
}
