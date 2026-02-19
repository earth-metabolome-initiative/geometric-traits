//! Tests for blanket trait impls on references (&M, &V).
//! Uses UFCS to force calling the blanket impl rather than
//! auto-deref to the underlying type's impl.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    traits::{
        BidirectionalVocabulary, Matrix, MatrixMut, RankSelectSparseMatrix, SizedSparseMatrix,
        SparseMatrix, SparseMatrixMut, Vocabulary,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;

fn build_csr(entries: Vec<(usize, usize)>, shape: (usize, usize)) -> TestCSR {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut csr, (r, c)).unwrap();
    }
    csr
}

// ============================================================================
// Matrix for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_matrix_shape_ufcs() {
    let csr = build_csr(vec![], (3, 4));
    let r = &csr;
    assert_eq!(<&TestCSR as Matrix>::shape(&r), vec![3, 4]);
}

#[test]
fn test_ref_matrix_total_values_ufcs() {
    let csr = build_csr(vec![], (3, 4));
    let r = &csr;
    assert_eq!(<&TestCSR as Matrix>::total_values(&r), 12);
}

// ============================================================================
// SparseMatrix for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_sparse_coordinates_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 1)], (2, 2));
    let r = &csr;
    let coords: Vec<(usize, usize)> = <&TestCSR as SparseMatrix>::sparse_coordinates(&r).collect();
    assert_eq!(coords, vec![(0, 0), (1, 1)]);
}

#[test]
fn test_ref_last_sparse_coordinates_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 1)], (2, 2));
    let r = &csr;
    assert_eq!(<&TestCSR as SparseMatrix>::last_sparse_coordinates(&r), Some((1, 1)));
}

#[test]
fn test_ref_is_empty_ufcs() {
    let csr = build_csr(vec![], (2, 2));
    let r = &csr;
    assert!(<&TestCSR as SparseMatrix>::is_empty(&r));
}

// ============================================================================
// SizedSparseMatrix for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_number_of_defined_values_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 1)], (2, 2));
    let r = &csr;
    assert_eq!(<&TestCSR as SizedSparseMatrix>::number_of_defined_values(&r), 2);
}

// ============================================================================
// RankSelectSparseMatrix for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_rank_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 1)], (2, 2));
    let r = &csr;
    assert_eq!(<&TestCSR as RankSelectSparseMatrix>::rank(&r, &(0, 0)), 0);
    assert_eq!(<&TestCSR as RankSelectSparseMatrix>::rank(&r, &(1, 1)), 1);
}

#[test]
fn test_ref_select_ufcs() {
    let csr = build_csr(vec![(0, 0), (1, 1)], (2, 2));
    let r = &csr;
    assert_eq!(<&TestCSR as RankSelectSparseMatrix>::select(&r, 0), (0, 0));
    assert_eq!(<&TestCSR as RankSelectSparseMatrix>::select(&r, 1), (1, 1));
}

// ============================================================================
// SparseValuedMatrix for &M (UFCS)
// ============================================================================

#[test]
fn test_ref_sparse_values_ufcs() {
    use geometric_traits::{
        impls::ValuedCSR2D,
        prelude::*,
        traits::{EdgesBuilder, SparseValuedMatrix},
    };
    type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;
    let vcsr: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(2)
        .expected_shape((2, 2))
        .edges(vec![(0, 0, 1.0), (1, 1, 2.0)].into_iter())
        .build()
        .unwrap();
    let r = &vcsr;
    let vals: Vec<f64> = <&TestValCSR as SparseValuedMatrix>::sparse_values(&r).collect();
    assert_eq!(vals, vec![1.0, 2.0]);
}

// ============================================================================
// Vocabulary for &V (UFCS)
// ============================================================================

#[test]
fn test_ref_vocabulary_convert_ufcs() {
    let vocab: Vec<&str> = vec!["a", "b", "c"];
    let r = &vocab;
    assert_eq!(<&Vec<&str> as Vocabulary>::convert(&r, &0), Some("a"));
    assert_eq!(<&Vec<&str> as Vocabulary>::len(&r), 3);
}

#[test]
fn test_ref_vocabulary_sources_ufcs() {
    let vocab: Vec<&str> = vec!["a", "b"];
    let r = &vocab;
    let sources: Vec<usize> = <&Vec<&str> as Vocabulary>::sources(&r).collect();
    assert_eq!(sources, vec![0, 1]);
}

#[test]
fn test_ref_vocabulary_destinations_ufcs() {
    let vocab: Vec<&str> = vec!["x", "y"];
    let r = &vocab;
    let destinations: Vec<&str> = <&Vec<&str> as Vocabulary>::destinations(&r).collect();
    assert_eq!(destinations, vec!["x", "y"]);
}

// ============================================================================
// BidirectionalVocabulary for &V (UFCS)
// ============================================================================

#[test]
fn test_ref_bidirectional_vocabulary_ufcs() {
    let vocab: usize = 5;
    let r = &vocab;
    assert_eq!(<&usize as BidirectionalVocabulary>::invert(&r, &3), Some(3_usize));
}
