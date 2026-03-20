//! Tests targeting specific tarpaulin coverage gaps.
//!
//! Many uncovered lines are tarpaulin measurement artifacts (e.g. `#[inline]`
//! identity returns, `debug_assert!` bodies, lender GAT adapters). This file
//! adds direct-call tests for the lines where a concrete exercising test is
//! feasible.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{
        CSR2D, GenericBiMatrix2D, SquareCSR2D, SubsetSquareMatrix, UpperTriangularCSR2D,
        VecMatrix2D,
    },
    traits::{
        Edges, Graph, LAPError, LAPJV, MatrixMut, MonoplexGraph, SparseMatrixMut, SquareMatrix,
        information_content::InformationContentError,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestUT = UpperTriangularCSR2D<TestCSR>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestBiMatrix = GenericBiMatrix2D<TestSquareCSR, TestSquareCSR>;

fn build_ut(entries: Vec<(usize, usize)>) -> TestUT {
    let mut ut: TestUT = UpperTriangularCSR2D::default();
    for (r, c) in entries {
        MatrixMut::add(&mut ut, (r, c)).unwrap();
    }
    ut
}

fn build_bimatrix(order: usize, entries: Vec<(usize, usize)>) -> TestBiMatrix {
    let mut m: TestSquareCSR = SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    m.extend(entries).unwrap();
    GenericBiMatrix2D::new(m)
}

fn build_square_matrix(
    order: usize,
    entries: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    let mut m: SquareCSR2D<CSR2D<usize, usize, usize>> =
        SquareCSR2D::with_sparse_shaped_capacity(order, entries.len());
    for e in entries {
        MatrixMut::add(&mut m, e).expect("add entry");
    }
    m
}

// ============================================================================
// UpperTriangularCSR2D: Edges::matrix(), MonoplexGraph::edges(),
// GrowableEdges::with_shape
// ============================================================================

#[test]
fn test_ut_edges_matrix() {
    let ut = build_ut(vec![(0, 1), (1, 2)]);
    let m = Edges::matrix(&ut);
    assert_eq!(m.order(), 3);
}

#[test]
fn test_ut_monoplex_edges() {
    let ut = build_ut(vec![(0, 1), (1, 2)]);
    let e = MonoplexGraph::edges(&ut);
    assert_eq!(e.order(), 3);
}

#[test]
fn test_ut_graph_has_nodes_and_edges() {
    let ut = build_ut(vec![(0, 1)]);
    assert!(Graph::has_nodes(&ut));
    assert!(Graph::has_edges(&ut));

    let empty: TestUT = UpperTriangularCSR2D::default();
    assert!(!Graph::has_nodes(&empty));
    assert!(!Graph::has_edges(&empty));
}

#[test]
fn test_ut_growable_edges_with_shape() {
    use geometric_traits::traits::GrowableEdges;
    let ut: TestUT = GrowableEdges::with_shape(5);
    assert_eq!(ut.order(), 5);
}

// ============================================================================
// GenericBiMatrix2D: Edges::matrix(), MonoplexGraph::edges()
// ============================================================================

#[test]
fn test_bimatrix_edges_matrix() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let m = Edges::matrix(&bm);
    assert_eq!(m.order(), 3);
}

#[test]
fn test_bimatrix_monoplex_edges() {
    let bm = build_bimatrix(3, vec![(0, 1), (1, 2)]);
    let e = MonoplexGraph::edges(&bm);
    assert_eq!(e.order(), 3);
}

#[test]
fn test_bimatrix_graph_has_nodes_and_edges() {
    let bm = build_bimatrix(3, vec![(0, 1)]);
    assert!(Graph::has_nodes(&bm));
    assert!(Graph::has_edges(&bm));

    let empty = build_bimatrix(3, vec![]);
    assert!(Graph::has_nodes(&empty));
    assert!(!Graph::has_edges(&empty));
}

// ============================================================================
// Dense LAPJV: non-square matrix error
// ============================================================================

#[test]
fn test_dense_lapjv_non_square_error() {
    let m = VecMatrix2D::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = m.lapjv(100.0);
    assert_eq!(result, Err(LAPError::NonSquareMatrix));
}

// ============================================================================
// InformationContentError: From<KahnError> direct test
// ============================================================================

#[test]
fn test_information_content_error_from_kahn_error() {
    use geometric_traits::traits::KahnError;
    let err: InformationContentError = KahnError::Cycle.into();
    assert_eq!(err, InformationContentError::NotDag);
}

// ============================================================================
// SubsetSquareMatrix: with_unsorted_indices out-of-bounds error path
// ============================================================================

#[test]
fn test_subset_unsorted_indices_out_of_bounds() {
    let matrix = build_square_matrix(3, vec![(0, 1), (1, 2)]);
    let result = SubsetSquareMatrix::with_unsorted_indices(matrix, vec![5].into_iter());
    assert!(result.is_err());
}
