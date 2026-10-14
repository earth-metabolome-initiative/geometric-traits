//! Tests for the PairwiseDijkstra all-pairs shortest-path algorithm.
#![cfg(feature = "std")]

#[cfg(feature = "arbitrary")]
use geometric_traits::test_utils::{check_pairwise_dijkstra_matches_floyd_warshall, from_bytes};
use geometric_traits::{
    impls::{ValuedCSR2D, VecMatrix2D},
    prelude::*,
    traits::{DenseValuedMatrix, EdgesBuilder},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;
type TestIntValCSR = ValuedCSR2D<usize, usize, usize, i32>;

fn build_matrix(
    rows: usize,
    columns: usize,
    edges: impl IntoIterator<Item = (usize, usize, f64)>,
) -> TestValCSR {
    let mut edges: Vec<(usize, usize, f64)> = edges.into_iter().collect();
    edges.sort_unstable_by_key(|left| (left.0, left.1));
    GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, columns))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn build_int_matrix(
    rows: usize,
    columns: usize,
    edges: impl IntoIterator<Item = (usize, usize, i32)>,
) -> TestIntValCSR {
    let mut edges: Vec<(usize, usize, i32)> = edges.into_iter().collect();
    edges.sort_unstable_by_key(|left| (left.0, left.1));
    GenericEdgesBuilder::<_, TestIntValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, columns))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn assert_distance(
    distances: &VecMatrix2D<Option<f64>>,
    source_id: usize,
    destination_id: usize,
    expected: Option<f64>,
) {
    let actual = distances.value((source_id, destination_id));
    match (actual, expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            let tolerance = expected.abs().max(1.0) * 1e-9;
            assert!(
                (actual - expected).abs() <= tolerance,
                "distance mismatch at ({source_id}, {destination_id}): expected {expected}, got {actual}"
            );
        }
        _ => {
            panic!(
                "distance mismatch at ({source_id}, {destination_id}): expected {expected:?}, got {actual:?}"
            );
        }
    }
}

#[test]
fn test_empty_matrix() {
    let csr = build_matrix(0, 0, core::iter::empty());
    let distances = csr.pairwise_dijkstra().unwrap();
    assert_eq!(distances.shape(), vec![0, 0]);
}

#[test]
fn test_single_node() {
    let csr = build_matrix(1, 1, core::iter::empty());
    let distances = csr.pairwise_dijkstra().unwrap();
    assert_eq!(distances.value((0, 0)), Some(0.0));
}

#[test]
fn test_directed_path_with_unreachable_pairs() {
    let csr = build_matrix(4, 4, [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0)]);
    let distances = csr.pairwise_dijkstra().unwrap();

    assert_distance(&distances, 0, 0, Some(0.0));
    assert_distance(&distances, 0, 1, Some(2.0));
    assert_distance(&distances, 0, 2, Some(5.0));
    assert_distance(&distances, 0, 3, Some(9.0));
    assert_distance(&distances, 3, 0, None);
    assert_distance(&distances, 3, 3, Some(0.0));
}

#[test]
fn test_prefers_shorter_indirect_path() {
    let csr = build_matrix(4, 4, [(0, 1, 2.0), (0, 2, 8.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let distances = csr.pairwise_dijkstra().unwrap();

    assert_distance(&distances, 0, 2, Some(3.0));
    assert_distance(&distances, 0, 3, Some(4.0));
}

#[test]
fn test_positive_self_loop_does_not_override_zero_diagonal() {
    let csr = build_matrix(2, 2, [(0, 0, 3.0), (0, 1, 1.0), (1, 0, 2.0)]);
    let distances = csr.pairwise_dijkstra().unwrap();

    assert_distance(&distances, 0, 0, Some(0.0));
    assert_distance(&distances, 1, 1, Some(0.0));
}

#[test]
fn test_non_square_error() {
    let csr = build_matrix(2, 3, [(0, 1, 1.0)]);
    let result = csr.pairwise_dijkstra();

    assert!(matches!(result, Err(PairwiseDijkstraError::NonSquareMatrix { rows: 2, columns: 3 })));
}

#[test]
fn test_non_finite_weight_error() {
    let csr = build_matrix(2, 2, [(0, 1, f64::INFINITY)]);
    let result = csr.pairwise_dijkstra();

    assert!(matches!(
        result,
        Err(PairwiseDijkstraError::NonFiniteWeight { source_id: 0, destination_id: 1 })
    ));
}

#[test]
fn test_negative_weight_error() {
    let csr = build_matrix(3, 3, [(0, 1, 1.0), (1, 2, -2.0)]);
    let result = csr.pairwise_dijkstra();

    assert!(matches!(
        result,
        Err(PairwiseDijkstraError::NegativeWeight { source_id: 1, destination_id: 2 })
    ));
}

#[test]
fn test_non_finite_distance_error() {
    let csr = build_matrix(3, 3, [(0, 1, 1e308), (1, 2, 1e308)]);
    let result = csr.pairwise_dijkstra();

    assert!(matches!(
        result,
        Err(PairwiseDijkstraError::NonFiniteDistance {
            source_id: 0,
            destination_id: 2,
            via_id: 1,
        })
    ));
}

#[test]
fn test_generic_integer_weights() {
    let csr = build_int_matrix(3, 3, [(0, 1, 4), (1, 2, 5), (0, 2, 20)]);
    let distances = csr.pairwise_dijkstra().unwrap();

    assert_eq!(distances.value((0, 2)), Some(9));
    assert_eq!(distances.value((2, 0)), None);
}

#[test]
fn test_matches_floyd_warshall_on_non_negative_weights() {
    let csr =
        build_matrix(5, 5, [(0, 1, 2.0), (0, 4, 15.0), (1, 2, 3.0), (2, 3, 1.5), (3, 4, 2.5)]);
    let pairwise_dijkstra = csr.pairwise_dijkstra().unwrap();
    let floyd_warshall = csr.floyd_warshall().unwrap();

    assert_eq!(pairwise_dijkstra, floyd_warshall);
}

#[cfg(feature = "arbitrary")]
#[test]
fn test_honggfuzz_crash_repro() {
    let bytes = [
        0x34, 0x31, 0x01, 0x00, 0xbf, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbf, 0xbf, 0xbf,
        0xbf, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf,
        0xbf, 0xbf, 0xbf, 0xe4, 0xe4, 0xe4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8a,
        0x32, 0x7f, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e, 0x4e,
        0x4e, 0x4e, 0x00, 0x00, 0xb3, 0x00, 0x00, 0x00, 0x00,
    ];
    let csr = from_bytes::<ValuedCSR2D<u16, u8, u8, f64>>(&bytes)
        .expect("honggfuzz crash bytes should decode into a ValuedCSR2D");
    check_pairwise_dijkstra_matches_floyd_warshall(&csr);
}
