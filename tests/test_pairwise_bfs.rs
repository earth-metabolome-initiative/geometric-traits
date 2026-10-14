//! Tests for the PairwiseBFS all-pairs shortest-path algorithm.
#![cfg(feature = "std")]

mod common;

use common::build_square_csr;
use geometric_traits::{
    impls::{GenericImplicitValuedMatrix2D, VecMatrix2D},
    prelude::*,
    traits::DenseValuedMatrix,
};

fn assert_distance(
    distances: &VecMatrix2D<Option<usize>>,
    source_id: usize,
    destination_id: usize,
    expected: Option<usize>,
) {
    assert_eq!(
        distances.value((source_id, destination_id)),
        expected,
        "distance mismatch at ({source_id}, {destination_id})"
    );
}

#[test]
fn test_empty_graph() {
    let csr = build_square_csr(0, vec![]);
    let distances = csr.pairwise_bfs();

    assert_eq!(distances.shape(), vec![0, 0]);
}

#[test]
fn test_single_node() {
    let csr = build_square_csr(1, vec![]);
    let distances = csr.pairwise_bfs();

    assert_eq!(distances.value((0, 0)), Some(0));
}

#[test]
fn test_directed_path_with_unreachable_pairs() {
    let csr = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);
    let distances = csr.pairwise_bfs();

    assert_distance(&distances, 0, 0, Some(0));
    assert_distance(&distances, 0, 1, Some(1));
    assert_distance(&distances, 0, 2, Some(2));
    assert_distance(&distances, 0, 3, Some(3));
    assert_distance(&distances, 3, 0, None);
    assert_distance(&distances, 3, 3, Some(0));
}

#[test]
fn test_directed_cycle() {
    let csr = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);
    let distances = csr.pairwise_bfs();

    assert_distance(&distances, 0, 2, Some(2));
    assert_distance(&distances, 2, 1, Some(2));
    assert_distance(&distances, 1, 1, Some(0));
}

#[test]
fn test_self_loop_does_not_override_zero_diagonal() {
    let csr = build_square_csr(2, vec![(0, 0), (0, 1), (1, 0)]);
    let distances = csr.pairwise_bfs();

    assert_distance(&distances, 0, 0, Some(0));
    assert_distance(&distances, 1, 1, Some(0));
}

#[test]
fn test_matches_unit_floyd_warshall() {
    let csr = build_square_csr(5, vec![(0, 1), (0, 3), (1, 2), (3, 4), (4, 2)]);
    let pairwise_bfs = csr.pairwise_bfs();
    let floyd_warshall = GenericImplicitValuedMatrix2D::new(csr.clone(), |_| 1usize)
        .floyd_warshall()
        .expect("unit-weight Floyd-Warshall should succeed");

    assert_eq!(pairwise_bfs, floyd_warshall);
}
