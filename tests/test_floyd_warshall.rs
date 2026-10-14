//! Tests for the Floyd-Warshall all-pairs shortest-path algorithm.
#![cfg(feature = "std")]

mod common;

use std::io::Read as _;

use flate2::read::GzDecoder;
use geometric_traits::{
    impls::{ValuedCSR2D, VecMatrix2D},
    prelude::*,
    traits::{DenseValuedMatrix, EdgesBuilder},
};
use serde::Deserialize;

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
            )
        }
    }
}

#[test]
fn test_empty_matrix() {
    let csr = build_matrix(0, 0, core::iter::empty());
    let distances = csr.floyd_warshall().unwrap();
    assert_eq!(distances.shape(), vec![0, 0]);
}

#[test]
fn test_single_node() {
    let csr = build_matrix(1, 1, core::iter::empty());
    let distances = csr.floyd_warshall().unwrap();
    assert_eq!(distances.value((0, 0)), Some(0.0));
}

#[test]
fn test_directed_path_with_unreachable_pairs() {
    let csr = build_matrix(4, 4, [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0)]);
    let distances = csr.floyd_warshall().unwrap();

    assert_distance(&distances, 0, 0, Some(0.0));
    assert_distance(&distances, 0, 1, Some(2.0));
    assert_distance(&distances, 0, 2, Some(5.0));
    assert_distance(&distances, 0, 3, Some(9.0));
    assert_distance(&distances, 3, 0, None);
    assert_distance(&distances, 3, 3, Some(0.0));
}

#[test]
fn test_prefers_shorter_indirect_path() {
    let csr = build_matrix(3, 3, [(0, 1, 5.0), (1, 2, 2.0), (0, 2, 10.0)]);
    let distances = csr.floyd_warshall().unwrap();

    assert_distance(&distances, 0, 2, Some(7.0));
    assert_distance(&distances, 2, 0, None);
}

#[test]
fn test_negative_edge_without_negative_cycle() {
    let csr = build_matrix(3, 3, [(0, 1, 1.0), (1, 2, -2.0), (0, 2, 5.0)]);
    let distances = csr.floyd_warshall().unwrap();

    assert_distance(&distances, 0, 2, Some(-1.0));
    assert_distance(&distances, 1, 1, Some(0.0));
}

#[test]
fn test_positive_self_loop_does_not_override_zero_diagonal() {
    let csr = build_matrix(2, 2, [(0, 0, 3.0), (0, 1, 1.0), (1, 0, 2.0)]);
    let distances = csr.floyd_warshall().unwrap();

    assert_distance(&distances, 0, 0, Some(0.0));
    assert_distance(&distances, 1, 1, Some(0.0));
}

#[test]
fn test_non_square_error() {
    let csr = build_matrix(2, 3, [(0, 1, 1.0)]);
    let result = csr.floyd_warshall();

    assert!(matches!(result, Err(FloydWarshallError::NonSquareMatrix { rows: 2, columns: 3 })));
}

#[test]
fn test_non_finite_weight_error() {
    let csr = build_matrix(2, 2, [(0, 1, f64::INFINITY)]);
    let result = csr.floyd_warshall();

    assert!(matches!(
        result,
        Err(FloydWarshallError::NonFiniteWeight { source_id: 0, destination_id: 1 })
    ));
}

#[test]
fn test_non_finite_distance_error() {
    let csr = build_matrix(3, 3, [(0, 1, 1e308), (1, 2, 1e308)]);
    let result = csr.floyd_warshall();

    assert!(matches!(
        result,
        Err(FloydWarshallError::NonFiniteDistance { source_id: 0, destination_id: 2, pivot_id: 1 })
    ));
}

#[test]
fn test_negative_self_loop_reports_negative_cycle() {
    let csr = build_matrix(2, 2, [(0, 0, -1.0)]);
    let result = csr.floyd_warshall();

    assert!(matches!(result, Err(FloydWarshallError::NegativeCycle { node_id: 0 })));
}

#[test]
fn test_negative_cycle_reports_error() {
    let csr = build_matrix(3, 3, [(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0)]);
    let result = csr.floyd_warshall();

    assert!(matches!(result, Err(FloydWarshallError::NegativeCycle { .. })));
}

#[test]
fn test_generic_integer_weights() {
    let csr = build_int_matrix(3, 3, [(0, 1, 4), (1, 2, 5), (0, 2, 20)]);
    let distances = csr.floyd_warshall().unwrap();

    assert_eq!(distances.value((0, 2)), Some(9));
    assert_eq!(distances.value((2, 0)), None);
}

// ============================================================================
// Ground-truth regression tests (NetworkX)
// ============================================================================

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    cases: Vec<GroundTruthCase>,
}

#[derive(Deserialize)]
struct GroundTruthCase {
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    distances: Vec<Option<f64>>,
}

fn load_fixture() -> Fixture {
    let ground_truth_gz = common::read_fixture("floyd_warshall_ground_truth.json.gz");
    let mut json = String::new();
    GzDecoder::new(ground_truth_gz.as_slice())
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json).expect("fixture JSON parse failed")
}

#[test]
fn test_ground_truth_metadata() {
    let fixture = load_fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.cases.len(), 2557);
}

#[test]
fn test_ground_truth_cases() {
    let fixture = load_fixture();

    for (case_index, case) in fixture.cases.iter().enumerate() {
        let csr = build_matrix(case.n, case.n, case.edges.iter().copied());
        let distances = csr.floyd_warshall().unwrap();

        assert_eq!(
            distances.shape(),
            vec![case.n, case.n],
            "case {case_index}: wrong output shape"
        );

        for source_id in 0..case.n {
            for destination_id in 0..case.n {
                let actual = distances.value((source_id, destination_id));
                let expected = case.distances[source_id * case.n + destination_id];
                match (actual, expected) {
                    (None, None) => {}
                    (Some(actual), Some(expected)) => {
                        let tolerance = expected.abs().max(1.0) * 1e-9;
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "case {case_index}: mismatch at ({source_id}, {destination_id}) with edges {:?}: expected {expected}, got {actual}",
                            case.edges
                        );
                    }
                    _ => {
                        panic!(
                            "case {case_index}: reachability mismatch at ({source_id}, {destination_id}) with edges {:?}: expected {expected:?}, got {actual:?}",
                            case.edges
                        );
                    }
                }
            }
        }
    }
}
