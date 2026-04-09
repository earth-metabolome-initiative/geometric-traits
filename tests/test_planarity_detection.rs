//! Integration tests for exact planarity detection on undirected graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, PlanarityDetection, PlanarityError, VocabularyBuilder},
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph, semantic_cases};

#[test]
fn test_planarity_matches_semantic_cases() {
    for case in semantic_cases() {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.is_planar().unwrap(),
            case.is_planar,
            "planarity mismatched fixture case {} ({})",
            case.name,
            case.family
        );
    }
}

#[test]
fn test_planarity_rejects_self_loops() {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols((0..3).enumerate())
        .build()
        .unwrap();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(2)
        .expected_shape(3)
        .edges([(0usize, 0usize), (0usize, 1usize)].into_iter())
        .build()
        .unwrap();
    let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));

    assert!(matches!(
        graph.is_planar(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::PlanarityError(
            PlanarityError::SelfLoopsUnsupported
        )))
    ));
}

fn regression_case(
    name: &str,
    family: &str,
    node_count: usize,
    edges: &[[usize; 2]],
    is_planar: bool,
) -> PlanarityFixtureCase {
    PlanarityFixtureCase {
        name: name.to_string(),
        family: family.to_string(),
        node_count,
        edges: edges.to_vec(),
        is_planar,
        is_outerplanar: false,
        planarity_obstruction_family: None,
        outerplanarity_obstruction_family: None,
        notes: "Regression case extracted from the Boyer local corpus.".to_string(),
    }
}

#[test]
fn test_planarity_accepts_outerplanar_cycle_chords_regression() {
    let case = regression_case(
        "outerplanar_cycle_chords_503906",
        "outerplanar_cycle_chords",
        6,
        &[[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_wheel_regression() {
    let case = regression_case(
        "wheel_546875",
        "wheel",
        7,
        &[
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 2],
            [1, 6],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_k4_subdivision_regression() {
    let case = regression_case(
        "k4_subdivision_813476",
        "k4_subdivision",
        9,
        &[[0, 1], [0, 6], [0, 8], [1, 2], [1, 3], [2, 3], [2, 4], [3, 7], [4, 5], [5, 6], [7, 8]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_random_planar_regression() {
    let case = regression_case(
        "erdos_renyi_554688",
        "erdos_renyi",
        12,
        &[
            [0, 8],
            [0, 11],
            [1, 2],
            [2, 3],
            [2, 5],
            [3, 6],
            [3, 7],
            [3, 8],
            [3, 10],
            [3, 11],
            [4, 6],
            [5, 10],
            [8, 9],
            [9, 11],
            [10, 11],
        ],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_rejects_k33_subdivision_corpus_regression() {
    let case = regression_case(
        "k33_subdivision_136470",
        "k33_subdivision",
        8,
        &[[0, 3], [0, 4], [0, 6], [1, 4], [1, 5], [1, 7], [2, 3], [2, 4], [2, 5], [3, 7], [5, 6]],
        false,
    );
    let graph = build_undigraph(&case);

    assert!(!graph.is_planar().unwrap(), "planarity should reject {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_erdos_renyi_corpus_regression() {
    let case = regression_case(
        "erdos_renyi_118000",
        "erdos_renyi",
        12,
        &[[0, 7], [0, 9], [1, 2], [1, 7], [1, 9], [2, 3], [2, 7], [2, 9], [3, 7], [4, 9], [8, 11]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_erdos_renyi_080328_corpus_regression() {
    let case = regression_case(
        "erdos_renyi_080328",
        "erdos_renyi",
        9,
        &[
            [0, 6],
            [0, 7],
            [1, 2],
            [1, 4],
            [2, 3],
            [2, 6],
            [2, 8],
            [3, 4],
            [3, 8],
            [4, 5],
            [5, 6],
            [5, 8],
        ],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_accepts_k4_subdivision_corpus_regression() {
    let case = regression_case(
        "k4_subdivision_287588",
        "k4_subdivision",
        9,
        &[[0, 2], [0, 3], [0, 7], [1, 5], [1, 7], [1, 8], [2, 5], [2, 6], [3, 4], [3, 8], [4, 6]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(graph.is_planar().unwrap(), "planarity should accept {} ({})", case.name, case.family);
}

#[test]
fn test_planarity_rejects_k5_subdivision_corpus_regression() {
    let case = regression_case(
        "k5_subdivision_110703",
        "k5_subdivision",
        13,
        &[
            [0, 1],
            [0, 3],
            [0, 4],
            [0, 12],
            [1, 3],
            [1, 7],
            [1, 9],
            [2, 8],
            [2, 10],
            [2, 11],
            [2, 12],
            [3, 5],
            [3, 8],
            [4, 5],
            [4, 6],
            [4, 9],
            [6, 11],
            [7, 10],
        ],
        false,
    );
    let graph = build_undigraph(&case);

    assert!(!graph.is_planar().unwrap(), "planarity should reject {} ({})", case.name, case.family);
}
