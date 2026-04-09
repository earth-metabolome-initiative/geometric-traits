//! Integration tests for exact `K_{2,3}` homeomorph detection on undirected
//! graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, K23HomeomorphDetection, K23HomeomorphError, VocabularyBuilder},
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph};

fn k23_case(
    name: &str,
    family: &str,
    node_count: usize,
    edges: &[[usize; 2]],
    expected: bool,
) -> (PlanarityFixtureCase, bool) {
    (
        PlanarityFixtureCase {
            name: name.to_string(),
            family: family.to_string(),
            node_count,
            edges: edges.to_vec(),
            is_planar: true,
            is_outerplanar: false,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "K23 homeomorph detection regression/semantic case.".to_string(),
        },
        expected,
    )
}

#[test]
fn test_k23_homeomorph_rejects_self_loops() {
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
        graph.has_k23_homeomorph(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K23HomeomorphError(
            K23HomeomorphError::SelfLoopsUnsupported
        )))
    ));
}

#[test]
fn test_k23_homeomorph_semantic_cases() {
    let cases = [
        k23_case("path_four", "tree", 4, &[[0, 1], [1, 2], [2, 3]], false),
        k23_case(
            "diamond_k4_minus_edge",
            "outerplanar_with_chord",
            4,
            &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            false,
        ),
        k23_case(
            "k4_complete",
            "outerplanarity_obstruction",
            4,
            &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            false,
        ),
        k23_case(
            "k23_complete_bipartite",
            "outerplanarity_obstruction",
            5,
            &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
            true,
        ),
        k23_case(
            "k33_complete_bipartite",
            "planarity_obstruction",
            6,
            &[[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            true,
        ),
        k23_case(
            "k5_complete",
            "planarity_obstruction",
            5,
            &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            true,
        ),
        k23_case(
            "subdivided_k4_edge",
            "k4_subdivision",
            5,
            &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1], [3, 4]],
            true,
        ),
    ];

    for (case, expected) in cases {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.has_k23_homeomorph().unwrap(),
            expected,
            "K23 homeomorph mismatch on {} ({})",
            case.name,
            case.family
        );
    }
}

#[test]
fn test_k23_homeomorph_accepts_theta_regression() {
    let (case, expected) = k23_case(
        "theta_three_length_two_paths",
        "theta",
        5,
        &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(
        graph.has_k23_homeomorph().unwrap() == expected,
        "K23 homeomorph should accept {} ({})",
        case.name,
        case.family
    );
}
