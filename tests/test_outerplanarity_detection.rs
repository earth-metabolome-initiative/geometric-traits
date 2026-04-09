//! Integration tests for exact outerplanarity detection on undirected graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, OuterplanarityDetection, OuterplanarityError, VocabularyBuilder},
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph, semantic_cases};

#[test]
fn test_outerplanarity_matches_semantic_cases() {
    for case in semantic_cases() {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.is_outerplanar().unwrap(),
            case.is_outerplanar,
            "outerplanarity mismatched fixture case {} ({})",
            case.name,
            case.family
        );
    }
}

#[test]
fn test_outerplanarity_rejects_self_loops() {
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
        graph.is_outerplanar(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::OuterplanarityError(
            OuterplanarityError::SelfLoopsUnsupported
        )))
    ));
}

#[test]
fn test_outerplanarity_accepts_outerplanar_cycle_chords_regression() {
    let case = PlanarityFixtureCase {
        name: "outerplanar_cycle_chords_503906".to_string(),
        family: "outerplanar_cycle_chords".to_string(),
        node_count: 6,
        edges: vec![[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]],
        is_planar: true,
        is_outerplanar: true,
        planarity_obstruction_family: None,
        outerplanarity_obstruction_family: None,
        notes: "Regression case extracted from the Boyer local corpus.".to_string(),
    };
    let graph = build_undigraph(&case);

    assert!(
        graph.is_outerplanar().unwrap(),
        "outerplanarity should accept {} ({})",
        case.name,
        case.family
    );
}
