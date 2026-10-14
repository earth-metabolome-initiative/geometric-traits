//! Semantic and fixture-backed tests for the Hopcroft-Tarjan
//! biconnected-components trait.
#![cfg(feature = "std")]

#[path = "support/biconnected_fixture.rs"]
mod biconnected_fixture;

use biconnected_fixture::{
    BiconnectedFixtureCase, build_undigraph, load_fixture_suite, semantic_cases,
};
use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{BiconnectedComponents, BiconnectedComponentsError, EdgesBuilder, VocabularyBuilder},
};

const EXHAUSTIVE_FIXTURE_NAME: &str = "biconnected_components_order5_exhaustive.json.gz";

fn assert_cases_match_algorithm(cases: &[BiconnectedFixtureCase]) {
    for case in cases {
        let graph = build_undigraph(case);
        let decomposition = graph.biconnected_components().unwrap();

        let edge_components: Vec<Vec<[usize; 2]>> =
            decomposition.edge_biconnected_components().cloned().collect();
        let vertex_components: Vec<Vec<usize>> =
            decomposition.vertex_biconnected_components().cloned().collect();
        let articulation_points: Vec<usize> = decomposition.articulation_points().collect();
        let bridges: Vec<[usize; 2]> = decomposition.bridges().collect();
        let omitted_vertices: Vec<usize> =
            decomposition.vertices_without_biconnected_component().collect();
        let cyclic_component_ids: Vec<usize> =
            decomposition.cyclic_biconnected_component_ids().collect();

        assert_eq!(
            edge_components, case.edge_biconnected_components,
            "edge blocks mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            vertex_components, case.vertex_biconnected_components,
            "vertex blocks mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            articulation_points, case.articulation_points,
            "articulation points mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            bridges, case.bridges,
            "bridges mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            omitted_vertices, case.vertices_without_biconnected_component,
            "omitted vertices mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            cyclic_component_ids, case.cyclic_biconnected_component_indices,
            "cyclic component ids mismatched fixture case {} ({})",
            case.name, case.family
        );
        assert_eq!(
            decomposition.number_of_biconnected_components(),
            case.edge_biconnected_components.len(),
            "biconnected-component count mismatched fixture case {} ({})",
            case.name,
            case.family
        );
        assert_eq!(
            decomposition.number_of_connected_components(),
            case.connected_components.len(),
            "connected-component count mismatched fixture case {} ({})",
            case.name,
            case.family
        );
        assert_eq!(
            decomposition.is_biconnected(),
            case.is_biconnected,
            "graph-level biconnected flag mismatched fixture case {} ({})",
            case.name,
            case.family
        );
        assert_eq!(
            graph.is_biconnected().unwrap(),
            case.is_biconnected,
            "trait convenience flag mismatched fixture case {} ({})",
            case.name,
            case.family
        );
    }
}

#[test]
fn test_biconnected_components_matches_semantic_cases() {
    let cases = semantic_cases();
    assert_cases_match_algorithm(&cases);
}

#[test]
fn test_biconnected_components_matches_exhaustive_order5_fixture_suite() {
    let suite = load_fixture_suite(EXHAUSTIVE_FIXTURE_NAME);
    assert_cases_match_algorithm(&suite.cases);
}

#[test]
fn test_biconnected_components_rejects_self_loops() {
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
        graph.biconnected_components(),
        Err(MonopartiteError::AlgorithmError(
            MonopartiteAlgorithmError::BiconnectedComponentsError(
                BiconnectedComponentsError::SelfLoopsUnsupported
            )
        ))
    ));
    assert!(matches!(
        graph.is_biconnected(),
        Err(MonopartiteError::AlgorithmError(
            MonopartiteAlgorithmError::BiconnectedComponentsError(
                BiconnectedComponentsError::SelfLoopsUnsupported
            )
        ))
    ));
}
