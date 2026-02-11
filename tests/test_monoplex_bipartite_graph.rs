//! Tests for MonoplexBipartiteGraph trait methods.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    naive_structs::named_types::BiGraph,
    prelude::*,
    traits::{EdgesBuilder, MonoplexBipartiteGraph, VocabularyBuilder},
};

/// Helper to create a simple bipartite graph for testing.
fn create_bipartite_graph() -> BiGraph<u16, u8> {
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)];

    let left_nodes: SortedVec<u16> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![10_u16, 20, 30].into_iter().enumerate())
        .build()
        .unwrap();

    let right_nodes: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![1_u8, 2, 3].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(5)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap()
}

// ============================================================================
// DOT output tests
// ============================================================================

#[test]
fn test_to_mb_dot_contains_graph_keyword() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    assert!(dot.contains("graph {"));
}

#[test]
fn test_to_mb_dot_contains_left_nodes() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Left nodes should be labeled L0, L1, L2
    assert!(dot.contains("L0"));
    assert!(dot.contains("L1"));
    assert!(dot.contains("L2"));
}

#[test]
fn test_to_mb_dot_left_nodes_red() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Left nodes should be colored red
    assert!(dot.contains("[color=red]"));
}

#[test]
fn test_to_mb_dot_contains_right_nodes() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Right nodes should be labeled R0, R1, R2
    assert!(dot.contains("R0"));
    assert!(dot.contains("R1"));
    assert!(dot.contains("R2"));
}

#[test]
fn test_to_mb_dot_right_nodes_blue() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Right nodes should be colored blue
    assert!(dot.contains("[color=blue]"));
}

#[test]
fn test_to_mb_dot_contains_edges() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Should contain edge declarations
    // Edges: (0,0), (0,1), (1,1), (1,2), (2,2)
    assert!(dot.contains("L0 -> R0;"));
    assert!(dot.contains("L0 -> R1;"));
    assert!(dot.contains("L1 -> R1;"));
    assert!(dot.contains("L1 -> R2;"));
    assert!(dot.contains("L2 -> R2;"));
}

#[test]
fn test_to_mb_dot_ends_properly() {
    let graph = create_bipartite_graph();
    let dot = graph.to_mb_dot();

    // Should end with closing brace
    assert!(dot.trim().ends_with('}'));
}

#[test]
fn test_to_mb_dot_empty_graph() {
    let left_nodes: SortedVec<u16> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(0)
        .symbols(core::iter::empty::<(usize, u16)>())
        .build()
        .unwrap();

    let right_nodes: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(0)
        .symbols(core::iter::empty::<(usize, u8)>())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(0)
            .expected_shape((0, 0))
            .edges(core::iter::empty())
            .build()
            .unwrap();

    let graph: BiGraph<u16, u8> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();

    let dot = graph.to_mb_dot();
    assert!(dot.contains("graph {"));
    assert!(dot.contains('}'));
}
