//! Tests for BipartiteGraph trait methods.

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    naive_structs::named_types::BiGraph,
    prelude::*,
    traits::{BipartiteGraph, EdgesBuilder, VocabularyBuilder},
};

/// Helper to create a simple bipartite graph for testing.
fn create_bipartite_graph() -> BiGraph<u16, u8> {
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 0)];

    let left_nodes: SortedVec<u16> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(4)
        .symbols(vec![10_u16, 20, 30, 40].into_iter().enumerate())
        .build()
        .unwrap();

    let right_nodes: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![1_u8, 2, 3].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(6)
            .expected_shape((4, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap()
}

// ============================================================================
// Left nodes tests
// ============================================================================

#[test]
fn test_left_nodes_vocabulary() {
    let graph = create_bipartite_graph();
    let vocab = graph.left_nodes_vocabulary();

    assert_eq!(vocab.len(), 4);
}

#[test]
fn test_left_node_ids() {
    let graph = create_bipartite_graph();
    let ids: Vec<usize> = graph.left_node_ids().collect();

    assert_eq!(ids, vec![0, 1, 2, 3]);
}

#[test]
fn test_left_nodes() {
    let graph = create_bipartite_graph();
    let nodes: Vec<u16> = graph.left_nodes().collect();

    assert_eq!(nodes, vec![10, 20, 30, 40]);
}

#[test]
fn test_left_node() {
    let graph = create_bipartite_graph();

    assert_eq!(graph.left_node(&0), Some(10));
    assert_eq!(graph.left_node(&1), Some(20));
    assert_eq!(graph.left_node(&2), Some(30));
    assert_eq!(graph.left_node(&3), Some(40));
    assert_eq!(graph.left_node(&4), None);
}

#[test]
fn test_left_node_id() {
    let graph = create_bipartite_graph();

    assert_eq!(graph.left_node_id(&10_u16), Some(0));
    assert_eq!(graph.left_node_id(&20_u16), Some(1));
    assert_eq!(graph.left_node_id(&30_u16), Some(2));
    assert_eq!(graph.left_node_id(&40_u16), Some(3));
    assert_eq!(graph.left_node_id(&50_u16), None);
}

#[test]
fn test_number_of_left_nodes() {
    let graph = create_bipartite_graph();
    assert_eq!(graph.number_of_left_nodes(), 4);
}

// ============================================================================
// Right nodes tests
// ============================================================================

#[test]
fn test_right_nodes_vocabulary() {
    let graph = create_bipartite_graph();
    let vocab = graph.right_nodes_vocabulary();

    assert_eq!(vocab.len(), 3);
}

#[test]
fn test_right_node_ids() {
    let graph = create_bipartite_graph();
    let ids: Vec<usize> = graph.right_node_ids().collect();

    assert_eq!(ids, vec![0, 1, 2]);
}

#[test]
fn test_right_nodes() {
    let graph = create_bipartite_graph();
    let nodes: Vec<u8> = graph.right_nodes().collect();

    assert_eq!(nodes, vec![1, 2, 3]);
}

#[test]
fn test_right_node() {
    let graph = create_bipartite_graph();

    assert_eq!(graph.right_node(&0), Some(1));
    assert_eq!(graph.right_node(&1), Some(2));
    assert_eq!(graph.right_node(&2), Some(3));
    assert_eq!(graph.right_node(&3), None);
}

#[test]
fn test_right_node_id() {
    let graph = create_bipartite_graph();

    assert_eq!(graph.right_node_id(&1_u8), Some(0));
    assert_eq!(graph.right_node_id(&2_u8), Some(1));
    assert_eq!(graph.right_node_id(&3_u8), Some(2));
    assert_eq!(graph.right_node_id(&4_u8), None);
}

#[test]
fn test_number_of_right_nodes() {
    let graph = create_bipartite_graph();
    assert_eq!(graph.number_of_right_nodes(), 3);
}

// ============================================================================
// Edge tests
// ============================================================================

#[test]
fn test_bipartite_number_of_edges() {
    let graph = create_bipartite_graph();
    assert_eq!(graph.number_of_edges(), 6);
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_empty_bipartite_graph() {
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

    assert_eq!(graph.number_of_left_nodes(), 0);
    assert_eq!(graph.number_of_right_nodes(), 0);
    assert_eq!(graph.number_of_edges(), 0);
}
