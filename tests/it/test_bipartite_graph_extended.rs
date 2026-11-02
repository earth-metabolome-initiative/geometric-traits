//! Extended tests for BipartiteGraph, MonoplexBipartiteGraph, and bipartite
//! operations.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    naive_structs::named_types::BiGraph,
    prelude::*,
    traits::{BipartiteGraph, EdgesBuilder, VocabularyBuilder},
};

/// Helper to create a bipartite graph.
fn create_bigraph(left: Vec<u16>, right: Vec<u8>, edges: Vec<(usize, usize)>) -> BiGraph<u16, u8> {
    let left_count = left.len();
    let right_count = right.len();
    let edge_count = edges.len();

    let left_nodes: SortedVec<u16> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(left_count)
        .symbols(left.into_iter().enumerate())
        .build()
        .unwrap();

    let right_nodes: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(right_count)
        .symbols(right.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(edge_count)
            .expected_shape((left_count, right_count))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap()
}

// ============================================================================
// Node count tests
// ============================================================================

#[test]
fn test_single_left_single_right() {
    let bg = create_bigraph(vec![10], vec![1], vec![(0, 0)]);

    assert_eq!(bg.number_of_left_nodes(), 1);
    assert_eq!(bg.number_of_right_nodes(), 1);
    assert_eq!(bg.number_of_edges(), 1);
}

#[test]
fn test_many_left_few_right() {
    let bg = create_bigraph(
        vec![10, 20, 30, 40, 50],
        vec![1, 2],
        vec![(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)],
    );

    assert_eq!(bg.number_of_left_nodes(), 5);
    assert_eq!(bg.number_of_right_nodes(), 2);
    assert_eq!(bg.number_of_edges(), 5);
}

#[test]
fn test_few_left_many_right() {
    let bg = create_bigraph(
        vec![10, 20],
        vec![1, 2, 3, 4, 5],
        vec![(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)],
    );

    assert_eq!(bg.number_of_left_nodes(), 2);
    assert_eq!(bg.number_of_right_nodes(), 5);
    assert_eq!(bg.number_of_edges(), 5);
}

// ============================================================================
// Node lookups
// ============================================================================

#[test]
fn test_left_node_lookup() {
    let bg = create_bigraph(vec![100, 200, 300], vec![1], vec![(0, 0), (1, 0), (2, 0)]);

    assert_eq!(bg.left_node(&0), Some(100));
    assert_eq!(bg.left_node(&1), Some(200));
    assert_eq!(bg.left_node(&2), Some(300));
    assert_eq!(bg.left_node(&3), None);
}

#[test]
fn test_right_node_lookup() {
    let bg = create_bigraph(vec![10], vec![1, 2, 3], vec![(0, 0), (0, 1), (0, 2)]);

    assert_eq!(bg.right_node(&0), Some(1));
    assert_eq!(bg.right_node(&1), Some(2));
    assert_eq!(bg.right_node(&2), Some(3));
    assert_eq!(bg.right_node(&3), None);
}

#[test]
fn test_left_node_id_lookup() {
    let bg = create_bigraph(vec![100, 200], vec![1], vec![(0, 0), (1, 0)]);

    assert_eq!(bg.left_node_id(&100_u16), Some(0));
    assert_eq!(bg.left_node_id(&200_u16), Some(1));
    assert_eq!(bg.left_node_id(&999_u16), None);
}

#[test]
fn test_right_node_id_lookup() {
    let bg = create_bigraph(vec![10], vec![5, 10, 15], vec![(0, 0), (0, 1), (0, 2)]);

    assert_eq!(bg.right_node_id(&5_u8), Some(0));
    assert_eq!(bg.right_node_id(&10_u8), Some(1));
    assert_eq!(bg.right_node_id(&15_u8), Some(2));
    assert_eq!(bg.right_node_id(&99_u8), None);
}

// ============================================================================
// Iterators
// ============================================================================

#[test]
fn test_left_node_ids_iterator() {
    let bg = create_bigraph(vec![10, 20, 30], vec![1], vec![(0, 0), (1, 0), (2, 0)]);

    let ids: Vec<usize> = bg.left_node_ids().collect();
    assert_eq!(ids, vec![0, 1, 2]);
}

#[test]
fn test_right_node_ids_iterator() {
    let bg = create_bigraph(vec![10], vec![1, 2, 3, 4], vec![(0, 0), (0, 1), (0, 2), (0, 3)]);

    let ids: Vec<usize> = bg.right_node_ids().collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

#[test]
fn test_left_nodes_iterator() {
    let bg = create_bigraph(vec![10, 20, 30], vec![1], vec![(0, 0), (1, 0), (2, 0)]);

    let nodes: Vec<u16> = bg.left_nodes().collect();
    assert_eq!(nodes, vec![10, 20, 30]);
}

#[test]
fn test_right_nodes_iterator() {
    let bg = create_bigraph(vec![10], vec![1, 2, 3], vec![(0, 0), (0, 1), (0, 2)]);

    let nodes: Vec<u8> = bg.right_nodes().collect();
    assert_eq!(nodes, vec![1, 2, 3]);
}

// ============================================================================
// Edge operations
// ============================================================================

#[test]
fn test_successors() {
    let bg = create_bigraph(vec![10, 20], vec![1, 2, 3], vec![(0, 0), (0, 2), (1, 1)]);

    let succ0: Vec<usize> = bg.successors(0).collect();
    assert_eq!(succ0, vec![0, 2]);

    let succ1: Vec<usize> = bg.successors(1).collect();
    assert_eq!(succ1, vec![1]);
}

#[test]
fn test_out_degree() {
    let bg = create_bigraph(vec![10, 20, 30], vec![1, 2], vec![(0, 0), (0, 1), (1, 0)]);

    assert_eq!(bg.out_degree(0), 2);
    assert_eq!(bg.out_degree(1), 1);
    assert_eq!(bg.out_degree(2), 0);
}

#[test]
fn test_has_edges() {
    let bg = create_bigraph(vec![10], vec![1], vec![(0, 0)]);
    assert!(bg.has_edges());

    let bg_empty = create_bigraph(vec![10], vec![1], vec![]);
    assert!(!bg_empty.has_edges());
}

#[test]
fn test_has_nodes() {
    let bg = create_bigraph(vec![10], vec![1], vec![(0, 0)]);
    assert!(bg.has_nodes());
}

// ============================================================================
// DOT output tests
// ============================================================================

#[test]
fn test_dot_output_contains_edges() {
    let bg = create_bigraph(vec![10, 20], vec![1, 2], vec![(0, 0), (0, 1), (1, 0)]);

    let dot = bg.to_mb_dot();
    assert!(dot.contains("graph"));
    assert!(dot.contains('}'));
}

#[test]
fn test_dot_output_empty_graph() {
    let bg = create_bigraph(vec![10], vec![1], vec![]);
    let dot = bg.to_mb_dot();
    assert!(dot.contains("graph"));
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_fully_connected_bipartite() {
    // Every left node connects to every right node
    let bg = create_bigraph(
        vec![10, 20],
        vec![1, 2, 3],
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    );

    assert_eq!(bg.number_of_edges(), 6);
    assert_eq!(bg.out_degree(0), 3);
    assert_eq!(bg.out_degree(1), 3);
}

#[test]
fn test_no_edges_bipartite() {
    let bg = create_bigraph(vec![10, 20], vec![1, 2], vec![]);

    assert_eq!(bg.number_of_edges(), 0);
    assert!(!bg.has_edges());
    assert_eq!(bg.out_degree(0), 0);
    assert_eq!(bg.out_degree(1), 0);
}
