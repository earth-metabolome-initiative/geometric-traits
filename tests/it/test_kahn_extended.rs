//! Extended tests for the Kahn topological sort algorithm.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder},
    traits::{Edges, EdgesBuilder, Kahn, KahnError, MonoplexGraph, VocabularyBuilder},
};

/// Helper to build a directed graph from node and edge lists.
fn build_digraph(node_list: Vec<usize>, edge_list: Vec<(usize, usize)>) -> DiGraph<usize> {
    let num_nodes = node_list.len();
    let num_edges = edge_list.len();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(node_list.into_iter().enumerate())
        .build()
        .unwrap();
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(nodes.len())
        .edges(edge_list.into_iter())
        .build()
        .unwrap();
    DiGraph::from((nodes, edges))
}

#[test]
fn test_kahn_simple_chain() {
    // Chain: 0 -> 1 -> 2
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 3);
    // In a chain, 0 must come before 1 which must come before 2
    assert!(ordering[0] < ordering[1]);
    assert!(ordering[1] < ordering[2]);
}

#[test]
fn test_kahn_diamond_dag() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 4);
    // 0 must come first (root), 3 must come last (only sink)
    assert!(ordering[0] < ordering[1]);
    assert!(ordering[0] < ordering[2]);
    assert!(ordering[1] < ordering[3]);
    assert!(ordering[2] < ordering[3]);
}

#[test]
fn test_kahn_cycle_returns_error() {
    // Cycle: 0 -> 1 -> 2 -> 0
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    let result = graph.edges().matrix().kahn();
    assert_eq!(result, Err(KahnError::Cycle));
}

#[test]
fn test_kahn_self_loop_returns_error() {
    // Self-loop: 0 -> 0
    let graph = build_digraph(vec![0, 1], vec![(0, 0)]);
    let result = graph.edges().matrix().kahn();
    assert_eq!(result, Err(KahnError::Cycle));
}

#[test]
fn test_kahn_no_edges() {
    // No edges: all nodes are independent; any ordering is valid.
    let graph = build_digraph(vec![0, 1, 2], vec![]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 3);
}

#[test]
fn test_kahn_single_node() {
    let graph = build_digraph(vec![0], vec![]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 1);
}

#[test]
fn test_kahn_wide_dag() {
    // Fan out: 0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 5);
    // Node 0 should come before all others
    for i in 1..5 {
        assert!(ordering[0] < ordering[i]);
    }
}

#[test]
fn test_kahn_two_node_cycle() {
    // Mutual cycle: 0 -> 1, 1 -> 0
    let graph = build_digraph(vec![0, 1], vec![(0, 1), (1, 0)]);
    let result = graph.edges().matrix().kahn();
    assert_eq!(result, Err(KahnError::Cycle));
}

#[test]
fn test_kahn_error_debug_and_clone() {
    let err = KahnError::Cycle;
    let cloned = err.clone();
    assert_eq!(err, cloned);
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("Cycle"));
}

#[test]
fn test_kahn_disconnected_dag() {
    // Two disconnected chains: 0 -> 1 and 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (2, 3)]);
    let ordering = graph.edges().matrix().kahn().unwrap();
    assert_eq!(ordering.len(), 4);
    assert!(ordering[0] < ordering[1]);
    assert!(ordering[2] < ordering[3]);
}
