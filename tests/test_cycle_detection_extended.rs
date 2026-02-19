//! Extended tests for CycleDetection trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::*,
    traits::{CycleDetection, EdgesBuilder, VocabularyBuilder},
};

/// Helper to build a DiGraph from directed edges.
fn build_digraph(node_count: usize, edges: Vec<(usize, usize)>) -> DiGraph<usize> {
    let nodes: Vec<usize> = (0..node_count).collect();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_count = edges.len();
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    DiGraph::from((nodes, edges))
}

// ============================================================================
// Acyclic graphs
// ============================================================================

#[test]
fn test_no_edges_no_cycle() {
    let graph = build_digraph(3, vec![]);
    assert!(!graph.has_cycle());
}

#[test]
fn test_chain_no_cycle() {
    // 0 -> 1 -> 2 -> 3
    let graph = build_digraph(4, vec![(0, 1), (1, 2), (2, 3)]);
    assert!(!graph.has_cycle());
}

#[test]
fn test_diamond_dag_no_cycle() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert!(!graph.has_cycle());
}

#[test]
fn test_tree_no_cycle() {
    // Binary tree: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    let graph = build_digraph(5, vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    assert!(!graph.has_cycle());
}

#[test]
fn test_single_node_no_cycle() {
    let graph = build_digraph(1, vec![]);
    assert!(!graph.has_cycle());
}

// ============================================================================
// Cyclic graphs
// ============================================================================

#[test]
fn test_self_loop_is_cycle() {
    let graph = build_digraph(1, vec![(0, 0)]);
    assert!(graph.has_cycle());
}

#[test]
fn test_two_node_cycle() {
    let graph = build_digraph(2, vec![(0, 1), (1, 0)]);
    assert!(graph.has_cycle());
}

#[test]
fn test_triangle_cycle() {
    let graph = build_digraph(3, vec![(0, 1), (1, 2), (2, 0)]);
    assert!(graph.has_cycle());
}

#[test]
fn test_cycle_in_subgraph() {
    // 0 -> 1, 1 -> 2, 2 -> 1 (cycle), 2 -> 3
    let graph = build_digraph(4, vec![(0, 1), (1, 2), (2, 1), (2, 3)]);
    assert!(graph.has_cycle());
}

#[test]
fn test_long_cycle() {
    // 0 -> 1 -> 2 -> 3 -> 4 -> 0
    let graph = build_digraph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    assert!(graph.has_cycle());
}

#[test]
fn test_cycle_with_disconnected_nodes() {
    // Cycle: 0 -> 1 -> 0, isolated: 2, 3
    let graph = build_digraph(4, vec![(0, 1), (1, 0)]);
    assert!(graph.has_cycle());
}

// ============================================================================
// Mixed structure tests
// ============================================================================

#[test]
fn test_dag_with_convergent_paths() {
    // Multiple paths converge but no cycle
    // 0 -> 2, 0 -> 3, 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
    let graph = build_digraph(5, vec![(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]);
    assert!(!graph.has_cycle());
}

#[test]
fn test_wide_dag_no_cycle() {
    // Star: 0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4
    let graph = build_digraph(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    assert!(!graph.has_cycle());
}
