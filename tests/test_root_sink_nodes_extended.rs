//! Extended tests for RootNodes and SinkNodes traits.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::*,
    traits::{EdgesBuilder, RootNodes, SinkNodes, VocabularyBuilder},
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
// RootNodes tests
// ============================================================================

#[test]
fn test_root_nodes_chain() {
    // 0 -> 1 -> 2 -> 3, only node 0 has no predecessors
    let graph = build_digraph(4, vec![(0, 1), (1, 2), (2, 3)]);
    let roots = graph.root_nodes();
    assert_eq!(roots, vec![0]);
}

#[test]
fn test_root_nodes_tree() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    let graph = build_digraph(5, vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let roots = graph.root_nodes();
    assert_eq!(roots, vec![0]);
}

#[test]
fn test_root_nodes_multiple_roots() {
    // 0 -> 2, 1 -> 2, 2 -> 3
    let graph = build_digraph(4, vec![(0, 2), (1, 2), (2, 3)]);
    let mut roots = graph.root_nodes();
    roots.sort_unstable();
    assert_eq!(roots, vec![0, 1]);
}

#[test]
fn test_root_nodes_no_edges() {
    // All nodes are roots when there are no edges (no predecessors)
    let graph = build_digraph(3, vec![]);
    let roots = graph.root_nodes();
    assert_eq!(roots, vec![0, 1, 2]);
}

#[test]
fn test_root_nodes_single_node() {
    let graph = build_digraph(1, vec![]);
    let roots = graph.root_nodes();
    assert_eq!(roots, vec![0]);
}

#[test]
fn test_root_nodes_diamond() {
    //   0
    //  / \
    // 1   2
    //  \ /
    //   3
    let graph = build_digraph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let roots = graph.root_nodes();
    assert_eq!(roots, vec![0]);
}

// ============================================================================
// SinkNodes tests
// ============================================================================

#[test]
fn test_sink_nodes_chain() {
    // 0 -> 1 -> 2 -> 3, node 3 has predecessors and no successors
    let graph = build_digraph(4, vec![(0, 1), (1, 2), (2, 3)]);
    let sinks = graph.sink_nodes();
    assert_eq!(sinks, vec![3]);
}

#[test]
fn test_sink_nodes_tree() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    // Sinks: 2, 3, 4 (have predecessors, no successors)
    let graph = build_digraph(5, vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let mut sinks = graph.sink_nodes();
    sinks.sort_unstable();
    assert_eq!(sinks, vec![2, 3, 4]);
}

#[test]
fn test_sink_nodes_multiple_predecessors() {
    // 0 -> 2, 1 -> 2
    // Node 2 is a sink (has predecessors, no successors)
    let graph = build_digraph(3, vec![(0, 2), (1, 2)]);
    let sinks = graph.sink_nodes();
    assert_eq!(sinks, vec![2]);
}

#[test]
fn test_sink_nodes_no_edges() {
    // No edges: no node has predecessors, so no sinks
    let graph = build_digraph(3, vec![]);
    let sinks = graph.sink_nodes();
    assert!(sinks.is_empty());
}

#[test]
fn test_sink_nodes_single_node_no_edges() {
    let graph = build_digraph(1, vec![]);
    let sinks = graph.sink_nodes();
    // No predecessors, so no sinks
    assert!(sinks.is_empty());
}

#[test]
fn test_sink_nodes_diamond() {
    let graph = build_digraph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let sinks = graph.sink_nodes();
    assert_eq!(sinks, vec![3]);
}

// ============================================================================
// Combined root and sink tests
// ============================================================================

#[test]
fn test_roots_and_sinks_chain() {
    let graph = build_digraph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

    let roots = graph.root_nodes();
    let sinks = graph.sink_nodes();

    assert_eq!(roots, vec![0]);
    assert_eq!(sinks, vec![4]);
}

#[test]
fn test_wide_dag_roots_and_sinks() {
    // Star: 0 -> 1, 0 -> 2, 0 -> 3
    let graph = build_digraph(4, vec![(0, 1), (0, 2), (0, 3)]);

    let roots = graph.root_nodes();
    let mut sinks = graph.sink_nodes();
    sinks.sort_unstable();

    assert_eq!(roots, vec![0]);
    assert_eq!(sinks, vec![1, 2, 3]);
}

#[test]
fn test_self_loop_not_a_sink() {
    // 0 -> 0 (self-loop), 0 -> 1
    // Node 1 is a sink (has predecessor 0, no successors)
    // Node 0 is not a sink (has successors: 0 and 1)
    let graph = build_digraph(2, vec![(0, 0), (0, 1)]);

    let sinks = graph.sink_nodes();
    assert_eq!(sinks, vec![1]);
}

#[test]
fn test_mutual_edge_no_sinks() {
    // 0 -> 1, 1 -> 0 (both have predecessors and successors)
    let graph = build_digraph(2, vec![(0, 1), (1, 0)]);

    let roots = graph.root_nodes();
    let sinks = graph.sink_nodes();

    assert!(roots.is_empty());
    assert!(sinks.is_empty());
}
