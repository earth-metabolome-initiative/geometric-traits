//! Tests for MonoplexMonopartiteGraph trait methods.

use geometric_traits::{
    impls::{SortedVec, SquareCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

/// Helper to create a simple DAG for testing.
fn create_dag() -> DiGraph<usize> {
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
    let edge_data: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(vec![0, 1, 2, 3, 4].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(5)
        .expected_shape(5)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    DiGraph::from((nodes, edges))
}

/// Helper to create a graph with self-loops.
fn create_graph_with_self_loops() -> DiGraph<usize> {
    let nodes: Vec<usize> = vec![0, 1, 2];
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (1, 1), (1, 2)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(vec![0, 1, 2].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(4)
        .expected_shape(3)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    DiGraph::from((nodes, edges))
}

// ============================================================================
// Self-loop tests
// ============================================================================

#[test]
fn test_has_self_loops_true() {
    let graph = create_graph_with_self_loops();
    assert!(graph.has_self_loops());
}

#[test]
fn test_has_self_loops_false() {
    let graph = create_dag();
    assert!(!graph.has_self_loops());
}

#[test]
fn test_number_of_self_loops() {
    let graph = create_graph_with_self_loops();
    assert_eq!(graph.number_of_self_loops(), 2);
}

#[test]
fn test_number_of_self_loops_zero() {
    let graph = create_dag();
    assert_eq!(graph.number_of_self_loops(), 0);
}

// ============================================================================
// Topological order tests
// ============================================================================

#[test]
fn test_is_topologically_sorted_true() {
    // DAG with nodes in topological order: 0 -> 1 -> 2 -> 3 -> 4
    let graph = create_dag();
    assert!(graph.is_topologically_sorted());
}

#[test]
fn test_is_topologically_sorted_false() {
    // Graph where node 3 points to node 1 (not topologically sorted)
    let nodes: Vec<usize> = vec![0, 1, 2, 3];
    let edge_data: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (3, 1)]; // 3 -> 1 breaks order

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(vec![0, 1, 2, 3].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(3)
        .expected_shape(4)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    let graph: DiGraph<usize> = DiGraph::from((nodes, edges));
    assert!(!graph.is_topologically_sorted());
}

// ============================================================================
// Path tests
// ============================================================================

#[test]
fn test_unique_paths_from() {
    let graph = create_dag();

    // From node 0, there should be multiple unique paths
    let paths = graph.unique_paths_from(0);
    assert!(!paths.is_empty());

    // From node 4 (sink), there should be only one path (just node 4)
    let paths_from_sink = graph.unique_paths_from(4);
    assert_eq!(paths_from_sink.len(), 1);
    assert_eq!(paths_from_sink[0], vec![4]);
}

#[test]
fn test_has_path() {
    let graph = create_dag();

    // There's a path from 0 to 4
    assert!(graph.has_path(0, 4));

    // There's a path from 0 to 3 (through 1 or 2)
    assert!(graph.has_path(0, 3));

    // There's no path from 4 to 0 (wrong direction)
    assert!(!graph.has_path(4, 0));

    // has_path returns false for same node (no self-loop)
    assert!(!graph.has_path(0, 0));
}

#[test]
fn test_is_reachable_through() {
    let graph = create_dag();

    // is_reachable_through(source, destination, passing_through)
    // Node 4 is reachable from 0, passing through 3
    assert!(graph.is_reachable_through(0, 4, 3));

    // Node 3 is reachable from 0, passing through 1
    assert!(graph.is_reachable_through(0, 3, 1));

    // Node 4 is reachable from 0, passing through 1
    assert!(graph.is_reachable_through(0, 4, 1));

    // Node 2 is NOT reachable from 0 passing through 3 (3 is after 2)
    assert!(!graph.is_reachable_through(0, 2, 3));
}

// ============================================================================
// Successors set test
// ============================================================================

#[test]
fn test_successors_set() {
    let graph = create_dag();

    // Node 0's successors set should contain all reachable nodes
    let successors_0 = graph.successors_set(0);
    assert!(successors_0.contains(&1));
    assert!(successors_0.contains(&2));
    assert!(successors_0.contains(&3));
    assert!(successors_0.contains(&4));
    assert!(!successors_0.contains(&0)); // Self not included

    // Node 4 (sink) has no successors
    let successors_4 = graph.successors_set(4);
    assert!(successors_4.is_empty());

    // Node 3's successors set should only contain 4
    let successors_3 = graph.successors_set(3);
    assert_eq!(successors_3.len(), 1);
    assert!(successors_3.contains(&4));
}
