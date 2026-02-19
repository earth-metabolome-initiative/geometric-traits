//! Extended tests for UndirectedMonopartiteMonoplexGraph trait methods:
//! neighbors, degree, degrees, and more undirected graph operations.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::VocabularyBuilder,
};

/// Helper to create an undirected graph with specific structure.
fn create_undirected_graph(node_count: usize, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let nodes: Vec<usize> = (0..node_count).collect();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_count = edges.len();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

// ============================================================================
// Neighbors tests
// ============================================================================

#[test]
fn test_neighbors_leaf_node() {
    // Star topology: node 0 connected to 1, 2, 3
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2), (0, 3)]);

    // Leaf node 1 should have only node 0 as neighbor
    let neighbors: Vec<usize> = graph.neighbors(1).collect();
    assert_eq!(neighbors, vec![0]);
}

#[test]
fn test_neighbors_hub_node() {
    // Star topology: node 0 connected to 1, 2, 3
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2), (0, 3)]);

    // Hub node 0 should have neighbors 1, 2, 3
    let neighbors: Vec<usize> = graph.neighbors(0).collect();
    assert_eq!(neighbors, vec![1, 2, 3]);
}

#[test]
fn test_neighbors_middle_of_chain() {
    // Chain: 0--1--2--3--4
    let graph = create_undirected_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

    // Middle node 2 should have neighbors 1, 3
    let neighbors: Vec<usize> = graph.neighbors(2).collect();
    assert_eq!(neighbors, vec![1, 3]);
}

#[test]
fn test_neighbors_isolated_node() {
    // Only nodes 0 and 1 are connected; node 2 is isolated
    let graph = create_undirected_graph(3, vec![(0, 1)]);

    let neighbors: Vec<usize> = graph.neighbors(2).collect();
    assert!(neighbors.is_empty());
}

#[test]
fn test_neighbors_triangle() {
    // Triangle: 0--1, 0--2, 1--2
    let graph = create_undirected_graph(3, vec![(0, 1), (0, 2), (1, 2)]);

    let n0: Vec<usize> = graph.neighbors(0).collect();
    assert_eq!(n0, vec![1, 2]);

    let n1: Vec<usize> = graph.neighbors(1).collect();
    assert_eq!(n1, vec![0, 2]);

    let n2: Vec<usize> = graph.neighbors(2).collect();
    assert_eq!(n2, vec![0, 1]);
}

// ============================================================================
// Degree tests
// ============================================================================

#[test]
fn test_degree_star_topology() {
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2), (0, 3)]);

    assert_eq!(graph.degree(0), 3); // Hub
    assert_eq!(graph.degree(1), 1); // Leaf
    assert_eq!(graph.degree(2), 1); // Leaf
    assert_eq!(graph.degree(3), 1); // Leaf
}

#[test]
fn test_degree_chain() {
    let graph = create_undirected_graph(4, vec![(0, 1), (1, 2), (2, 3)]);

    assert_eq!(graph.degree(0), 1); // End
    assert_eq!(graph.degree(1), 2); // Middle
    assert_eq!(graph.degree(2), 2); // Middle
    assert_eq!(graph.degree(3), 1); // End
}

#[test]
fn test_degree_isolated_node() {
    let graph = create_undirected_graph(3, vec![(0, 1)]);
    assert_eq!(graph.degree(2), 0);
}

#[test]
fn test_degree_complete_graph_k4() {
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);

    // Every node in K4 has degree 3
    for node in 0..4 {
        assert_eq!(graph.degree(node), 3);
    }
}

// ============================================================================
// Degrees iterator tests
// ============================================================================

#[test]
fn test_degrees_star_topology() {
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2), (0, 3)]);

    let degrees: Vec<usize> = graph.degrees().collect();
    assert_eq!(degrees, vec![3, 1, 1, 1]);
}

#[test]
fn test_degrees_chain() {
    let graph = create_undirected_graph(4, vec![(0, 1), (1, 2), (2, 3)]);

    let degrees: Vec<usize> = graph.degrees().collect();
    assert_eq!(degrees, vec![1, 2, 2, 1]);
}

#[test]
fn test_degrees_with_isolated() {
    let graph = create_undirected_graph(4, vec![(0, 1), (0, 2)]);

    let degrees: Vec<usize> = graph.degrees().collect();
    assert_eq!(degrees, vec![2, 1, 1, 0]);
}

#[test]
fn test_degrees_no_edges() {
    let graph = create_undirected_graph(3, vec![]);

    let degrees: Vec<usize> = graph.degrees().collect();
    assert_eq!(degrees, vec![0, 0, 0]);
}

// ============================================================================
// Combined graph property tests
// ============================================================================

#[test]
fn test_undirected_graph_counts() {
    let graph = create_undirected_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

    assert_eq!(graph.number_of_nodes(), 5);
    // Each undirected edge creates 2 directed entries
    assert_eq!(graph.number_of_edges(), 8);
    assert!(graph.has_nodes());
    assert!(graph.has_edges());
}

#[test]
fn test_undirected_graph_nodes() {
    let graph = create_undirected_graph(3, vec![(0, 1)]);

    let node_ids: Vec<usize> = graph.node_ids().collect();
    assert_eq!(node_ids, vec![0, 1, 2]);

    let nodes: Vec<usize> = graph.nodes().collect();
    assert_eq!(nodes, vec![0, 1, 2]);
}

#[test]
fn test_undirected_degree_sum_equals_twice_edges() {
    // Handshaking lemma: sum of degrees = 2 * number of edges (undirected)
    let graph = create_undirected_graph(5, vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]);

    let degree_sum: usize = graph.degrees().sum();
    // number_of_edges() returns directed count = 2 * undirected edges
    assert_eq!(degree_sum, graph.number_of_edges());
}
