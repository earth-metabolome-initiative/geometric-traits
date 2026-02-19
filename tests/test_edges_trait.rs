//! Tests for Edges trait methods: has_successors, has_edges, number_of_edges,
//! out_degree, out_degrees, sparse_coordinates, has_successor.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder},
    traits::{Edges, EdgesBuilder, Graph, MonoplexGraph, VocabularyBuilder},
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

// ============================================================================
// has_successors tests
// ============================================================================

#[test]
fn test_has_successors_true() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(graph.edges().has_successors(0));
    assert!(graph.edges().has_successors(1));
}

#[test]
fn test_has_successors_false() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(!graph.edges().has_successors(2));
}

// ============================================================================
// has_successor tests
// ============================================================================

#[test]
fn test_has_successor_true() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(graph.edges().has_successor(0, 1));
}

#[test]
fn test_has_successor_false() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(!graph.edges().has_successor(0, 2));
    assert!(!graph.edges().has_successor(2, 0));
}

// ============================================================================
// out_degree tests
// ============================================================================

#[test]
fn test_out_degree() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3), (1, 2)]);
    assert_eq!(graph.edges().out_degree(0), 3);
    assert_eq!(graph.edges().out_degree(1), 1);
    assert_eq!(graph.edges().out_degree(2), 0);
    assert_eq!(graph.edges().out_degree(3), 0);
}

// ============================================================================
// out_degrees tests
// ============================================================================

#[test]
fn test_out_degrees() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3)]);
    let degrees: Vec<usize> = graph.edges().out_degrees().collect();
    assert_eq!(degrees, vec![2, 1, 0, 0]);
}

// ============================================================================
// number_of_edges / has_edges
// ============================================================================

#[test]
fn test_number_of_edges() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert_eq!(graph.edges().number_of_edges(), 2);
    assert_eq!(graph.number_of_edges(), 2);
}

#[test]
fn test_has_edges_true() {
    let graph = build_digraph(vec![0, 1], vec![(0, 1)]);
    assert!(graph.edges().has_edges());
    assert!(graph.has_edges());
}

#[test]
fn test_has_edges_false() {
    let graph = build_digraph(vec![0, 1], vec![]);
    assert!(!graph.edges().has_edges());
    assert!(!graph.has_edges());
}

// ============================================================================
// has_nodes
// ============================================================================

#[test]
fn test_has_nodes() {
    let graph = build_digraph(vec![0, 1], vec![]);
    assert!(graph.has_nodes());
}

// ============================================================================
// sparse_coordinates from Edges trait
// ============================================================================

#[test]
fn test_edges_sparse_coordinates() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let coords: Vec<(usize, usize)> = graph.edges().sparse_coordinates().collect();
    assert_eq!(coords, vec![(0, 1), (1, 2)]);
}

// ============================================================================
// successors from Edges trait
// ============================================================================

#[test]
fn test_edges_successors() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3)]);
    let succs: Vec<usize> = graph.edges().successors(0).collect();
    assert_eq!(succs, vec![1, 2, 3]);
}
