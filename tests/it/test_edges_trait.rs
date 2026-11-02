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
    assert!(Edges::has_successors(graph.edges(), 0));
    assert!(Edges::has_successors(graph.edges(), 1));
}

#[test]
fn test_has_successors_false() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(!Edges::has_successors(graph.edges(), 2));
}

// ============================================================================
// has_successor tests
// ============================================================================

#[test]
fn test_has_successor_true() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(Edges::has_successor(graph.edges(), 0, 1));
}

#[test]
fn test_has_successor_false() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(!Edges::has_successor(graph.edges(), 0, 2));
    assert!(!Edges::has_successor(graph.edges(), 2, 0));
}

// ============================================================================
// out_degree tests
// ============================================================================

#[test]
fn test_out_degree() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3), (1, 2)]);
    assert_eq!(Edges::out_degree(graph.edges(), 0), 3);
    assert_eq!(Edges::out_degree(graph.edges(), 1), 1);
    assert_eq!(Edges::out_degree(graph.edges(), 2), 0);
    assert_eq!(Edges::out_degree(graph.edges(), 3), 0);
}

// ============================================================================
// out_degrees tests
// ============================================================================

#[test]
fn test_out_degrees() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3)]);
    let degrees: Vec<usize> = Edges::out_degrees(graph.edges()).collect();
    assert_eq!(degrees, vec![2, 1, 0, 0]);
}

// ============================================================================
// number_of_edges / has_edges
// ============================================================================

#[test]
fn test_number_of_edges() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert_eq!(Edges::number_of_edges(graph.edges()), 2);
    assert_eq!(MonoplexGraph::number_of_edges(&graph), 2);
}

#[test]
fn test_has_edges_true() {
    let graph = build_digraph(vec![0, 1], vec![(0, 1)]);
    assert!(Edges::has_edges(graph.edges()));
    assert!(Graph::has_edges(&graph));
}

#[test]
fn test_has_edges_false() {
    let graph = build_digraph(vec![0, 1], vec![]);
    assert!(!Edges::has_edges(graph.edges()));
    assert!(!Graph::has_edges(&graph));
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
    let coords: Vec<(usize, usize)> = Edges::sparse_coordinates(graph.edges()).collect();
    assert_eq!(coords, vec![(0, 1), (1, 2)]);
}

// ============================================================================
// successors from Edges trait
// ============================================================================

#[test]
fn test_edges_successors() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3)]);
    let succs: Vec<usize> = Edges::successors(graph.edges(), 0).collect();
    assert_eq!(succs, vec![1, 2, 3]);
}
