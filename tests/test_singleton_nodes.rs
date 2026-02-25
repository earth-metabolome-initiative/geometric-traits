//! Test submodule for the `SingletonNodes` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder, MonopartiteGraph, MonoplexGraph},
    traits::{EdgesBuilder, SingletonNodes, VocabularyBuilder},
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
fn test_singleton_nodes_basic() {
    // Nodes 0 and 1 are connected; nodes 2 and 3 are singletons.
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1)]);
    assert_eq!(graph.singleton_nodes(), vec![2, 3]);
}

#[test]
fn test_singleton_nodes_all_connected() {
    // All nodes participate in edges: no singletons.
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert_eq!(graph.singleton_nodes(), Vec::<usize>::new());
}

#[test]
fn test_singleton_nodes_all_singletons() {
    // No edges at all: every node is a singleton.
    let graph = build_digraph(vec![0, 1, 2, 3], vec![]);
    assert_eq!(graph.singleton_nodes(), vec![0, 1, 2, 3]);
}

#[test]
fn test_singleton_nodes_self_loop_not_singleton() {
    // A self-loop means the node has both a successor and a predecessor (itself).
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 0)]);
    // Node 0 has a self-loop so it's not a singleton; nodes 1, 2 are singletons.
    assert_eq!(graph.singleton_nodes(), vec![1, 2]);
}

#[test]
fn test_singleton_nodes_single_node_no_edges() {
    let graph = build_digraph(vec![0], vec![]);
    assert_eq!(graph.singleton_nodes(), vec![0]);
}

#[test]
fn test_singleton_nodes_chain() {
    // Chain: 0 -> 1 -> 2 -> 3; node 4 is a singleton.
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(graph.singleton_nodes(), vec![4]);
}

#[test]
fn test_singleton_nodes_star_topology() {
    // Node 0 connects to all others: no singletons.
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    assert_eq!(graph.singleton_nodes(), Vec::<usize>::new());
}

#[test]
fn test_singleton_nodes_mixed() {
    // Edges connect 0->1 and 2->3; node 4 is a singleton.
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (2, 3)]);
    assert_eq!(graph.singleton_nodes(), vec![4]);
}

#[test]
fn test_no_singleton_nodes() {
    // Fully connected cycle graph: 0->1->2->3->0, all nodes have both incoming and
    // outgoing edges.
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    assert_eq!(graph.number_of_nodes(), 4);
    assert_eq!(graph.number_of_edges(), 4);
    assert_eq!(graph.singleton_nodes(), Vec::new());
}

#[test]
fn test_all_singleton_nodes() {
    // Graph with nodes but no edges: every node is a singleton.
    let graph = build_digraph(vec![0, 1, 2, 3], vec![]);
    assert_eq!(graph.number_of_nodes(), 4);
    assert_eq!(graph.number_of_edges(), 0);
    assert_eq!(graph.singleton_nodes(), vec![0, 1, 2, 3]);
}

#[test]
fn test_some_singleton_nodes() {
    // Mixed graph: nodes 0 and 1 are connected, nodes 2 and 3 are isolated.
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1)]);
    assert_eq!(graph.number_of_nodes(), 4);
    assert_eq!(graph.number_of_edges(), 1);
    assert_eq!(graph.singleton_nodes(), vec![2, 3]);
}

#[test]
fn test_single_node_no_edges() {
    // Edge case: a single-node graph with no edges. The lone node is a singleton.
    let graph = build_digraph(vec![0], vec![]);
    assert_eq!(graph.number_of_nodes(), 1);
    assert_eq!(graph.number_of_edges(), 0);
    assert_eq!(graph.singleton_nodes(), vec![0]);
}
