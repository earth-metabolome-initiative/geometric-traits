//! Integration tests for tree and forest detection on undirected graphs.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder, algorithms::tree_detection::TreeDetection},
};

fn build_undigraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let num_nodes = nodes.len();
    let mut edges: Vec<(usize, usize)> = edges
        .into_iter()
        .map(|(src, dst)| if src <= dst { (src, dst) } else { (dst, src) })
        .collect();
    edges.sort_unstable();
    let num_edges = edges.len();
    let node_vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_mat: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(node_vocab.len())
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((node_vocab, edge_mat))
}

#[test]
fn test_empty_graph_is_forest_but_not_tree() {
    let graph = build_undigraph(vec![], vec![]);
    assert!(graph.is_forest());
    assert!(!graph.is_tree());
}

#[test]
fn test_singleton_graph_is_tree_and_forest() {
    let graph = build_undigraph(vec![0], vec![]);
    assert!(graph.is_forest());
    assert!(graph.is_tree());
}

#[test]
fn test_path_graph_is_tree() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    assert!(graph.is_forest());
    assert!(graph.is_tree());
}

#[test]
fn test_disconnected_acyclic_graph_is_forest_but_not_tree() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (2, 3)]);
    assert!(graph.is_forest());
    assert!(!graph.is_tree());
}

#[test]
fn test_cycle_graph_is_neither_tree_nor_forest() {
    let graph = build_undigraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    assert!(!graph.is_forest());
    assert!(!graph.is_tree());
}

#[test]
fn test_self_loop_is_neither_tree_nor_forest() {
    let graph = build_undigraph(vec![0], vec![(0, 0)]);
    assert!(!graph.is_forest());
    assert!(!graph.is_tree());
}
