//! Integration tests for bipartite detection on undirected graphs.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, VocabularyBuilder, algorithms::bipartite_detection::BipartiteDetection,
    },
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

fn assert_valid_coloring(graph: &UndiGraph<usize>, coloring: &[u8]) {
    assert_eq!(coloring.len(), graph.number_of_nodes());
    for &color in coloring {
        assert!(color <= 1);
    }
    for node in 0..graph.number_of_nodes() {
        for neighbor in graph.neighbors(node) {
            assert_ne!(coloring[node], coloring[neighbor]);
        }
    }
}

#[test]
fn test_empty_graph_is_bipartite() {
    let graph = build_undigraph(vec![], vec![]);
    let coloring = graph.bipartite_coloring().unwrap();

    assert!(graph.is_bipartite());
    assert!(coloring.is_empty());
}

#[test]
fn test_even_cycle_is_bipartite() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    let coloring = graph.bipartite_coloring().unwrap();

    assert!(graph.is_bipartite());
    assert_valid_coloring(&graph, &coloring);
}

#[test]
fn test_disconnected_forest_is_bipartite() {
    let graph = build_undigraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (2, 3)]);
    let coloring = graph.bipartite_coloring().unwrap();

    assert!(graph.is_bipartite());
    assert_valid_coloring(&graph, &coloring);
}

#[test]
fn test_odd_cycle_is_not_bipartite() {
    let graph = build_undigraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);

    assert!(!graph.is_bipartite());
    assert!(graph.bipartite_coloring().is_none());
}

#[test]
fn test_self_loop_is_not_bipartite() {
    let graph = build_undigraph(vec![0], vec![(0, 0)]);

    assert!(!graph.is_bipartite());
    assert!(graph.bipartite_coloring().is_none());
}
