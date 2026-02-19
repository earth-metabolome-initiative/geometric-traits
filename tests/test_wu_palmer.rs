//! Test submodule for the `WuPalmer` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder},
    traits::{EdgesBuilder, ScalarSimilarity, VocabularyBuilder, WuPalmer},
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
fn test_wu_palmer_self_similarity() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2), (1, 2)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    // Self-similarity must be 1.0
    for node_id in 0..3usize {
        let sim = wu_palmer.similarity(&node_id, &node_id);
        assert!(sim > 0.99, "Self-similarity of node {node_id} should be ~1.0, got {sim}");
    }
}

#[test]
fn test_wu_palmer_different_nodes() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2), (1, 2)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    let sim_0_1 = wu_palmer.similarity(&0, &1);
    assert!(sim_0_1 < 0.99, "Similarity of different nodes should be < 1.0, got {sim_0_1}");
    assert!(sim_0_1 >= 0.0, "Similarity should be non-negative, got {sim_0_1}");
}

#[test]
fn test_wu_palmer_on_cycle_returns_error() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    let result = graph.wu_palmer();
    assert!(result.is_err(), "Wu-Palmer should fail on cyclic graph");
}

#[test]
fn test_wu_palmer_chain() {
    // Chain: 0 -> 1 -> 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    // Closer nodes should have higher similarity
    let sim_1_2 = wu_palmer.similarity(&1, &2);
    let sim_1_3 = wu_palmer.similarity(&1, &3);
    assert!(
        sim_1_2 >= sim_1_3,
        "Adjacent nodes should be more similar than distant ones: {sim_1_2} vs {sim_1_3}"
    );
}

#[test]
fn test_wu_palmer_diamond() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    // Nodes 1 and 2 are siblings under root 0
    let sim_1_2 = wu_palmer.similarity(&1, &2);
    assert!(
        (0.0..=1.0).contains(&sim_1_2),
        "Sibling similarity should be in [0, 1], got {sim_1_2}"
    );
}

#[test]
fn test_wu_palmer_symmetry() {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    let sim_1_2 = wu_palmer.similarity(&1, &2);
    let sim_2_1 = wu_palmer.similarity(&2, &1);
    assert!(
        (sim_1_2 - sim_2_1).abs() < 1e-10,
        "Wu-Palmer should be symmetric: {sim_1_2} vs {sim_2_1}"
    );
}

#[test]
fn test_wu_palmer_single_edge() {
    let graph = build_digraph(vec![0, 1], vec![(0, 1)]);
    let wu_palmer = graph.wu_palmer().unwrap();

    let sim = wu_palmer.similarity(&0, &1);
    assert!((0.0..=1.0).contains(&sim), "Similarity should be in [0, 1], got {sim}");
}
