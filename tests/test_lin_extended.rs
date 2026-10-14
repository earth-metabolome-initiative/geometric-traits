//! Extended tests for the `Lin` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder, Lin},
    traits::{EdgesBuilder, MonopartiteGraph, ScalarSimilarity, VocabularyBuilder},
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
fn test_lin_self_similarity_on_tree() -> Result<(), Box<dyn std::error::Error>> {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2), (1, 2)]);
    let lin = graph.lin(&[1, 1, 1])?;

    for nodeid in graph.node_ids() {
        let self_sim = lin.similarity(&nodeid, &nodeid);
        assert!(self_sim > 0.99, "Self-similarity of node {nodeid} should be ~1.0, got {self_sim}");
    }
    Ok(())
}

#[test]
fn test_lin_different_nodes_less_than_one() -> Result<(), Box<dyn std::error::Error>> {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2), (1, 2)]);
    let lin = graph.lin(&[1, 1, 1])?;

    let sim_0_1 = lin.similarity(&0, &1);
    assert!(sim_0_1 < 0.99, "Cross-node similarity should be < 1.0, got {sim_0_1}");
    assert!(sim_0_1 >= 0.0, "Similarity should be non-negative, got {sim_0_1}");
    Ok(())
}

#[test]
fn test_lin_symmetry() -> Result<(), Box<dyn std::error::Error>> {
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let lin = graph.lin(&[1, 1, 1, 1])?;

    let sim_1_2 = lin.similarity(&1, &2);
    let sim_2_1 = lin.similarity(&2, &1);
    assert!((sim_1_2 - sim_2_1).abs() < 1e-10, "Lin should be symmetric: {sim_1_2} vs {sim_2_1}");
    Ok(())
}

#[test]
fn test_lin_on_cycle_returns_error() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    let result = graph.lin(&[1, 1, 1]);
    assert!(result.is_err(), "Lin should fail on cyclic graph");
}

#[test]
fn test_lin_wrong_occurrence_size() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let result = graph.lin(&[1, 1]);
    assert!(result.is_err(), "Lin should fail with mismatched occurrence size");
}

#[test]
fn test_lin_tree_with_varying_occurrences() -> Result<(), Box<dyn std::error::Error>> {
    // Tree: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let lin = graph.lin(&[1, 2, 1, 3, 2])?;

    // Siblings (3, 4) under node 1 should have positive similarity
    let sim_3_4 = lin.similarity(&3, &4);
    assert!(sim_3_4 >= 0.0, "Sibling similarity should be >= 0, got {sim_3_4}");
    Ok(())
}

#[test]
fn test_lin_chain() -> Result<(), Box<dyn std::error::Error>> {
    // Chain: 0 -> 1 -> 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    let lin = graph.lin(&[1, 1, 1, 1])?;

    let sim_0_1 = lin.similarity(&0, &1);
    let sim_0_3 = lin.similarity(&0, &3);

    // Closer nodes should generally have higher Lin similarity
    assert!(
        sim_0_1 >= sim_0_3,
        "Closer nodes should have higher similarity: {sim_0_1} vs {sim_0_3}"
    );
    Ok(())
}
