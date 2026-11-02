//! Test submodule for the `Resnik` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder},
    traits::{EdgesBuilder, Resnik, ScalarSimilarity, VocabularyBuilder},
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
fn test_resnik_simple_chain() -> Result<(), Box<dyn std::error::Error>> {
    // Simple chain: 0 -> 1 -> 2
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let resnik = graph.resnik(&[1, 1, 1])?;

    // Self-similarity should be non-negative
    let self_sim = resnik.similarity(&0, &0);
    assert!(self_sim >= 0.0, "Self-similarity should be >= 0, got {self_sim}");
    Ok(())
}

#[test]
fn test_resnik_diamond_dag() -> Result<(), Box<dyn std::error::Error>> {
    // Diamond DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let resnik = graph.resnik(&[1, 1, 1, 1])?;

    // Nodes 1 and 2 share common ancestor 0
    let sim_1_2 = resnik.similarity(&1, &2);
    assert!(sim_1_2 >= 0.0, "Similarity should be >= 0, got {sim_1_2}");

    // Nodes that are closer in the DAG should have higher similarity
    let sim_1_3 = resnik.similarity(&1, &3);
    assert!(sim_1_3 >= 0.0, "Similarity should be >= 0, got {sim_1_3}");
    Ok(())
}

#[test]
fn test_resnik_with_cycle_returns_error() {
    // Graph with a cycle: not a DAG
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    let result = graph.resnik(&[1, 1, 1]);
    assert!(result.is_err(), "Resnik should fail on cyclic graph");
}

#[test]
fn test_resnik_wrong_occurrence_length() {
    // Occurrence vector length does not match number of nodes
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let result = graph.resnik(&[1, 1]);
    assert!(result.is_err(), "Resnik should fail with mismatched occurrence size");
}

#[test]
fn test_resnik_tree_structure() -> Result<(), Box<dyn std::error::Error>> {
    // Tree: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let resnik = graph.resnik(&[1, 1, 1, 1, 1])?;

    // Nodes 3 and 4 share ancestor 1 (closer) and 0 (further)
    let sim_3_4 = resnik.similarity(&3, &4);
    // Nodes 2 and 3 share only ancestor 0
    let sim_2_3 = resnik.similarity(&2, &3);

    // Siblings (3,4) should have at least as high similarity as cousins (2,3)
    assert!(
        sim_3_4 >= sim_2_3,
        "Siblings should be at least as similar as cousins: {sim_3_4} vs {sim_2_3}"
    );
    Ok(())
}

#[test]
fn test_resnik_single_node() -> Result<(), Box<dyn std::error::Error>> {
    let graph = build_digraph(vec![0], vec![]);
    let resnik = graph.resnik(&[1])?;
    let sim = resnik.similarity(&0, &0);
    assert!(sim >= 0.0, "Self-similarity of single node should be >= 0");
    Ok(())
}
