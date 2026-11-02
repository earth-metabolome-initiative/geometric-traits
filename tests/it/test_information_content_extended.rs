//! Extended tests for the `InformationContent` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder, InformationContent},
    traits::{EdgesBuilder, VocabularyBuilder, information_content::InformationContentError},
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
fn test_ic_simple_chain() -> Result<(), Box<dyn std::error::Error>> {
    // Chain: 0 -> 1 -> 2
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let ic = graph.information_content(&[1, 1, 1])?;

    // Root node (0) should have lowest IC (highest probability)
    // Leaf node (2) should have highest IC (lowest probability)
    assert!(ic[0] <= ic[2], "Root should have lower IC than leaf");
    Ok(())
}

#[test]
fn test_ic_diamond_dag() -> Result<(), Box<dyn std::error::Error>> {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let ic = graph.information_content(&[1, 1, 1, 1])?;

    // All IC values should be non-negative (assuming valid computation)
    for node_id in 0..4usize {
        assert!(ic[node_id].is_finite(), "IC for node {node_id} should be finite");
    }
    Ok(())
}

#[test]
fn test_ic_wrong_occurrence_size() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);

    // Too few occurrences
    let result = graph.information_content(&[1]);
    assert_eq!(
        result,
        Err(InformationContentError::UnequalOccurrenceSize { expected: 3, found: 1 })
    );

    // Too many occurrences
    let result = graph.information_content(&[1, 1, 1, 1]);
    assert_eq!(
        result,
        Err(InformationContentError::UnequalOccurrenceSize { expected: 3, found: 4 })
    );
}

#[test]
fn test_ic_sink_node_zero_occurrence() {
    // Chain: 0 -> 1 -> 2; node 2 is a sink
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    let result = graph.information_content(&[1, 1, 0]);
    assert_eq!(result, Err(InformationContentError::SinkNodeZeroOccurrence));
}

#[test]
fn test_ic_cyclic_graph_returns_not_dag() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)]);
    let result = graph.information_content(&[1, 1, 1]);
    assert_eq!(result, Err(InformationContentError::NotDag));
}

#[test]
fn test_ic_tree_structure() -> Result<(), Box<dyn std::error::Error>> {
    // Tree: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let ic = graph.information_content(&[1, 1, 1, 1, 1])?;

    // Root (0) should have lowest IC
    assert!(ic[0] <= ic[1], "Root should have lower IC than internal node");
    assert!(ic[0] <= ic[2], "Root should have lower IC than leaf");
    Ok(())
}

#[test]
fn test_ic_error_display() {
    let err = InformationContentError::NotDag;
    let msg = format!("{err}");
    assert!(msg.contains("DAG"), "Display should mention DAG");

    let err = InformationContentError::UnequalOccurrenceSize { expected: 5, found: 3 };
    let msg = format!("{err}");
    assert!(msg.contains('5'), "Display should mention expected size");
    assert!(msg.contains('3'), "Display should mention found size");

    let err = InformationContentError::SinkNodeZeroOccurrence;
    let msg = format!("{err}");
    assert!(msg.contains("Sink"), "Display should mention sink node");
}

#[test]
fn test_ic_error_equality() {
    assert_eq!(InformationContentError::NotDag, InformationContentError::NotDag);
    assert_ne!(InformationContentError::NotDag, InformationContentError::SinkNodeZeroOccurrence);
    assert_eq!(
        InformationContentError::UnequalOccurrenceSize { expected: 1, found: 2 },
        InformationContentError::UnequalOccurrenceSize { expected: 1, found: 2 }
    );
    assert_ne!(
        InformationContentError::UnequalOccurrenceSize { expected: 1, found: 2 },
        InformationContentError::UnequalOccurrenceSize { expected: 3, found: 4 }
    );
}

#[test]
fn test_ic_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(InformationContentError::NotDag);
    assert!(err.to_string().contains("DAG"));
}

#[test]
fn test_ic_single_node() -> Result<(), Box<dyn std::error::Error>> {
    let graph = build_digraph(vec![0], vec![]);
    let ic = graph.information_content(&[1])?;
    assert!(ic[0].is_finite(), "IC of single node should be finite");
    Ok(())
}
