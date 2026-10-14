//! Tests for algorithm edge cases to increase coverage on resnik, wu_palmer,
//! tarjan, connected_components, and information_content.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, InformationContent, Resnik, ScalarSimilarity, VocabularyBuilder, WuPalmer,
        algorithms::connected_components::{ConnectedComponents, ConnectedComponentsError},
    },
};

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

fn build_undigraph(node_list: Vec<usize>, edge_list: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let num_nodes = node_list.len();
    let num_edges = edge_list.len();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(node_list.into_iter().enumerate())
        .build()
        .unwrap();
    let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(nodes.len())
        .edges(edge_list.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

// ============================================================================
// WuPalmer edge cases
// ============================================================================

#[test]
fn test_wu_palmer_diamond_dag() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let wp = graph.wu_palmer().unwrap();
    // Nodes 1 and 2 are siblings under root 0
    let sim = wp.similarity(&1, &2);
    assert!((0.0..=1.0).contains(&sim));
    // Node 3 is deep, 1 is shallow
    let sim13 = wp.similarity(&1, &3);
    assert!((0.0..=1.0).contains(&sim13));
}

#[test]
fn test_wu_palmer_multiple_roots() {
    // Two separate trees: 0 -> 1, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (2, 3)]);
    let wp = graph.wu_palmer().unwrap();
    // Nodes from different trees
    let sim = wp.similarity(&1, &3);
    // Should be 0 since they're in different components
    assert!(sim >= 0.0);
}

#[test]
fn test_wu_palmer_wide_tree() {
    // Root 0 with many children: 0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    let wp = graph.wu_palmer().unwrap();
    // All children should have same similarity to each other
    let sim12 = wp.similarity(&1, &2);
    let sim34 = wp.similarity(&3, &4);
    assert!((sim12 - sim34).abs() < f64::EPSILON);
}

#[test]
fn test_wu_palmer_deep_chain() {
    // Chain: 0 -> 1 -> 2 -> 3 -> 4
    let graph = build_digraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let wp = graph.wu_palmer().unwrap();
    // Nearby nodes should have higher similarity than distant ones
    let sim34 = wp.similarity(&3, &4);
    let sim14 = wp.similarity(&1, &4);
    assert!(sim34 > sim14);
}

// ============================================================================
// Resnik edge cases
// ============================================================================

#[test]
fn test_resnik_diamond_dag() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let occurrences = vec![1, 1, 1, 2];
    let resnik = graph.resnik(&occurrences).unwrap();
    // Siblings 1 and 2 under common ancestor 0
    let sim12 = resnik.similarity(&1, &2);
    assert!(sim12 >= 0.0);
    // Same node should have max similarity
    let sim11 = resnik.similarity(&1, &1);
    assert!(sim11 >= sim12);
}

#[test]
fn test_resnik_multiple_roots() {
    // Two roots: 0 -> 2, 1 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 2), (1, 3)]);
    let occurrences = vec![1, 1, 2, 2];
    let resnik = graph.resnik(&occurrences).unwrap();
    // Nodes from different components
    let sim23 = resnik.similarity(&2, &3);
    assert!(sim23 >= 0.0);
}

#[test]
fn test_resnik_deep_chain() {
    // Chain: 0 -> 1 -> 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    let occurrences = vec![1, 1, 1, 3];
    let resnik = graph.resnik(&occurrences).unwrap();
    // Closer nodes have higher IC for common ancestor
    let sim23 = resnik.similarity(&2, &3);
    let sim03 = resnik.similarity(&0, &3);
    assert!(sim23 >= sim03);
}

// ============================================================================
// Information Content edge cases
// ============================================================================

#[test]
fn test_ic_wrong_occurrences_length() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2)]);
    let result = graph.information_content(&[1, 1]); // 2 instead of 3
    assert!(result.is_err());
}

#[test]
fn test_ic_sink_zero_occurrence() {
    let graph = build_digraph(vec![0, 1, 2], vec![(0, 1), (0, 2)]);
    // Nodes 1 and 2 are sinks, giving 0 occurrence for node 1
    let result = graph.information_content(&[1, 0, 1]);
    assert!(result.is_err());
}

#[test]
fn test_ic_valid_diamond() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let graph = build_digraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let result = graph.information_content(&[1, 1, 1, 2]);
    assert!(result.is_ok());
    let ic = result.unwrap();
    // IC values should be finite
    for i in 0..4 {
        assert!(ic[i].is_finite(), "IC[{i}] = {} should be finite", ic[i]);
    }
}

// ============================================================================
// Connected Components edge cases
// ============================================================================

#[test]
fn test_cc_single_node() {
    let graph = build_undigraph(vec![0], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 1);
}

#[test]
fn test_cc_all_isolated() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 4);
    assert_eq!(cc.smallest_component_size(), 1);
    assert_eq!(cc.largest_component_size(), 1);
}

#[test]
fn test_cc_two_components() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (2, 3)]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 2);
}

#[test]
fn test_cc_error_display() {
    let e = ConnectedComponentsError::TooManyComponents;
    let s = format!("{e}");
    assert!(s.contains("too many"));
}

// ============================================================================
// Tarjan edge cases
// ============================================================================

#[test]
fn test_tarjan_single_node_no_edges() {
    let m: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(1)
        .edges(std::iter::empty())
        .build()
        .unwrap();
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], vec![0]);
}

#[test]
fn test_tarjan_two_disjoint_cycles() {
    // Cycle 1: 0 -> 1 -> 0
    // Cycle 2: 2 -> 3 -> 2
    let m: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(4)
        .expected_shape(4)
        .edges(vec![(0, 1), (1, 0), (2, 3), (3, 2)].into_iter())
        .build()
        .unwrap();
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 2);
}

#[test]
fn test_tarjan_complex_graph() {
    // Graph with mix of SCCs and non-SCC nodes
    // SCC1: 0 -> 1 -> 2 -> 0
    // Node 3: 2 -> 3 (no back edge)
    // SCC2: 4 -> 5 -> 4
    let mut edges = vec![(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 4)];
    edges.sort_unstable();
    let m: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(6)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    // Should find: {4,5}, {3}, {0,1,2}
    assert_eq!(sccs.len(), 3);
}

#[test]
fn test_tarjan_all_nodes_one_scc() {
    // Complete graph K3: all nodes in one SCC
    let m: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(6)
        .expected_shape(3)
        .edges(vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)].into_iter())
        .build()
        .unwrap();
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 3);
}
