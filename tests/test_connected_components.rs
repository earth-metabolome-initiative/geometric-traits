//! Tests for ConnectedComponents trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::{connected_components::ConnectedComponentsResult, *},
    traits::{ConnectedComponents, VocabularyBuilder},
};

/// Helper to create an undirected graph with multiple components.
fn create_graph_with_components() -> UndiGraph<usize> {
    // Component 1: nodes 0, 1, 2 (connected)
    // Component 2: nodes 3, 4 (connected)
    // Component 3: node 5 (isolated)
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
    let edge_data: Vec<(usize, usize)> = vec![
        (0, 1), // Component 1
        (1, 2), // Component 1
        (3, 4), // Component 2
    ];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(vec![0, 1, 2, 3, 4, 5].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(3)
        .expected_shape(6)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    UndiGraph::from((nodes, edges))
}

/// Helper to create a fully connected graph (single component).
fn create_fully_connected_graph() -> UndiGraph<usize> {
    // Edges must be in sorted order (row-major)
    let edge_data: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(4)
        .symbols(vec![0, 1, 2, 3].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(6)
        .expected_shape(4)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    UndiGraph::from((nodes, edges))
}

// ============================================================================
// Number of components tests
// ============================================================================

#[test]
fn test_number_of_components_multiple() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 3);
}

#[test]
fn test_number_of_components_single() {
    let graph = create_fully_connected_graph();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 1);
}

// ============================================================================
// Component size tests
// ============================================================================

#[test]
fn test_largest_component_size() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Largest component has 3 nodes (0, 1, 2)
    assert_eq!(cc.largest_component_size(), 3);
}

#[test]
fn test_smallest_component_size() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Smallest component has 1 node (5 - isolated)
    assert_eq!(cc.smallest_component_size(), 1);
}

// ============================================================================
// Component of node tests
// ============================================================================

#[test]
fn test_component_of_node() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Nodes 0, 1, 2 should be in the same component
    let comp_0: usize = cc.component_of_node(0);
    let comp_1: usize = cc.component_of_node(1);
    let comp_2: usize = cc.component_of_node(2);
    assert_eq!(comp_0, comp_1);
    assert_eq!(comp_1, comp_2);

    // Nodes 3, 4 should be in the same component (different from 0, 1, 2)
    let comp_3: usize = cc.component_of_node(3);
    let comp_4: usize = cc.component_of_node(4);
    assert_eq!(comp_3, comp_4);
    assert_ne!(comp_0, comp_3);

    // Node 5 should be in its own component
    let comp_5: usize = cc.component_of_node(5);
    assert_ne!(comp_5, comp_0);
    assert_ne!(comp_5, comp_3);
}

// ============================================================================
// Component identifiers tests
// ============================================================================

#[test]
fn test_component_identifiers() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    let identifiers: Vec<usize> = cc.component_identifiers().collect();
    assert_eq!(identifiers.len(), 6);

    // All nodes should have a valid component identifier
    for &id in &identifiers {
        assert!(id < cc.number_of_components());
    }
}

// ============================================================================
// Node IDs of component tests
// ============================================================================

#[test]
fn test_node_ids_of_component() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Find the component containing node 0
    let comp_0: usize = cc.component_of_node(0);

    // Get all node IDs in that component
    let mut node_ids: Vec<usize> = cc.node_ids_of_component(comp_0).collect();
    node_ids.sort_unstable();

    // Should contain nodes 0, 1, 2
    assert_eq!(node_ids, vec![0, 1, 2]);
}

#[test]
fn test_nodes_of_component() {
    let graph = create_graph_with_components();
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Find the component containing node 3
    let comp_3: usize = cc.component_of_node(3);

    // Get all nodes in that component
    let mut nodes: Vec<usize> = cc.nodes_of_component(comp_3).collect();
    nodes.sort_unstable();

    // Should contain nodes 3, 4
    assert_eq!(nodes, vec![3, 4]);
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_all_isolated_nodes() {
    // Graph with only isolated nodes
    let node_data: Vec<usize> = vec![0, 1, 2];
    let edge_data: Vec<(usize, usize)> = vec![];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_data.len())
        .symbols(vec![0, 1, 2].into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(3)
        .edges(edge_data.into_iter())
        .build()
        .unwrap();

    let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Each node is its own component
    assert_eq!(cc.number_of_components(), 3);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
}
