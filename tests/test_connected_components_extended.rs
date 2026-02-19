//! Extended tests for ConnectedComponents covering edge cases,
//! node_ids_of_component, nodes_of_component, and component_identifiers.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::{connected_components::ConnectedComponentsResult, *},
    traits::{ConnectedComponents, VocabularyBuilder},
};

/// Helper to build an undirected graph.
fn build_undi_graph(node_count: usize, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let nodes: Vec<usize> = (0..node_count).collect();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_count = edges.len();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

// ============================================================================
// Single node graph
// ============================================================================

#[test]
fn test_single_node_graph() {
    let graph = build_undi_graph(1, vec![]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
    assert_eq!(cc.component_of_node(0), 0);
}

// ============================================================================
// Two node connected graph
// ============================================================================

#[test]
fn test_two_node_connected() {
    let graph = build_undi_graph(2, vec![(0, 1)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 2);
    assert_eq!(cc.component_of_node(0), cc.component_of_node(1));
}

// ============================================================================
// Two node disconnected graph
// ============================================================================

#[test]
fn test_two_node_disconnected() {
    let graph = build_undi_graph(2, vec![]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 2);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
    assert_ne!(cc.component_of_node(0), cc.component_of_node(1));
}

// ============================================================================
// Path graph
// ============================================================================

#[test]
fn test_path_graph_single_component() {
    // 0--1--2--3--4 (single component)
    let graph = build_undi_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 5);

    // All nodes in same component
    let c = cc.component_of_node(0);
    for i in 1..5 {
        assert_eq!(cc.component_of_node(i), c);
    }
}

// ============================================================================
// node_ids_of_component tests
// ============================================================================

#[test]
fn test_node_ids_of_all_components() {
    // 0--1, 2--3, 4 isolated
    let graph = build_undi_graph(5, vec![(0, 1), (2, 3)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 3);

    // Collect all components and verify no duplicates
    let mut all_nodes: Vec<usize> = Vec::new();
    for comp_id in 0..cc.number_of_components() {
        let nodes: Vec<usize> = cc.node_ids_of_component(comp_id).collect();
        all_nodes.extend(nodes);
    }
    all_nodes.sort_unstable();
    assert_eq!(all_nodes, vec![0, 1, 2, 3, 4]);
}

// ============================================================================
// nodes_of_component tests
// ============================================================================

#[test]
fn test_nodes_of_component_symbols() {
    // 3 nodes, 2 connected
    let graph = build_undi_graph(3, vec![(0, 1)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    // Component of node 0 should contain symbols 0 and 1
    let comp_0 = cc.component_of_node(0);
    let mut nodes: Vec<usize> = cc.nodes_of_component(comp_0).collect();
    nodes.sort_unstable();
    assert_eq!(nodes, vec![0, 1]);

    // Component of node 2 should contain only symbol 2
    let comp_2 = cc.component_of_node(2);
    let nodes: Vec<usize> = cc.nodes_of_component(comp_2).collect();
    assert_eq!(nodes, vec![2]);
}

// ============================================================================
// component_identifiers tests
// ============================================================================

#[test]
fn test_component_identifiers_length() {
    let graph = build_undi_graph(6, vec![(0, 1), (2, 3), (4, 5)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    let ids: Vec<usize> = cc.component_identifiers().collect();
    assert_eq!(ids.len(), 6);

    // Paired nodes should share identifiers
    assert_eq!(ids[0], ids[1]);
    assert_eq!(ids[2], ids[3]);
    assert_eq!(ids[4], ids[5]);

    // Different pairs should have different identifiers
    assert_ne!(ids[0], ids[2]);
    assert_ne!(ids[0], ids[4]);
    assert_ne!(ids[2], ids[4]);
}

// ============================================================================
// Large component with complex connectivity
// ============================================================================

#[test]
fn test_cycle_graph_single_component() {
    // Cycle: 0--1--2--3--4--0
    let graph = build_undi_graph(5, vec![(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 5);
    assert_eq!(cc.smallest_component_size(), 5);
}

// ============================================================================
// Mixed sizes
// ============================================================================

#[test]
fn test_mixed_component_sizes() {
    // Component of size 4: 0--1--2--3
    // Component of size 2: 4--5
    // Component of size 1: 6
    let graph = build_undi_graph(7, vec![(0, 1), (1, 2), (2, 3), (4, 5)]);
    let cc: ConnectedComponentsResult<'_, _, usize> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 3);
    assert_eq!(cc.largest_component_size(), 4);
    assert_eq!(cc.smallest_component_size(), 1);
}

// ============================================================================
// Marker type u8
// ============================================================================

#[test]
fn test_connected_components_u8_marker() {
    let graph = build_undi_graph(4, vec![(0, 1), (2, 3)]);
    let cc: ConnectedComponentsResult<'_, _, u8> = graph.connected_components().unwrap();

    assert_eq!(cc.number_of_components(), 2_u8);
}
