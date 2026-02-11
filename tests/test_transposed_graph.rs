//! Tests for transposed graph traits: TransposedEdges, TransposedMonoplexGraph.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{EdgesBuilder, TransposedEdges, VocabularyBuilder},
};

/// Type alias for the bimatrix used in tests.
type TestBiMatrix = GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>;

/// Type alias for the test graph.
type TestGraph = GenericGraph<SortedVec<usize>, TestBiMatrix>;

/// Helper to create a simple graph for testing.
fn create_test_graph() -> TestGraph {
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
    let edge_data: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(6)
            .expected_shape((5, 5))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let edges: TestBiMatrix = GenericBiMatrix2D::new(edges);

    GenericGraph::from((nodes, edges))
}

// ============================================================================
// TransposedEdges tests
// ============================================================================

#[test]
fn test_predecessors() {
    let graph = create_test_graph();

    // Node 0 has no predecessors (it's a root)
    let preds_0: Vec<usize> = graph.predecessors(0).collect();
    assert!(preds_0.is_empty());

    // Node 1 has predecessor 0
    let preds_1: Vec<usize> = graph.predecessors(1).collect();
    assert_eq!(preds_1, vec![0]);

    // Node 2 has predecessors 0 and 1
    let preds_2: Vec<usize> = graph.predecessors(2).collect();
    assert_eq!(preds_2, vec![0, 1]);

    // Node 3 has predecessors 1 and 2
    let preds_3: Vec<usize> = graph.predecessors(3).collect();
    assert_eq!(preds_3, vec![1, 2]);

    // Node 4 has predecessor 3
    let preds_4: Vec<usize> = graph.predecessors(4).collect();
    assert_eq!(preds_4, vec![3]);
}

#[test]
fn test_has_predecessor() {
    let graph = create_test_graph();

    // Check specific predecessor relationships
    assert!(graph.has_predecessor(1, 0));
    assert!(!graph.has_predecessor(0, 1));

    assert!(graph.has_predecessor(2, 0));
    assert!(graph.has_predecessor(2, 1));
    assert!(!graph.has_predecessor(2, 3));

    assert!(graph.has_predecessor(3, 1));
    assert!(graph.has_predecessor(3, 2));
    assert!(!graph.has_predecessor(3, 0));

    assert!(graph.has_predecessor(4, 3));
    assert!(!graph.has_predecessor(4, 0));
}

#[test]
fn test_has_predecessors() {
    let graph = create_test_graph();

    // Node 0 has no predecessors
    assert!(!graph.has_predecessors(0));

    // All other nodes have predecessors
    assert!(graph.has_predecessors(1));
    assert!(graph.has_predecessors(2));
    assert!(graph.has_predecessors(3));
    assert!(graph.has_predecessors(4));
}

#[test]
fn test_in_degree() {
    let graph = create_test_graph();

    assert_eq!(graph.in_degree(0), 0);
    assert_eq!(graph.in_degree(1), 1);
    assert_eq!(graph.in_degree(2), 2);
    assert_eq!(graph.in_degree(3), 2);
    assert_eq!(graph.in_degree(4), 1);
}

#[test]
fn test_in_degrees() {
    let graph = create_test_graph();

    let in_degrees: Vec<usize> = graph.in_degrees().collect();
    assert_eq!(in_degrees, vec![0, 1, 2, 2, 1]);
}

// ============================================================================
// TransposedEdges trait methods on edges directly
// ============================================================================

#[test]
fn test_transposed_edges_direct() {
    let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 3)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(4)
            .expected_shape((4, 4))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    let bimatrix: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
        GenericBiMatrix2D::new(edges);

    // Test predecessors via the TransposedEdges trait
    let preds: Vec<usize> = TransposedEdges::predecessors(&bimatrix, 2).collect();
    assert_eq!(preds, vec![0, 1]);

    // Test has_predecessor
    assert!(TransposedEdges::has_predecessor(&bimatrix, 2, 0));
    assert!(TransposedEdges::has_predecessor(&bimatrix, 2, 1));
    assert!(!TransposedEdges::has_predecessor(&bimatrix, 2, 3));

    // Test has_predecessors
    assert!(!TransposedEdges::has_predecessors(&bimatrix, 0));
    assert!(TransposedEdges::has_predecessors(&bimatrix, 1));
    assert!(TransposedEdges::has_predecessors(&bimatrix, 2));
    assert!(TransposedEdges::has_predecessors(&bimatrix, 3));

    // Test in_degree
    assert_eq!(TransposedEdges::in_degree(&bimatrix, 0), 0);
    assert_eq!(TransposedEdges::in_degree(&bimatrix, 1), 1);
    assert_eq!(TransposedEdges::in_degree(&bimatrix, 2), 2);
    assert_eq!(TransposedEdges::in_degree(&bimatrix, 3), 1);

    // Test in_degrees
    let in_degrees: Vec<usize> = TransposedEdges::in_degrees(&bimatrix).collect();
    assert_eq!(in_degrees, vec![0, 1, 2, 1]);
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_single_node_no_edges() {
    let nodes: Vec<usize> = vec![0];
    let edges: Vec<(usize, usize)> = vec![];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(0)
            .expected_shape((1, 1))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    let edges: TestBiMatrix = GenericBiMatrix2D::new(edges);

    let graph: TestGraph = GenericGraph::from((nodes, edges));

    assert!(!graph.has_predecessors(0));
    assert_eq!(graph.in_degree(0), 0);
    let preds: Vec<usize> = graph.predecessors(0).collect();
    assert!(preds.is_empty());
}

#[test]
fn test_self_loop_predecessor() {
    let nodes: Vec<usize> = vec![0, 1];
    let edges: Vec<(usize, usize)> = vec![(0, 0), (0, 1)]; // Self-loop on 0

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(2)
            .expected_shape((2, 2))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    let edges: TestBiMatrix = GenericBiMatrix2D::new(edges);

    let graph: TestGraph = GenericGraph::from((nodes, edges));

    // Node 0 has itself as predecessor due to self-loop
    assert!(graph.has_predecessor(0, 0));
    assert!(graph.has_predecessors(0));
    assert_eq!(graph.in_degree(0), 1);

    // Node 1 has node 0 as predecessor
    assert!(graph.has_predecessor(1, 0));
    assert_eq!(graph.in_degree(1), 1);
}
