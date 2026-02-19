//! Extended tests for TransposedEdges trait: predecessors, has_predecessor,
//! has_predecessors, in_degree, in_degrees.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

/// Type alias for the bimatrix used in tests.
type TestBiMatrix = GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>;

/// Type alias for the test graph.
type TestGraph = GenericGraph<SortedVec<usize>, TestBiMatrix>;

/// Helper to build a graph that supports transposed operations.
fn build_transposed_graph(node_count: usize, edges: Vec<(usize, usize)>) -> TestGraph {
    let nodes: Vec<usize> = (0..node_count).collect();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_count = edges.len();
    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(edge_count)
            .expected_shape((node_count, node_count))
            .edges(edges.into_iter())
            .build()
            .unwrap();
    let edges: TestBiMatrix = GenericBiMatrix2D::new(edges);
    GenericGraph::from((nodes, edges))
}

// ============================================================================
// predecessors tests
// ============================================================================

#[test]
fn test_predecessors_chain() {
    // 0 -> 1 -> 2 -> 3
    let graph = build_transposed_graph(4, vec![(0, 1), (1, 2), (2, 3)]);

    let preds: Vec<usize> = graph.predecessors(0).collect();
    assert!(preds.is_empty());

    let preds: Vec<usize> = graph.predecessors(1).collect();
    assert_eq!(preds, vec![0]);

    let preds: Vec<usize> = graph.predecessors(2).collect();
    assert_eq!(preds, vec![1]);

    let preds: Vec<usize> = graph.predecessors(3).collect();
    assert_eq!(preds, vec![2]);
}

#[test]
fn test_predecessors_diamond() {
    //   0
    //  / \
    // 1   2
    //  \ /
    //   3
    let graph = build_transposed_graph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

    let mut preds: Vec<usize> = graph.predecessors(3).collect();
    preds.sort_unstable();
    assert_eq!(preds, vec![1, 2]);

    let preds: Vec<usize> = graph.predecessors(0).collect();
    assert!(preds.is_empty());
}

#[test]
fn test_predecessors_convergent() {
    // 0 -> 3, 1 -> 3, 2 -> 3
    let graph = build_transposed_graph(4, vec![(0, 3), (1, 3), (2, 3)]);

    let mut preds: Vec<usize> = graph.predecessors(3).collect();
    preds.sort_unstable();
    assert_eq!(preds, vec![0, 1, 2]);
}

// ============================================================================
// has_predecessor tests
// ============================================================================

#[test]
fn test_has_predecessor_various() {
    let graph = build_transposed_graph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(graph.has_predecessor(1, 0));
    assert!(graph.has_predecessor(2, 0));
    assert!(graph.has_predecessor(2, 1));
    assert!(!graph.has_predecessor(0, 1));
    assert!(!graph.has_predecessor(0, 2));
}

// ============================================================================
// has_predecessors tests
// ============================================================================

#[test]
fn test_has_predecessors_chain() {
    let graph = build_transposed_graph(3, vec![(0, 1), (1, 2)]);

    assert!(!graph.has_predecessors(0));
    assert!(graph.has_predecessors(1));
    assert!(graph.has_predecessors(2));
}

#[test]
fn test_has_predecessors_isolated() {
    let graph = build_transposed_graph(3, vec![(0, 1)]);

    assert!(!graph.has_predecessors(0));
    assert!(graph.has_predecessors(1));
    assert!(!graph.has_predecessors(2));
}

// ============================================================================
// in_degree tests
// ============================================================================

#[test]
fn test_in_degree_chain() {
    let graph = build_transposed_graph(4, vec![(0, 1), (1, 2), (2, 3)]);

    assert_eq!(graph.in_degree(0), 0);
    assert_eq!(graph.in_degree(1), 1);
    assert_eq!(graph.in_degree(2), 1);
    assert_eq!(graph.in_degree(3), 1);
}

#[test]
fn test_in_degree_convergent() {
    let graph = build_transposed_graph(4, vec![(0, 3), (1, 3), (2, 3)]);

    assert_eq!(graph.in_degree(0), 0);
    assert_eq!(graph.in_degree(1), 0);
    assert_eq!(graph.in_degree(2), 0);
    assert_eq!(graph.in_degree(3), 3);
}

#[test]
fn test_in_degree_self_loop() {
    let graph = build_transposed_graph(2, vec![(0, 0), (0, 1)]);

    assert_eq!(graph.in_degree(0), 1);
    assert_eq!(graph.in_degree(1), 1);
}

// ============================================================================
// in_degrees tests
// ============================================================================

#[test]
fn test_in_degrees_chain() {
    let graph = build_transposed_graph(4, vec![(0, 1), (1, 2), (2, 3)]);

    let in_degs: Vec<usize> = graph.in_degrees().collect();
    assert_eq!(in_degs, vec![0, 1, 1, 1]);
}

#[test]
fn test_in_degrees_star() {
    let graph = build_transposed_graph(4, vec![(0, 1), (0, 2), (0, 3)]);

    let in_degs: Vec<usize> = graph.in_degrees().collect();
    assert_eq!(in_degs, vec![0, 1, 1, 1]);
}

#[test]
fn test_in_degrees_diamond() {
    let graph = build_transposed_graph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

    let in_degs: Vec<usize> = graph.in_degrees().collect();
    assert_eq!(in_degs, vec![0, 1, 1, 2]);
}

#[test]
fn test_in_degrees_no_edges() {
    let graph = build_transposed_graph(3, vec![]);

    let in_degs: Vec<usize> = graph.in_degrees().collect();
    assert_eq!(in_degs, vec![0, 0, 0]);
}

// ============================================================================
// Combined in-degree and out-degree tests
// ============================================================================

#[test]
fn test_in_and_out_degree_consistency() {
    let graph = build_transposed_graph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

    let total_out: usize = graph.out_degrees().sum();
    let total_in: usize = graph.in_degrees().sum();
    assert_eq!(total_out, total_in);
    assert_eq!(total_out, graph.number_of_edges());
}

#[test]
fn test_cycle_in_and_out_degrees() {
    // 3-cycle: 0 -> 1 -> 2 -> 0
    let graph = build_transposed_graph(3, vec![(0, 1), (1, 2), (2, 0)]);

    for node in 0..3 {
        assert_eq!(graph.in_degree(node), 1);
        assert_eq!(graph.out_degree(node), 1);
    }
}

#[test]
fn test_bidirectional_edge() {
    // 0 <-> 1
    let graph = build_transposed_graph(2, vec![(0, 1), (1, 0)]);

    assert_eq!(graph.in_degree(0), 1);
    assert_eq!(graph.in_degree(1), 1);
    assert_eq!(graph.out_degree(0), 1);
    assert_eq!(graph.out_degree(1), 1);
}
