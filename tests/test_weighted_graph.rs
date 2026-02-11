//! Tests for weighted graph traits: WeightedEdges, WeightedMonoplexGraph.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder, WeightedEdges},
};

/// Helper to create a simple weighted graph for testing.
fn create_weighted_graph() -> GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>>
{
    let nodes: Vec<usize> = vec![0, 1, 2, 3];
    let edge_data: Vec<(usize, usize, f64)> =
        vec![(0, 1, 1.0), (0, 2, 2.5), (1, 2, 0.5), (1, 3, 3.0), (2, 3, 1.5)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(5)
            .expected_shape((4, 4))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    GenericGraph::from((nodes, edges))
}

// ============================================================================
// WeightedEdges tests
// ============================================================================

#[test]
fn test_successor_weights() {
    let graph = create_weighted_graph();

    // Node 0 has successors 1 (weight 1.0) and 2 (weight 2.5)
    let weights_0: Vec<f64> = graph.successor_weights(0).collect();
    assert_eq!(weights_0.len(), 2);
    assert!((weights_0[0] - 1.0).abs() < f64::EPSILON);
    assert!((weights_0[1] - 2.5).abs() < f64::EPSILON);

    // Node 1 has successors 2 (weight 0.5) and 3 (weight 3.0)
    let weights_1: Vec<f64> = graph.successor_weights(1).collect();
    assert_eq!(weights_1.len(), 2);
    assert!((weights_1[0] - 0.5).abs() < f64::EPSILON);
    assert!((weights_1[1] - 3.0).abs() < f64::EPSILON);

    // Node 2 has successor 3 (weight 1.5)
    let weights_2: Vec<f64> = graph.successor_weights(2).collect();
    assert_eq!(weights_2.len(), 1);
    assert!((weights_2[0] - 1.5).abs() < f64::EPSILON);

    // Node 3 has no successors
    let weights_3: Vec<f64> = graph.successor_weights(3).collect();
    assert!(weights_3.is_empty());
}

#[test]
fn test_max_successor_weight() {
    let graph = create_weighted_graph();

    // Node 0: max weight is 2.5 (to node 2)
    let max_0 = graph.max_successor_weight(0);
    assert!(max_0.is_some());
    assert!((max_0.unwrap() - 2.5).abs() < f64::EPSILON);

    // Node 1: max weight is 3.0 (to node 3)
    let max_1 = graph.max_successor_weight(1);
    assert!(max_1.is_some());
    assert!((max_1.unwrap() - 3.0).abs() < f64::EPSILON);

    // Node 2: max weight is 1.5 (only one successor)
    let max_2 = graph.max_successor_weight(2);
    assert!(max_2.is_some());
    assert!((max_2.unwrap() - 1.5).abs() < f64::EPSILON);

    // Node 3: no successors
    let max_3 = graph.max_successor_weight(3);
    assert!(max_3.is_none());
}

#[test]
fn test_min_successor_weight() {
    let graph = create_weighted_graph();

    // Node 0: min weight is 1.0 (to node 1)
    let min_0 = graph.min_successor_weight(0);
    assert!(min_0.is_some());
    assert!((min_0.unwrap() - 1.0).abs() < f64::EPSILON);

    // Node 1: min weight is 0.5 (to node 2)
    let min_1 = graph.min_successor_weight(1);
    assert!(min_1.is_some());
    assert!((min_1.unwrap() - 0.5).abs() < f64::EPSILON);

    // Node 2: min weight is 1.5 (only one successor)
    let min_2 = graph.min_successor_weight(2);
    assert!(min_2.is_some());
    assert!((min_2.unwrap() - 1.5).abs() < f64::EPSILON);

    // Node 3: no successors
    let min_3 = graph.min_successor_weight(3);
    assert!(min_3.is_none());
}

#[test]
fn test_max_successor_weight_and_id() {
    let graph = create_weighted_graph();

    // Node 0: max weight is 2.5 to node 2
    let max_0 = graph.max_successor_weight_and_id(0);
    assert!(max_0.is_some());
    let (weight, id) = max_0.unwrap();
    assert!((weight - 2.5).abs() < f64::EPSILON);
    assert_eq!(id, 2);

    // Node 1: max weight is 3.0 to node 3
    let max_1 = graph.max_successor_weight_and_id(1);
    assert!(max_1.is_some());
    let (weight, id) = max_1.unwrap();
    assert!((weight - 3.0).abs() < f64::EPSILON);
    assert_eq!(id, 3);

    // Node 3: no successors
    let max_3 = graph.max_successor_weight_and_id(3);
    assert!(max_3.is_none());
}

#[test]
fn test_min_successor_weight_and_id() {
    let graph = create_weighted_graph();

    // Node 0: min weight is 1.0 to node 1
    let min_0 = graph.min_successor_weight_and_id(0);
    assert!(min_0.is_some());
    let (weight, id) = min_0.unwrap();
    assert!((weight - 1.0).abs() < f64::EPSILON);
    assert_eq!(id, 1);

    // Node 1: min weight is 0.5 to node 2
    let min_1 = graph.min_successor_weight_and_id(1);
    assert!(min_1.is_some());
    let (weight, id) = min_1.unwrap();
    assert!((weight - 0.5).abs() < f64::EPSILON);
    assert_eq!(id, 2);

    // Node 3: no successors
    let min_3 = graph.min_successor_weight_and_id(3);
    assert!(min_3.is_none());
}

// ============================================================================
// WeightedEdges trait directly on edges
// ============================================================================

#[test]
fn test_weighted_edges_direct() {
    let edge_data: Vec<(usize, usize, f64)> = vec![(0, 1, 5.0), (0, 2, 3.0), (1, 2, 7.0)];

    let edges: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(3)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    // Test successor_weights via WeightedEdges trait
    let weights: Vec<f64> = WeightedEdges::successor_weights(&edges, 0).collect();
    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 5.0).abs() < f64::EPSILON);
    assert!((weights[1] - 3.0).abs() < f64::EPSILON);

    // Test max_successor_weight
    let max = WeightedEdges::max_successor_weight(&edges, 0);
    assert!(max.is_some());
    assert!((max.unwrap() - 5.0).abs() < f64::EPSILON);

    // Test min_successor_weight
    let min = WeightedEdges::min_successor_weight(&edges, 0);
    assert!(min.is_some());
    assert!((min.unwrap() - 3.0).abs() < f64::EPSILON);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_single_edge_weights() {
    let nodes: Vec<usize> = vec![0, 1];
    let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 42.0)];

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 2))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> =
        GenericGraph::from((nodes, edges));

    // Single edge, so min == max
    let max = graph.max_successor_weight(0);
    let min = graph.min_successor_weight(0);
    assert_eq!(max, min);
    assert!((max.unwrap() - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_integer_weights() {
    let edges: Vec<(usize, usize, u32)> = vec![(0, 1, 10), (0, 2, 20), (1, 2, 5)];

    let edges: ValuedCSR2D<usize, usize, usize, u32> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, u32>>::default()
            .expected_number_of_edges(3)
            .expected_shape((3, 3))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    // Test with integer weights via WeightedEdges trait
    let max = WeightedEdges::max_successor_weight(&edges, 0);
    assert_eq!(max, Some(20));

    let min = WeightedEdges::min_successor_weight(&edges, 0);
    assert_eq!(min, Some(10));
}
