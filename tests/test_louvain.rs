//! Tests for the Louvain community detection trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LouvainConfig, LouvainError},
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

fn build_weighted_graph(
    node_count: usize,
    directed_edges: Vec<(usize, usize, f64)>,
) -> WeightedMatrix {
    let mut edges = directed_edges;
    edges.sort_unstable_by(
        |(left_source, left_destination, _), (right_source, right_destination, _)| {
            (left_source, left_destination).cmp(&(right_source, right_destination))
        },
    );

    GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn build_undirected_weighted_graph(
    node_count: usize,
    undirected_edges: Vec<(usize, usize, f64)>,
) -> WeightedMatrix {
    let mut directed_edges = Vec::with_capacity(undirected_edges.len() * 2);
    for (source, destination, weight) in undirected_edges {
        directed_edges.push((source, destination, weight));
        if source != destination {
            directed_edges.push((destination, source, weight));
        }
    }
    build_weighted_graph(node_count, directed_edges)
}

#[test]
fn test_louvain_detects_two_communities() {
    let graph = build_undirected_weighted_graph(
        6,
        vec![
            (0, 1, 10.0),
            (0, 2, 10.0),
            (1, 2, 10.0),
            (3, 4, 10.0),
            (3, 5, 10.0),
            (4, 5, 10.0),
            (2, 3, 0.1),
        ],
    );

    let result = Louvain::<usize>::louvain(&graph, LouvainConfig::default()).unwrap();
    let partition = result.final_partition();

    assert_eq!(partition.len(), 6);
    assert_eq!(partition[0], partition[1]);
    assert_eq!(partition[1], partition[2]);
    assert_eq!(partition[3], partition[4]);
    assert_eq!(partition[4], partition[5]);
    assert_ne!(partition[0], partition[3]);
}

#[test]
fn test_louvain_is_deterministic_for_a_fixed_seed() {
    let graph = build_undirected_weighted_graph(
        8,
        vec![
            (0, 1, 4.0),
            (0, 2, 4.0),
            (1, 2, 4.0),
            (2, 3, 0.4),
            (3, 4, 4.0),
            (3, 5, 4.0),
            (4, 5, 4.0),
            (5, 6, 0.4),
            (6, 7, 4.0),
            (4, 7, 0.3),
        ],
    );

    let config = LouvainConfig {
        seed: 7,
        resolution: 1.0,
        modularity_threshold: 1.0e-7,
        max_levels: 100,
        max_local_passes: 100,
    };
    let first = Louvain::<usize>::louvain(&graph, config.clone()).unwrap();
    let second = Louvain::<usize>::louvain(&graph, config).unwrap();

    assert_eq!(first.final_partition(), second.final_partition());
    assert!((first.final_modularity() - second.final_modularity()).abs() <= 1.0e-12);
}

#[test]
fn test_louvain_returns_hierarchy_levels() {
    let graph = build_undirected_weighted_graph(
        6,
        vec![
            (0, 1, 8.0),
            (0, 2, 8.0),
            (1, 2, 8.0),
            (3, 4, 8.0),
            (3, 5, 8.0),
            (4, 5, 8.0),
            (2, 3, 0.2),
        ],
    );

    let result = Louvain::<usize>::louvain(&graph, LouvainConfig::default()).unwrap();

    assert!(!result.levels().is_empty());
    assert_eq!(result.levels().last().unwrap().partition(), result.final_partition());
    assert_eq!(result.levels().last().unwrap().modularity(), result.final_modularity());
}

#[test]
fn test_louvain_rejects_non_positive_weights() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 0.0)]);

    let error = Louvain::<usize>::louvain(&graph, LouvainConfig::default()).unwrap_err();
    assert!(matches!(error, LouvainError::NonPositiveWeight { .. }));
}

#[test]
fn test_louvain_rejects_non_symmetric_matrices() {
    let graph = build_weighted_graph(3, vec![(0, 1, 1.0), (1, 2, 1.0)]);

    let error = Louvain::<usize>::louvain(&graph, LouvainConfig::default()).unwrap_err();
    assert!(matches!(error, LouvainError::NonSymmetricEdge { .. }));
}

#[test]
fn test_louvain_marker_overflow_returns_error() {
    let graph = build_weighted_graph(300, Vec::new());
    let error = Louvain::<u8>::louvain(&graph, LouvainConfig::default()).unwrap_err();

    assert!(matches!(error, LouvainError::TooManyCommunities));
}
