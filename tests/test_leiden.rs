//! Tests for the Leiden community detection trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LeidenConfig, ModularityError},
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

fn partition_communities_are_connected(graph: &WeightedMatrix, partition: &[usize]) -> bool {
    let node_count = partition.len();
    if node_count == 0 {
        return true;
    }

    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    let mut nodes_per_community: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    for (node, community) in partition.iter().copied().enumerate() {
        nodes_per_community[community].push(node);
    }

    let mut is_member = vec![false; node_count];
    let mut visited = vec![false; node_count];
    let mut stack = Vec::new();

    for nodes in nodes_per_community {
        if nodes.len() <= 1 {
            continue;
        }

        for node in &nodes {
            is_member[*node] = true;
        }

        stack.clear();
        let start = nodes[0];
        stack.push(start);
        visited[start] = true;

        let mut visited_count = 0usize;
        while let Some(node) = stack.pop() {
            visited_count += 1;
            for destination in graph.sparse_row(node) {
                if destination < node_count && is_member[destination] && !visited[destination] {
                    visited[destination] = true;
                    stack.push(destination);
                }
            }
        }

        if visited_count != nodes.len() {
            return false;
        }

        for node in &nodes {
            is_member[*node] = false;
            visited[*node] = false;
        }
    }

    true
}

#[test]
fn test_leiden_detects_two_communities() {
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

    let result = Leiden::<usize>::leiden(&graph, &LeidenConfig::default()).unwrap();
    let partition = result.final_partition();

    assert_eq!(partition.len(), 6);
    assert_eq!(partition[0], partition[1]);
    assert_eq!(partition[1], partition[2]);
    assert_eq!(partition[3], partition[4]);
    assert_eq!(partition[4], partition[5]);
    assert_ne!(partition[0], partition[3]);
}

#[test]
fn test_leiden_is_deterministic_for_a_fixed_seed() {
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

    let config = LeidenConfig {
        seed: 7,
        resolution: 1.0,
        modularity_threshold: 1.0e-7,
        max_levels: 100,
        max_local_passes: 100,
        max_refinement_passes: 100,
        theta: 0.01,
    };
    let first = Leiden::<usize>::leiden(&graph, &config).unwrap();
    let second = Leiden::<usize>::leiden(&graph, &config).unwrap();

    assert_eq!(first.final_partition(), second.final_partition());
    assert!((first.final_modularity() - second.final_modularity()).abs() <= 1.0e-12);
}

#[test]
fn test_leiden_returns_hierarchy_levels() {
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

    let result = Leiden::<usize>::leiden(&graph, &LeidenConfig::default()).unwrap();

    assert!(!result.levels().is_empty());
    assert_eq!(result.levels().last().unwrap().partition(), result.final_partition());
    assert!(
        (result.levels().last().unwrap().modularity() - result.final_modularity()).abs() <= 1.0e-12
    );
}

#[test]
fn test_leiden_rejects_invalid_theta() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let config = LeidenConfig { theta: 0.0, ..LeidenConfig::default() };
    let error = Leiden::<usize>::leiden(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidTheta));
}

#[test]
fn test_leiden_rejects_zero_max_refinement_passes() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let config = LeidenConfig { max_refinement_passes: 0, ..LeidenConfig::default() };
    let error = Leiden::<usize>::leiden(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidMaxRefinementPasses));
}

#[test]
fn test_leiden_rejects_non_positive_weights() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 0.0)]);

    let error = Leiden::<usize>::leiden(&graph, &LeidenConfig::default()).unwrap_err();
    assert!(matches!(error, ModularityError::NonPositiveWeight { .. }));
}

#[test]
fn test_leiden_rejects_non_symmetric_matrices() {
    let graph = build_weighted_graph(3, vec![(0, 1, 1.0), (1, 2, 1.0)]);

    let error = Leiden::<usize>::leiden(&graph, &LeidenConfig::default()).unwrap_err();
    assert!(matches!(error, ModularityError::NonSymmetricEdge { .. }));
}

#[test]
fn test_leiden_marker_overflow_returns_error() {
    let graph = build_weighted_graph(300, Vec::new());
    let error = Leiden::<u8>::leiden(&graph, &LeidenConfig::default()).unwrap_err();

    assert!(matches!(error, ModularityError::TooManyCommunities));
}

#[test]
fn test_leiden_empty_graph() {
    let graph = build_weighted_graph(0, Vec::new());
    let result = Leiden::<usize>::leiden(&graph, &LeidenConfig::default()).unwrap();
    assert!(result.final_partition().is_empty());
    assert!(result.final_modularity().abs() <= 1.0e-12);
}

#[test]
fn test_leiden_communities_are_connected() {
    let graph = build_undirected_weighted_graph(
        10,
        vec![
            (0, 1, 12.0),
            (1, 2, 12.0),
            (2, 3, 12.0),
            (4, 5, 12.0),
            (5, 6, 12.0),
            (6, 7, 12.0),
            (7, 8, 12.0),
            (8, 9, 12.0),
            (3, 4, 0.02),
            (2, 6, 0.02),
        ],
    );

    let config = LeidenConfig { seed: 9, theta: 0.01, ..LeidenConfig::default() };
    let result = Leiden::<usize>::leiden(&graph, &config).unwrap();

    assert!(partition_communities_are_connected(&graph, result.final_partition()));
}
