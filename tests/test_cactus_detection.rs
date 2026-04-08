//! Integration tests for cactus graph detection on undirected graphs.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{CactusDetection, EdgesBuilder, VocabularyBuilder},
};

fn build_undigraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let mut edges: Vec<(usize, usize)> = edges
        .into_iter()
        .map(
            |(source, destination)| {
                if source <= destination { (source, destination) } else { (destination, source) }
            },
        )
        .collect();
    edges.sort_unstable();

    let node_vocabulary: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_matrix: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_vocabulary.len())
        .edges(edges.into_iter())
        .build()
        .unwrap();

    UndiGraph::from((node_vocabulary, edge_matrix))
}

fn count_simple_paths(
    adjacency: &[Vec<bool>],
    current: usize,
    target: usize,
    blocked_edge: (usize, usize),
    visited: &mut [bool],
    limit: usize,
) -> usize {
    if current == target {
        return 1;
    }

    visited[current] = true;
    let mut count = 0usize;

    for next in 0..adjacency.len() {
        if !adjacency[current][next]
            || visited[next]
            || (current == blocked_edge.0 && next == blocked_edge.1)
            || (current == blocked_edge.1 && next == blocked_edge.0)
        {
            continue;
        }

        count += count_simple_paths(adjacency, next, target, blocked_edge, visited, limit);
        if count >= limit {
            break;
        }
    }

    visited[current] = false;
    count
}

fn is_cactus_oracle(number_of_nodes: usize, edges: &[(usize, usize)]) -> bool {
    let mut adjacency = vec![vec![false; number_of_nodes]; number_of_nodes];
    for &(source, destination) in edges {
        if source == destination || adjacency[source][destination] {
            return false;
        }
        adjacency[source][destination] = true;
        adjacency[destination][source] = true;
    }

    for &(source, destination) in edges {
        let mut visited = vec![false; number_of_nodes];
        if count_simple_paths(
            &adjacency,
            source,
            destination,
            (source, destination),
            &mut visited,
            2,
        ) >= 2
        {
            return false;
        }
    }

    true
}

#[test]
fn test_empty_graph_is_cactus() {
    let graph = build_undigraph(vec![], vec![]);
    assert!(graph.is_cactus());
}

#[test]
fn test_tree_is_cactus() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    assert!(graph.is_cactus());
}

#[test]
fn test_simple_cycle_is_cactus() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    assert!(graph.is_cactus());
}

#[test]
fn test_two_triangles_sharing_articulation_vertex_is_cactus() {
    let graph =
        build_undigraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]);
    assert!(graph.is_cactus());
}

#[test]
fn test_disconnected_cactus_components_and_isolated_vertices_are_cactus() {
    let graph =
        build_undigraph(vec![0, 1, 2, 3, 4, 5, 6], vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)]);
    assert!(graph.is_cactus());
}

#[test]
fn test_two_triangles_sharing_edge_is_not_cactus() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)]);
    assert!(!graph.is_cactus());
}

#[test]
fn test_square_with_diagonal_is_not_cactus() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);
    assert!(!graph.is_cactus());
}

#[test]
fn test_complete_graph_k4_is_not_cactus() {
    let graph =
        build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    assert!(!graph.is_cactus());
}

#[test]
fn test_self_loop_is_not_cactus() {
    let graph = build_undigraph(vec![0, 1], vec![(0, 0), (0, 1)]);
    assert!(!graph.is_cactus());
}

#[test]
fn test_cactus_detection_matches_oracle_on_all_graphs_with_five_nodes() {
    let number_of_nodes = 5usize;
    let possible_edges: Vec<(usize, usize)> = (0..number_of_nodes)
        .flat_map(|source| {
            ((source + 1)..number_of_nodes).map(move |destination| (source, destination))
        })
        .collect();

    for mask in 0usize..(1usize << possible_edges.len()) {
        let edges: Vec<(usize, usize)> = possible_edges
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, edge)| ((mask & (1usize << index)) != 0).then_some(edge))
            .collect();

        let graph = build_undigraph((0..number_of_nodes).collect(), edges.clone());
        assert_eq!(
            graph.is_cactus(),
            is_cactus_oracle(number_of_nodes, &edges),
            "mask={mask}, edges={edges:?}"
        );
    }
}

#[test]
#[ignore = "exhaustive order-6 oracle sweep is slower than the default test suite"]
fn test_cactus_detection_matches_oracle_on_all_graphs_with_six_nodes() {
    let number_of_nodes = 6usize;
    let possible_edges: Vec<(usize, usize)> = (0..number_of_nodes)
        .flat_map(|source| {
            ((source + 1)..number_of_nodes).map(move |destination| (source, destination))
        })
        .collect();

    for mask in 0usize..(1usize << possible_edges.len()) {
        let edges: Vec<(usize, usize)> = possible_edges
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, edge)| ((mask & (1usize << index)) != 0).then_some(edge))
            .collect();

        let graph = build_undigraph((0..number_of_nodes).collect(), edges.clone());
        assert_eq!(
            graph.is_cactus(),
            is_cactus_oracle(number_of_nodes, &edges),
            "mask={mask}, edges={edges:?}"
        );
    }
}

#[test]
#[ignore = "exhaustive order-7 oracle sweep is much slower than the default test suite"]
fn test_cactus_detection_matches_oracle_on_all_graphs_with_seven_nodes() {
    let number_of_nodes = 7usize;
    let possible_edges: Vec<(usize, usize)> = (0..number_of_nodes)
        .flat_map(|source| {
            ((source + 1)..number_of_nodes).map(move |destination| (source, destination))
        })
        .collect();

    for mask in 0usize..(1usize << possible_edges.len()) {
        let edges: Vec<(usize, usize)> = possible_edges
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, edge)| ((mask & (1usize << index)) != 0).then_some(edge))
            .collect();

        let graph = build_undigraph((0..number_of_nodes).collect(), edges.clone());
        assert_eq!(
            graph.is_cactus(),
            is_cactus_oracle(number_of_nodes, &edges),
            "mask={mask}, edges={edges:?}"
        );
    }
}
