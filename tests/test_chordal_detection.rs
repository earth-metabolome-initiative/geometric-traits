//! Integration tests for chordal graph detection on undirected graphs.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
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

fn is_chordal_oracle(number_of_nodes: usize, edges: &[(usize, usize)]) -> bool {
    let mut adjacency = vec![vec![false; number_of_nodes]; number_of_nodes];
    for &(source, destination) in edges {
        if source == destination {
            return false;
        }
        adjacency[source][destination] = true;
        adjacency[destination][source] = true;
    }

    for mask in 0usize..(1usize << number_of_nodes) {
        let subset_size = mask.count_ones() as usize;
        if subset_size < 4 {
            continue;
        }

        let subset: Vec<usize> =
            (0..number_of_nodes).filter(|node| (mask & (1usize << node)) != 0).collect();

        let mut degree_two = true;
        let mut induced_edge_count = 0usize;
        for &node in &subset {
            let degree = subset
                .iter()
                .copied()
                .filter(|&other| other != node && adjacency[node][other])
                .count();
            if degree != 2 {
                degree_two = false;
                break;
            }
            induced_edge_count += degree;
        }

        if !degree_two || induced_edge_count / 2 != subset_size {
            continue;
        }

        let mut visited = vec![false; number_of_nodes];
        let mut stack = vec![subset[0]];
        visited[subset[0]] = true;
        let mut visited_count = 0usize;

        while let Some(node) = stack.pop() {
            visited_count += 1;
            for &other in &subset {
                if adjacency[node][other] && !visited[other] {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }

        if visited_count == subset_size {
            return false;
        }
    }

    true
}

#[test]
fn test_tree_is_chordal() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (1, 3)]);
    assert!(graph.is_chordal());
    assert!(graph.perfect_elimination_ordering().is_some());
}

#[test]
fn test_cycle_c4_is_not_chordal() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (0, 3)]);
    assert!(!graph.is_chordal());
    assert!(graph.perfect_elimination_ordering().is_none());
}

#[test]
fn test_cycle_with_chord_is_chordal() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]);
    assert!(graph.is_chordal());
    let ordering = graph.perfect_elimination_ordering().unwrap();
    assert!(graph.is_perfect_elimination_ordering(&ordering));
}

#[test]
fn test_disconnected_graph_is_chordal_if_each_component_is_chordal() {
    let graph = build_undigraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (1, 2), (3, 4)]);
    assert!(graph.is_chordal());
}

#[test]
fn test_self_loop_is_not_chordal() {
    let graph = build_undigraph(vec![0, 1], vec![(0, 0), (0, 1)]);
    assert!(!graph.is_chordal());
    assert!(graph.perfect_elimination_ordering().is_none());
}

#[test]
fn test_invalid_perfect_elimination_ordering_is_rejected() {
    let graph = build_undigraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);
    assert!(graph.is_perfect_elimination_ordering(&[0, 1, 2]));
    assert!(!graph.is_perfect_elimination_ordering(&[1, 0, 2]));
    assert!(!graph.is_perfect_elimination_ordering(&[0, 0, 2]));
    assert!(!graph.is_perfect_elimination_ordering(&[0, 1]));
}

#[test]
fn test_mcs_returns_peo_for_chordal_graph() {
    let graph = build_undigraph(vec![0, 1, 2, 3, 4], vec![(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)]);
    let ordering = graph.maximum_cardinality_search_ordering();
    assert!(graph.is_perfect_elimination_ordering(&ordering));
}

#[test]
fn test_chordal_detection_matches_oracle_on_all_graphs_with_five_nodes() {
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
            graph.is_chordal(),
            is_chordal_oracle(number_of_nodes, &edges),
            "mask={mask}, edges={edges:?}"
        );
    }
}
