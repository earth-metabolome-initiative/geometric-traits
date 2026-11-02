//! Integration tests for exact undirected diameter computation.
#![cfg(feature = "std")]

use std::collections::VecDeque;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        Diameter, DiameterError, EdgesBuilder, VocabularyBuilder,
        algorithms::randomized_graphs::{
            barabasi_albert, barbell_graph, complete_graph, cycle_graph, erdos_renyi_gnm,
            grid_graph, path_graph, petersen_graph, star_graph, watts_strogatz, wheel_graph,
        },
    },
};

type UndirectedEdges = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type UndirectedGraph = UndiGraph<usize>;

fn build_graph_from_edges(edges: UndirectedEdges) -> UndirectedGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(edges.order())
        .symbols((0..edges.order()).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

fn build_graph(node_count: usize, edges: &[(usize, usize)]) -> UndirectedGraph {
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_unstable();
    let edges: UndirectedEdges = UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(node_count)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap();
    build_graph_from_edges(edges)
}

fn diameter_oracle(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    let order = graph.number_of_nodes();
    if order <= 1 {
        return Ok(0);
    }

    let mut distances = vec![usize::MAX; order];
    let mut queue = VecDeque::with_capacity(order);
    let mut diameter = 0;

    for source in graph.node_ids() {
        distances.fill(usize::MAX);
        queue.clear();
        distances[source] = 0;
        queue.push_back(source);
        let mut visited = 1;
        let mut eccentricity = 0;

        while let Some(node) = queue.pop_front() {
            let node_distance = distances[node];
            eccentricity = eccentricity.max(node_distance);

            for neighbor in graph.neighbors(node) {
                if distances[neighbor] != usize::MAX {
                    continue;
                }

                distances[neighbor] = node_distance + 1;
                visited += 1;
                queue.push_back(neighbor);
            }
        }

        if visited != order {
            return Err(DiameterError::DisconnectedGraph);
        }

        diameter = diameter.max(eccentricity);
    }

    Ok(diameter)
}

fn computed_diameter(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    graph.diameter().map_err(|error| {
        match error {
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::DiameterError(error)) => {
                error
            }
            other => panic!("unexpected diameter error shape: {other}"),
        }
    })
}

#[test]
fn test_empty_graph_has_zero_diameter() {
    let graph = build_graph(0, &[]);
    assert_eq!(computed_diameter(&graph).unwrap(), 0);
}

#[test]
fn test_singleton_graph_has_zero_diameter() {
    let graph = build_graph(1, &[]);
    assert_eq!(computed_diameter(&graph).unwrap(), 0);
}

#[test]
fn test_path_graph_diameter_matches_length() {
    for order in 2..=10 {
        let graph = build_graph_from_edges(path_graph(order));
        assert_eq!(computed_diameter(&graph).unwrap(), order - 1);
    }
}

#[test]
fn test_cycle_graph_diameter_matches_half_order() {
    for order in 3..=12 {
        let graph = build_graph_from_edges(cycle_graph(order));
        assert_eq!(computed_diameter(&graph).unwrap(), order / 2);
    }
}

#[test]
fn test_complete_graph_diameter_is_one() {
    for order in 2..=10 {
        let graph = build_graph_from_edges(complete_graph(order));
        assert_eq!(computed_diameter(&graph).unwrap(), 1);
    }
}

#[test]
fn test_star_graph_diameter_is_two() {
    for order in 3..=10 {
        let graph = build_graph_from_edges(star_graph(order));
        assert_eq!(computed_diameter(&graph).unwrap(), 2);
    }
}

#[test]
fn test_disconnected_graph_returns_error() {
    let graph = build_graph(4, &[(0, 1), (2, 3)]);
    assert_eq!(computed_diameter(&graph), Err(DiameterError::DisconnectedGraph));
}

#[test]
fn test_matches_oracle_on_deterministic_graph_families() {
    let graphs = [
        build_graph_from_edges(path_graph(9)),
        build_graph_from_edges(cycle_graph(11)),
        build_graph_from_edges(complete_graph(8)),
        build_graph_from_edges(star_graph(7)),
        build_graph_from_edges(grid_graph(3, 4)),
        build_graph_from_edges(wheel_graph(9)),
        build_graph_from_edges(barbell_graph(4, 3)),
        build_graph_from_edges(petersen_graph()),
    ];

    for graph in graphs {
        assert_eq!(computed_diameter(&graph), diameter_oracle(&graph));
    }
}

#[test]
fn test_matches_oracle_on_small_random_graphs() {
    for seed in 1..=8 {
        let graphs = [
            ("erdos_renyi_gnm", build_graph_from_edges(erdos_renyi_gnm(seed, 24, 36))),
            ("barabasi_albert", build_graph_from_edges(barabasi_albert(seed, 24, 3))),
            ("watts_strogatz", build_graph_from_edges(watts_strogatz(seed, 24, 4, 0.2))),
        ];

        for (family, graph) in graphs {
            assert_eq!(
                computed_diameter(&graph),
                diameter_oracle(&graph),
                "family={family} seed={seed}"
            );
        }
    }
}
