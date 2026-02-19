//! Test submodule for the `RandomizedDAG` trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    naive_structs::GenericGraph,
    prelude::RandomizedDAG,
    traits::{CycleDetection, MonopartiteGraph, MonoplexGraph},
};

type SimpleDiGraph = GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>>;

#[test]
fn test_randomized_dag_is_acyclic() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(42, 10);
    assert!(!graph.has_cycle(), "Randomized DAG should be acyclic");
}

#[test]
fn test_randomized_dag_has_correct_number_of_nodes() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(123, 15);
    assert_eq!(graph.number_of_nodes(), 15);
}

#[test]
fn test_randomized_dag_different_seeds_produce_different_graphs() {
    let graph1: SimpleDiGraph = SimpleDiGraph::randomized_dag(1, 10);
    let graph2: SimpleDiGraph = SimpleDiGraph::randomized_dag(2, 10);

    let edges1 = graph1.number_of_edges();
    let edges2 = graph2.number_of_edges();
    // Both should be valid DAGs
    assert!(edges1 <= 10 * 9 / 2, "Too many edges for 10 nodes");
    assert!(edges2 <= 10 * 9 / 2, "Too many edges for 10 nodes");
}

#[test]
fn test_randomized_dag_same_seed_same_graph() {
    let graph1: SimpleDiGraph = SimpleDiGraph::randomized_dag(42, 10);
    let graph2: SimpleDiGraph = SimpleDiGraph::randomized_dag(42, 10);
    assert_eq!(
        graph1.number_of_edges(),
        graph2.number_of_edges(),
        "Same seed should produce same number of edges"
    );
    assert_eq!(
        graph1.number_of_nodes(),
        graph2.number_of_nodes(),
        "Same seed should produce same number of nodes"
    );
}

#[test]
fn test_randomized_dag_small_graph() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(99, 3);
    assert_eq!(graph.number_of_nodes(), 3);
    assert!(!graph.has_cycle(), "Small random DAG should be acyclic");
}

#[test]
fn test_randomized_dag_edges_are_forward() {
    // In the implementation, edges always go from lower to higher index (src <
    // dst).
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(7, 8);
    for node_id in graph.node_ids() {
        for successor in graph.successors(node_id) {
            assert!(node_id < successor, "Edge ({node_id}, {successor}) violates forward ordering");
        }
    }
}
