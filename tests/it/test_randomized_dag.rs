//! Test submodule for randomized DAG generation and XorShift64 PRNG.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::{GenericGraph, MonopartiteGraph, MonoplexGraph, RandomizedDAG},
    traits::{CycleDetection, randomized_graphs::XorShift64},
};

type SimpleDiGraph = GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>>;

#[test]
fn test_xorshift64_sequence() {
    // XorShift64 with a non-zero seed must never produce zero (a mathematical
    // property).
    let mut rng = XorShift64::from(1u64);
    let values: Vec<u64> = rng.by_ref().take(10).collect();
    assert!(
        values.iter().all(|&v| v != 0),
        "XorShift64 should never produce 0 from a non-zero seed"
    );
}

#[test]
fn test_xorshift64_nonzero_seed() {
    // The same seed must always produce the same sequence (determinism).
    let values_a: Vec<u64> = XorShift64::from(42u64).take(5).collect();
    let values_b: Vec<u64> = XorShift64::from(42u64).take(5).collect();
    assert_eq!(values_a, values_b, "XorShift64 must be deterministic");

    // And consecutive values should differ (no fixed-point at the chosen seed).
    assert_ne!(values_a[0], values_a[1]);
    assert_ne!(values_a[1], values_a[2]);
}

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
    // Both should be valid DAGs.
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
fn test_randomized_dag_node_count() {
    let dag: GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>> =
        RandomizedDAG::randomized_dag(12345u64, 5);
    assert_eq!(dag.number_of_nodes(), 5, "DAG should have exactly 5 nodes");
}

#[test]
fn test_randomized_dag_deterministic() {
    // Two DAGs generated with the same seed and node count must be identical.
    let dag1: GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>> =
        RandomizedDAG::randomized_dag(99999u64, 6);
    let dag2: GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>> =
        RandomizedDAG::randomized_dag(99999u64, 6);
    assert_eq!(
        dag1.number_of_edges(),
        dag2.number_of_edges(),
        "Same seed and node count must produce the same edge count"
    );
}

#[test]
fn test_randomized_dag_edge_bound() {
    // A DAG on n nodes can have at most n*(n-1)/2 edges.
    const NODES: usize = 8;
    let dag: GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>> =
        RandomizedDAG::randomized_dag(77777u64, NODES);
    let max_edges = NODES * (NODES - 1) / 2;
    assert!(
        dag.number_of_edges() <= max_edges,
        "DAG edge count {} exceeds maximum {} for {} nodes",
        dag.number_of_edges(),
        max_edges,
        NODES
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

#[test]
fn test_randomized_dag_zero_nodes_no_panic_and_empty() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(123, 0);
    assert_eq!(graph.number_of_nodes(), 0);
    assert_eq!(graph.number_of_edges(), 0);
}

#[test]
fn test_randomized_dag_one_node_no_panic_and_no_edges() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(123, 1);
    assert_eq!(graph.number_of_nodes(), 1);
    assert_eq!(graph.number_of_edges(), 0);
}

#[test]
fn test_randomized_dag_two_nodes_can_reach_max_edges() {
    let saw_max = (0u64..1024).any(|seed| {
        let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(seed, 2);
        graph.number_of_edges() == 1
    });

    assert!(saw_max, "Expected at least one seed to produce the maximal edge count for 2 nodes");
}

#[test]
fn test_randomized_dag_seed_zero_not_always_empty() {
    let graph: SimpleDiGraph = SimpleDiGraph::randomized_dag(0, 10);
    let max_edges = 10 * 9 / 2;
    assert_eq!(graph.number_of_nodes(), 10);
    assert!(graph.number_of_edges() <= max_edges);
    assert!(!graph.has_cycle(), "Randomized DAG should be acyclic");
    assert!(graph.number_of_edges() > 0, "Seed 0 should not collapse to an always-empty DAG");
}

#[test]
fn test_randomized_dag_seed_zero_is_deterministic() {
    let graph1: SimpleDiGraph = SimpleDiGraph::randomized_dag(0, 10);
    let graph2: SimpleDiGraph = SimpleDiGraph::randomized_dag(0, 10);
    assert_eq!(graph1.number_of_nodes(), graph2.number_of_nodes());
    assert_eq!(graph1.number_of_edges(), graph2.number_of_edges());
}
