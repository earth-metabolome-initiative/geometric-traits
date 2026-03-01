//! Extended tests for Tarjan's algorithm for strongly connected components.
#![cfg(feature = "std")]

use std::collections::BTreeSet;

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::*,
    traits::{EdgesBuilder, Tarjan},
};

/// Helper to build a SquareCSR2D from directed edges.
fn build_square_csr(
    node_count: usize,
    edges: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Basic SCC tests
// ============================================================================

#[test]
fn test_tarjan_single_node_no_edge() {
    let m = build_square_csr(1, vec![]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], vec![0]);
}

#[test]
fn test_tarjan_self_loop() {
    let m = build_square_csr(1, vec![(0, 0)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], vec![0]);
}

#[test]
fn test_tarjan_two_node_cycle() {
    let m = build_square_csr(2, vec![(0, 1), (1, 0)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 2);
}

#[test]
fn test_tarjan_chain_no_cycles() {
    // 0 -> 1 -> 2 -> 3 (DAG, no cycles)
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    // Each node is its own SCC
    assert_eq!(sccs.len(), 4);
    for scc in &sccs {
        assert_eq!(scc.len(), 1);
    }
}

#[test]
fn test_tarjan_triangle_cycle() {
    // 0 -> 1 -> 2 -> 0
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 3);
}

#[test]
fn test_tarjan_two_separate_sccs() {
    // SCC 1: 0 -> 1 -> 0
    // SCC 2: 2 -> 3 -> 2
    // 1 -> 2 (bridge, not a cycle)
    let m = build_square_csr(4, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 2);

    // Verify both SCCs have 2 nodes each
    let mut sizes: Vec<usize> = sccs.iter().map(std::vec::Vec::len).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![2, 2]);
}

#[test]
fn test_tarjan_diamond_dag() {
    //   0
    //  / \
    // 1   2
    //  \ /
    //   3
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 4); // DAG, all singletons
}

#[test]
fn test_tarjan_complete_graph_3() {
    // Complete directed graph on 3 nodes: all are in one SCC
    let m = build_square_csr(3, vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 3);
}

#[test]
fn test_tarjan_disconnected_components() {
    // Two disconnected singletons: 0, 1 (no edges)
    let m = build_square_csr(2, vec![]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 2);
}

#[test]
fn test_tarjan_complex_graph() {
    // Three SCCs:
    // SCC {0, 1, 2}: 0->1->2->0
    // SCC {3}: just 3 (reached from 2)
    // SCC {4, 5}: 4->5->4
    // Bridges: 2->3, 3->4
    let m = build_square_csr(6, vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 4)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 3);

    let mut sizes: Vec<usize> = sccs.iter().map(std::vec::Vec::len).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![1, 2, 3]);
}

// ============================================================================
// Verify SCC membership
// ============================================================================

#[test]
fn test_tarjan_scc_membership() {
    // 0 <-> 1, 2 <-> 3, bridge 1 -> 2
    let m = build_square_csr(4, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();

    // Find which SCC contains node 0
    let scc_of_0 = sccs.iter().find(|scc| scc.contains(&0)).unwrap();
    assert!(scc_of_0.contains(&1), "Nodes 0 and 1 should be in the same SCC");

    let scc_of_2 = sccs.iter().find(|scc| scc.contains(&2)).unwrap();
    assert!(scc_of_2.contains(&3), "Nodes 2 and 3 should be in the same SCC");
}

// ============================================================================
// Total node count across all SCCs
// ============================================================================

#[test]
fn test_tarjan_all_nodes_covered() {
    let m = build_square_csr(5, vec![(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)]);

    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    let total_nodes: usize = sccs.iter().map(std::vec::Vec::len).sum();
    assert_eq!(total_nodes, 5, "All nodes must appear in exactly one SCC");

    // Verify each node appears exactly once
    let mut all_nodes: Vec<usize> = sccs.into_iter().flatten().collect();
    all_nodes.sort_unstable();
    assert_eq!(all_nodes, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_tarjan_regression_does_not_split_scc_after_singleton_pop() {
    // One non-trivial SCC {0,1,2} plus sink singleton {3}.
    // The DFS reaches 3 from 2 and must return to 2 (not rebind to 1).
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 3)]);

    let actual: BTreeSet<Vec<usize>> = m
        .tarjan()
        .map(|mut scc| {
            scc.sort_unstable();
            scc
        })
        .collect();
    let expected: BTreeSet<Vec<usize>> = BTreeSet::from([vec![0, 1, 2], vec![3]]);

    assert_eq!(actual, expected);
}
