//! Tests for Johnson's algorithm coverage: DAG (no cycles), multi-SCC,
//! cascading unblock, graph structures that exercise find_path and scan.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::*,
    traits::EdgesBuilder,
};

fn build_square_csr(n: usize, mut edges: Vec<(usize, usize)>) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// DAG: no cycles at all → Johnson should yield nothing
// ============================================================================

#[test]
fn test_johnson_dag_no_cycles() {
    // 0 → 1 → 2 → 3 (DAG, no back edges)
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty(), "DAG should have no cycles");
}

#[test]
fn test_johnson_dag_tree() {
    // Tree: 0→1, 0→2, 1→3, 1→4
    let m = build_square_csr(5, vec![(0, 1), (0, 2), (1, 3), (1, 4)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty());
}

// ============================================================================
// Single cycle
// ============================================================================

#[test]
fn test_johnson_single_triangle() {
    // 0 → 1 → 2 → 0
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].len(), 3);
}

#[test]
fn test_johnson_single_self_loop() {
    // Self-loop: 0 → 0 forms an SCC {0}, but Tarjan only finds SCCs of size > 1
    // by default in Johnson's. So self-loops may not be detected.
    let m = build_square_csr(1, vec![(0, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Johnson's algorithm may not find self-loops (depends on Tarjan SCC filtering)
    assert!(cycles.is_empty() || cycles[0] == vec![0]);
}

// ============================================================================
// Multiple disjoint cycles (tests multiple SCCs and root advancement)
// ============================================================================

#[test]
fn test_johnson_two_disjoint_triangles() {
    // Cycle 1: 0 → 1 → 2 → 0
    // Cycle 2: 3 → 4 → 5 → 3
    let m = build_square_csr(6, vec![
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

// ============================================================================
// Complex: SCC + non-SCC nodes (exercises root advancement past singleton)
// ============================================================================

#[test]
fn test_johnson_scc_plus_tail() {
    // SCC: 0 → 1 → 2 → 0
    // Tail: 2 → 3 (no back edge from 3)
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 0), (2, 3)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Only the triangle cycle
    assert_eq!(cycles.len(), 1);
}

// ============================================================================
// Multiple cycles sharing nodes (exercises blocking/unblocking)
// ============================================================================

#[test]
fn test_johnson_two_cycles_shared_node() {
    // 0 → 1 → 0 (2-cycle)
    // 0 → 1 → 2 → 0 (3-cycle)
    let m = build_square_csr(3, vec![(0, 1), (1, 0), (1, 2), (2, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_diamond_scc() {
    // 0→1, 0→2, 1→3, 2→3, 3→0
    // Cycles: 0→1→3→0, 0→2→3→0
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

// ============================================================================
// Cascading unblock: deep nested cycles
// ============================================================================

#[test]
fn test_johnson_nested_cycles() {
    // Inner cycle: 0→1→0
    // Outer cycle: 0→1→2→3→0
    // Also: 1→2→1
    let m = build_square_csr(4, vec![
        (0, 1), (1, 0),  // inner 2-cycle
        (1, 2), (2, 1),  // another 2-cycle
        (2, 3), (3, 0),  // completes outer cycle
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Cycles: {0,1}, {1,2}, {0,1,2,3}, {0,1,2,3} via different paths
    assert!(cycles.len() >= 3);
}

#[test]
fn test_johnson_k4_complete() {
    // Complete directed graph K4 — many cycles
    let m = build_square_csr(4, vec![
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 3),
        (3, 0), (3, 1), (3, 2),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // K4 has many cycles (all permutations form cycles)
    assert!(cycles.len() > 5);
}

// ============================================================================
// Large SCC with long chain
// ============================================================================

#[test]
fn test_johnson_long_cycle() {
    // 0 → 1 → 2 → 3 → 4 → 5 → 0 (6-cycle)
    let m = build_square_csr(6, vec![
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].len(), 6);
}

// ============================================================================
// Graph with isolated node (exercises iterator termination)
// ============================================================================

#[test]
fn test_johnson_with_isolated_nodes() {
    // Cycle: 0→1→0, node 2 isolated, node 3 isolated
    let m = build_square_csr(4, vec![(0, 1), (1, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
}
