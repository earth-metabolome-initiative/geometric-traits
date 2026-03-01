//! Extended tests for Johnson's algorithm for finding all cycles.
#![cfg(feature = "std")]

use std::collections::BTreeSet;

use geometric_traits::traits::Johnson;

mod common;

use common::build_square_csr;

fn canonical_cycle(cycle: &[usize]) -> Vec<usize> {
    let n = cycle.len();
    let mut best = cycle.to_vec();
    for shift in 1..n {
        let mut rotated = Vec::with_capacity(n);
        rotated.extend_from_slice(&cycle[shift..]);
        rotated.extend_from_slice(&cycle[..shift]);
        if rotated < best {
            best = rotated;
        }
    }
    best
}

// ============================================================================
// No cycles
// ============================================================================

#[test]
fn test_johnson_dag_no_cycles() {
    // 0 -> 1 -> 2 -> 3 (pure DAG)
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty(), "A DAG should have no cycles");
}

#[test]
fn test_johnson_disconnected_dag() {
    let m = build_square_csr(3, vec![(0, 1)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty());
}

// ============================================================================
// Single cycle
// ============================================================================

#[test]
fn test_johnson_single_triangle_cycle() {
    // 0 -> 1 -> 2 -> 0
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].len(), 3);
}

#[test]
fn test_johnson_two_node_cycle() {
    // 0 -> 1 -> 0
    let m = build_square_csr(2, vec![(0, 1), (1, 0)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].len(), 2);
}

// ============================================================================
// Multiple cycles
// ============================================================================

#[test]
fn test_johnson_two_separate_cycles() {
    // Cycle 1: 0 -> 1 -> 0
    // Cycle 2: 2 -> 3 -> 2
    // Bridge: 1 -> 2
    let m = build_square_csr(4, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_overlapping_cycles() {
    // 0 -> 1 -> 2 -> 0 (triangle)
    // 0 -> 2 -> 0 (shortcut)
    // This creates 2 distinct cycles
    let m = build_square_csr(3, vec![(0, 1), (0, 2), (1, 2), (2, 0)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Should find: [0,1,2] and [0,2]
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_regression_does_not_drop_valid_cycle() {
    // This graph has three elementary cycles:
    // [0,1], [1,2], and [0,2,1].
    let m = build_square_csr(3, vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]);

    let actual: BTreeSet<Vec<usize>> = m.johnson().map(|cycle| canonical_cycle(&cycle)).collect();
    let expected: BTreeSet<Vec<usize>> = BTreeSet::from([vec![0, 1], vec![1, 2], vec![0, 2, 1]]);

    assert_eq!(actual, expected);
}

#[test]
fn test_johnson_regression_resume_after_yield_keeps_exploring_branch() {
    // Verified deterministic counterexample from corpus minimization:
    // cycles are [0,1], [0,2], and [0,2,1].
    //
    // Node 3 is a dead-end tail from 2 and should not affect cycle enumeration.
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 3)]);

    let actual: BTreeSet<Vec<usize>> = m.johnson().map(|cycle| canonical_cycle(&cycle)).collect();
    let expected: BTreeSet<Vec<usize>> = BTreeSet::from([vec![0, 1], vec![0, 2], vec![0, 2, 1]]);

    assert_eq!(actual, expected);
}

// ============================================================================
// Self-loops
// ============================================================================

#[test]
fn test_johnson_self_loop_not_detected() {
    // Johnson's algorithm skips singleton SCCs (including self-loops),
    // as it only searches for cycles in SCCs with 2+ nodes.
    let m = build_square_csr(2, vec![(0, 0), (0, 1)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty(), "Self-loop alone does not form a multi-node SCC");
}

#[test]
fn test_johnson_self_loop_inside_multi_node_scc_is_not_emitted() {
    // 1 has a self-loop and also participates in a 2-cycle with 3.
    // Johnson should enumerate only cycles with 2+ nodes.
    let m = build_square_csr(4, vec![(0, 1), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1)]);

    let actual: BTreeSet<Vec<usize>> = m.johnson().map(|cycle| canonical_cycle(&cycle)).collect();
    let expected: BTreeSet<Vec<usize>> = BTreeSet::from([vec![1, 3]]);

    assert_eq!(actual, expected);
}

#[test]
fn test_johnson_self_loops_do_not_appear_alongside_multi_node_cycles() {
    // Triangle plus self-loops inside the same SCC.
    let m = build_square_csr(4, vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();

    assert!(!cycles.is_empty());
    assert!(cycles.iter().all(|cycle| cycle.len() > 1));
}

// ============================================================================
// Complete graph
// ============================================================================

#[test]
fn test_johnson_complete_3_node() {
    // K3: all 6 directed edges
    let m = build_square_csr(3, vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // K3 has 5 cycles: three 2-cycles (0,1), (0,2), (1,2), two 3-cycles
    assert_eq!(cycles.len(), 5);
}

// ============================================================================
// Cycle with tail
// ============================================================================

#[test]
fn test_johnson_cycle_with_tail() {
    // 0 -> 1 -> 2 -> 3 -> 1 (cycle is 1->2->3->1, tail is 0->1)
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3), (3, 1)]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 1);
    // The cycle should be 1 -> 2 -> 3 (not including node 0)
    assert_eq!(cycles[0].len(), 3);
    assert!(!cycles[0].contains(&0));
}

// ============================================================================
// No edges
// ============================================================================

#[test]
fn test_johnson_no_edges() {
    let m = build_square_csr(3, vec![]);

    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty());
}
