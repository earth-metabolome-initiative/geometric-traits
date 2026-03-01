//! Tests for Johnson's algorithm exercising complex graph structures:
//! diamond SCCs, deeply nested cycles, multiple SCCs with unblock paths.
#![cfg(feature = "std")]

use geometric_traits::traits::Johnson;

mod common;

use common::build_square_csr;

// ============================================================================
// Diamond-shaped SCC with block_map exercised
// ============================================================================

#[test]
fn test_johnson_diamond_scc() {
    // 0 -> 1 -> 3 -> 0 (triangle cycle)
    // 0 -> 2 -> 3 -> 0 (alternative path)
    // This creates a diamond where 0 reaches 3 via both 1 and 2.
    // Expected cycles: [0,1,3], [0,2,3], and [0,1,3,2] or similar combos
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Should find all elementary cycles through the diamond
    assert!(cycles.len() >= 2, "Diamond SCC should have at least 2 cycles, found {}", cycles.len());
}

#[test]
fn test_johnson_complete_4_directed() {
    // Complete directed graph K4: all 12 directed edges
    let mut edges = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                edges.push((i, j));
            }
        }
    }
    let m = build_square_csr(4, edges);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // K4 has many cycles
    assert!(cycles.len() > 5, "K4 should have many cycles, found {}", cycles.len());
}

#[test]
fn test_johnson_two_disjoint_sccs_separate_roots() {
    // SCC1: 0 -> 1 -> 2 -> 0
    // SCC2: 4 -> 5 -> 6 -> 4
    // Node 3 connects them (not in any cycle)
    let m = build_square_csr(
        7,
        vec![
            (0, 1),
            (1, 2),
            (2, 0), // SCC1
            (2, 3), // bridge
            (3, 4), // bridge
            (4, 5),
            (5, 6),
            (6, 4), // SCC2
        ],
    );
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_nested_cycles_sharing_nodes() {
    // 0 -> 1 -> 2 -> 0 (outer cycle)
    // 1 -> 0 (direct back edge, creates 2-node cycle with 0)
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0), (1, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 2, "Should find at least the 2-cycle [0,1] and 3-cycle [0,1,2]");
}

#[test]
fn test_johnson_dense_scc_with_multiple_back_edges() {
    // Dense 5-node SCC: edges creating many elementary cycles
    let m = build_square_csr(
        5,
        vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0), // main ring
            (0, 2),
            (2, 4),
            (4, 1), // shortcuts
        ],
    );
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 3, "Dense SCC should have multiple cycles, found {}", cycles.len());
}

// ============================================================================
// Edge cases that exercise block_map and unblock paths
// ============================================================================

#[test]
fn test_johnson_multiple_paths_to_root() {
    // Graph where node 0 can be reached from 2 via two different paths,
    // exercising the block_map + unblock mechanism
    let m = build_square_csr(
        5,
        vec![
            (0, 1),
            (0, 3),
            (1, 2),
            (2, 0), // cycle 0->1->2->0
            (3, 4),
            (4, 2), // 0->3->4->2->0 is another cycle
        ],
    );
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 2);
}

#[test]
fn test_johnson_self_loops_with_multi_node_cycle() {
    // Self-loops can form single-node SCCs. Combined with multi-node cycle.
    let m = build_square_csr(
        4,
        vec![
            (0, 0), // self-loop — Johnson skips singleton SCCs
            (0, 1),
            (1, 2),
            (2, 0), // triangle
            (1, 1), // self-loop
        ],
    );
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // The self-loops form their own SCCs. The triangle 0->1->2->0 also forms one.
    // With self-loops included in the SCC (since (0,0) and (1,1) are within
    // the same strongly connected component), we get additional cycles.
    assert!(!cycles.is_empty());
}

#[test]
fn test_johnson_linear_chain_no_cycles() {
    // 0 -> 1 -> 2 -> 3 -> 4 (no cycles)
    let m = build_square_csr(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.is_empty());
}
