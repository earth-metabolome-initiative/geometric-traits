//! Tests for vertex match inference from edge matches.
#![cfg(feature = "alloc")]

use geometric_traits::prelude::*;

// ===========================================================================
// shared_endpoint tests
// ===========================================================================

#[test]
fn test_shared_endpoint_src_src() {
    // (1,2) and (1,3) share vertex 1
    assert_eq!(shared_endpoint((1_u32, 2), (1, 3)), Some(1));
}

#[test]
fn test_shared_endpoint_src_dst() {
    // (1,2) and (3,1) share vertex 1
    assert_eq!(shared_endpoint((1_u32, 2), (3, 1)), Some(1));
}

#[test]
fn test_shared_endpoint_dst_src() {
    // (1,2) and (2,3) share vertex 2
    assert_eq!(shared_endpoint((1_u32, 2), (2, 3)), Some(2));
}

#[test]
fn test_shared_endpoint_dst_dst() {
    // (1,2) and (3,2) share vertex 2
    assert_eq!(shared_endpoint((1_u32, 2), (3, 2)), Some(2));
}

#[test]
fn test_shared_endpoint_disjoint() {
    assert_eq!(shared_endpoint::<u32>((1, 2), (3, 4)), None);
}

#[test]
fn test_shared_endpoint_same_edge() {
    // Same edge shares both endpoints → None
    assert_eq!(shared_endpoint::<u32>((1, 2), (1, 2)), None);
}

#[test]
fn test_shared_endpoint_reversed_edge() {
    // Reversed edge shares both endpoints → None
    assert_eq!(shared_endpoint::<u32>((1, 2), (2, 1)), None);
}

#[test]
fn test_shared_endpoint_self_loop_shared() {
    // Self-loop (1,1) with (1,2): both a==c and b==c → multiple matches → None
    assert_eq!(shared_endpoint::<u32>((1, 1), (1, 2)), None);
}

// ===========================================================================
// infer_vertex_matches tests
// ===========================================================================

#[test]
fn test_infer_empty_clique() {
    let result = infer_vertex_matches::<u32, _>(&[], &[], &[], &[], |_, _, _, _| true);
    assert!(result.is_empty());
}

#[test]
fn test_infer_single_isolated_edge_true() {
    // Single edge (0,1) ↔ (10,11), disambiguate returns true → 0↔10, 1↔11
    let clique = [0_usize];
    let vertex_pairs = [(0_usize, 0_usize)];
    let edge_map1 = [(0_u32, 1_u32)];
    let edge_map2 = [(10_u32, 11_u32)];

    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);
    assert_eq!(result, vec![(0, 10), (1, 11)]);
}

#[test]
fn test_infer_single_isolated_edge_false() {
    // disambiguate returns false → 0↔11, 1↔10
    let clique = [0_usize];
    let vertex_pairs = [(0_usize, 0_usize)];
    let edge_map1 = [(0_u32, 1_u32)];
    let edge_map2 = [(10_u32, 11_u32)];

    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| false);
    assert_eq!(result, vec![(0, 11), (1, 10)]);
}

#[test]
fn test_infer_two_adjacent_edges() {
    // Path 0-1-2 matched to 10-11-12.
    // LG edges: 0=(0,1), 1=(1,2) and 0=(10,11), 1=(11,12)
    // Clique: [0, 1] in product pairs [(0,0), (1,1)]
    let clique = [0, 1];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1)];
    let edge_map1 = [(0_u32, 1_u32), (1, 2)];
    let edge_map2 = [(10_u32, 11_u32), (11, 12)];

    let result = infer_vertex_matches(
        &clique,
        &vertex_pairs,
        &edge_map1,
        &edge_map2,
        |_, _, _, _| true, // never called
    );
    // Phase 1: 1↔11 (shared endpoint)
    // Phase 2: 0↔10 (from edge (0,1)↔(10,11) with 1↔11), 2↔12 (from edge
    // (1,2)↔(11,12) with 1↔11)
    assert_eq!(result, vec![(0, 10), (1, 11), (2, 12)]);
}

#[test]
fn test_infer_triangle_to_triangle() {
    // K3: vertices 0,1,2 with edges (0,1),(0,2),(1,2)
    // Matched to K3: vertices 10,11,12 with edges (10,11),(10,12),(11,12)
    // LG vertices: 0=(0,1), 1=(0,2), 2=(1,2)
    // Product pairs: [(0,0), (1,1), (2,2)]
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (0, 2), (1, 2)];
    let edge_map2 = [(10_u32, 11_u32), (10, 12), (11, 12)];

    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| {
            panic!("should not be called for fully connected clique")
        });
    // Phase 1 resolves all: 0↔10 (shared between edges 0&1), 1↔11 (shared between
    // 0&2), 2↔12 (shared between 1&2)
    assert_eq!(result.len(), 3);
    assert_eq!(result, vec![(0, 10), (1, 11), (2, 12)]);
}

#[test]
fn test_infer_path_of_three_edges() {
    // Path 0-1-2-3 matched to 10-11-12-13
    // LG edges: 0=(0,1), 1=(1,2), 2=(2,3)
    // LG edges: 0=(10,11), 1=(11,12), 2=(12,13)
    // Clique: [0, 1, 2] in product pairs [(0,0), (1,1), (2,2)]
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (1, 2), (2, 3)];
    let edge_map2 = [(10_u32, 11_u32), (11, 12), (12, 13)];

    let result = infer_vertex_matches(
        &clique,
        &vertex_pairs,
        &edge_map1,
        &edge_map2,
        |_, _, _, _| true, // never called
    );
    // Phase 1: 1↔11 (edges 0&1), 2↔12 (edges 1&2)
    // Phase 2: 0↔10, 3↔13
    assert_eq!(result, vec![(0, 10), (1, 11), (2, 12), (3, 13)]);
}

#[test]
fn test_infer_star_center_resolved() {
    // Star: center=0, leaves=1,2,3. Edges: (0,1),(0,2),(0,3)
    // Matched to: center=10, leaves=11,12,13. Edges: (10,11),(10,12),(10,13)
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (0, 2), (0, 3)];
    let edge_map2 = [(10_u32, 11_u32), (10, 12), (10, 13)];

    let result = infer_vertex_matches(
        &clique,
        &vertex_pairs,
        &edge_map1,
        &edge_map2,
        |_, _, _, _| true, // never called
    );
    // Phase 1: 0↔10 (shared between all edge pairs)
    // Phase 2: 1↔11, 2↔12, 3↔13
    assert_eq!(result, vec![(0, 10), (1, 11), (2, 12), (3, 13)]);
}

#[test]
fn test_infer_mixed_connected_and_isolated() {
    // Two matched edges forming a path + one isolated edge.
    // Path: (0,1)↔(10,11), (1,2)↔(11,12) — connected
    // Isolated: (5,6)↔(15,16) — needs disambiguation
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (1, 2), (5, 6)];
    let edge_map2 = [(10_u32, 11_u32), (11, 12), (15, 16)];

    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);
    // Phase 1: 1↔11
    // Phase 2: 0↔10, 2↔12
    // Phase 3: disambiguate(5,6,15,16) → true → 5↔15, 6↔16
    assert_eq!(result, vec![(0, 10), (1, 11), (2, 12), (5, 15), (6, 16)]);
}

#[test]
fn test_infer_disambiguate_called_only_for_isolated() {
    // Verify disambiguate is never called for edges resolved by Phase 1/2.
    let clique = [0, 1];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1)];
    let edge_map1 = [(0_u32, 1_u32), (1, 2)];
    let edge_map2 = [(10_u32, 11_u32), (11, 12)];

    let mut called = false;
    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| {
            called = true;
            true
        });
    assert!(!called, "disambiguate should not be called when all edges are connected");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_infer_result_is_injective() {
    // For any valid MCES, the vertex mapping should be injective:
    // no two N1 values map to the same N2, and no two N2 values have the same N1.
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (0, 2), (1, 2)];
    let edge_map2 = [(10_u32, 11_u32), (10, 12), (11, 12)];

    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);

    // Check N1 uniqueness (already guaranteed by BTreeMap)
    for i in 0..result.len() {
        for j in (i + 1)..result.len() {
            assert_ne!(result[i].0, result[j].0, "duplicate N1 in result");
            assert_ne!(result[i].1, result[j].1, "duplicate N2 in result");
        }
    }
}

#[test]
fn test_infer_conflicting_symmetric_does_not_panic() {
    // Simulate a clique from K4 vs K4 that produces conflicting vertex
    // mappings due to symmetry. This previously panicked with debug_assert.
    //
    // Matched edges (G1 → G2):
    //   (0,1) → (0,2)   — edges share vertex 0 in G1
    //   (0,2) → (1,3)   — edges share vertex 0 in G1
    //   (0,3) → (2,3)   — edges share vertex 0 in G1
    //
    // Phase 1: edges (0,1)→(0,2) and (0,2)→(1,3) share vertex 0 in G1.
    //   In G2: (0,2) and (1,3) share NO vertex → no mapping.
    //   edges (0,1)→(0,2) and (0,3)→(2,3) share vertex 0 in G1.
    //   In G2: (0,2) and (2,3) share vertex 2 → map 0→2.
    //   edges (0,2)→(1,3) and (0,3)→(2,3) share vertex 0 in G1.
    //   In G2: (1,3) and (2,3) share vertex 3 → map 0→3. CONFLICT with 0→2!
    //
    // The conflict should be silently skipped (keep 0→2).
    let clique = [0, 1, 2];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1), (2, 2)];
    let edge_map1 = [(0_u32, 1_u32), (0, 2), (0, 3)];
    let edge_map2 = [(0_u32, 2_u32), (1, 3), (2, 3)];

    // Should not panic.
    let result =
        infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);

    // The result should still be a valid (partial) mapping — no duplicate N1 keys.
    for i in 0..result.len() {
        for j in (i + 1)..result.len() {
            assert_ne!(result[i].0, result[j].0, "duplicate N1 in result");
        }
    }
}
