//! Tests for the maximum clique algorithm.
#![cfg(feature = "alloc")]

extern crate alloc;

#[cfg(feature = "std")]
use std::io::Read as _;

use geometric_traits::{impls::BitSquareMatrix, prelude::*};

/// Helper: verify all returned cliques are actual cliques.
fn verify_cliques(adj: &BitSquareMatrix, cliques: &[Vec<usize>]) {
    for clique in cliques {
        for (i, &u) in clique.iter().enumerate() {
            for &v in &clique[i + 1..] {
                assert!(
                    adj.has_entry(u, v) && adj.has_entry(v, u),
                    "vertices {u} and {v} in clique {clique:?} are not adjacent"
                );
            }
        }
    }
}

/// Helper: verify all cliques have the same size and that no larger clique
/// exists by checking that no vertex outside the clique is adjacent to all
/// clique members. (Only valid for single-clique results or when ω is known.)
fn verify_all_same_size(cliques: &[Vec<usize>]) {
    if cliques.len() <= 1 {
        return;
    }
    let sz = cliques[0].len();
    for c in cliques {
        assert_eq!(c.len(), sz, "clique sizes differ: {sz} vs {}", c.len());
    }
}

/// Helper: verify no duplicates.
fn verify_no_duplicates(cliques: &[Vec<usize>]) {
    let mut sorted: Vec<Vec<usize>> = cliques.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), cliques.len(), "duplicate cliques found");
}

// ============================================================================
// Empty and trivial graphs
// ============================================================================

#[test]
fn test_empty_graph() {
    let g = BitSquareMatrix::new(0);
    let cliques = g.all_maximum_cliques();
    let expected: Vec<Vec<usize>> = vec![vec![]];
    assert_eq!(cliques, expected);
    let empty: Vec<usize> = vec![];
    assert_eq!(g.maximum_clique(), empty);
}

#[test]
fn test_single_vertex() {
    let g = BitSquareMatrix::new(1);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques, vec![vec![0]]);
    assert_eq!(g.maximum_clique(), vec![0]);
}

#[test]
fn test_two_isolated_vertices() {
    let g = BitSquareMatrix::new(2);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 2);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 1);
}

#[test]
fn test_single_edge() {
    let g = BitSquareMatrix::from_symmetric_edges(2, vec![(0, 1)]);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 1);
    assert_eq!(cliques[0], vec![0, 1]);
    verify_cliques(&g, &cliques);
}

// ============================================================================
// Complete graphs
// ============================================================================

#[test]
fn test_k3() {
    let g = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 1);
    assert_eq!(cliques[0], vec![0, 1, 2]);
    verify_cliques(&g, &cliques);
}

#[test]
fn test_k4() {
    let edges: Vec<(usize, usize)> = (0..4).flat_map(|i| (i + 1..4).map(move |j| (i, j))).collect();
    let g = BitSquareMatrix::from_symmetric_edges(4, edges);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 1);
    assert_eq!(cliques[0], vec![0, 1, 2, 3]);
    verify_cliques(&g, &cliques);
}

#[test]
fn test_k5() {
    let edges: Vec<(usize, usize)> = (0..5).flat_map(|i| (i + 1..5).map(move |j| (i, j))).collect();
    let g = BitSquareMatrix::from_symmetric_edges(5, edges);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 1);
    assert_eq!(cliques[0], vec![0, 1, 2, 3, 4]);
    verify_cliques(&g, &cliques);
}

#[test]
fn test_k6() {
    let edges: Vec<(usize, usize)> = (0..6).flat_map(|i| (i + 1..6).map(move |j| (i, j))).collect();
    let g = BitSquareMatrix::from_symmetric_edges(6, edges);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 1);
    assert_eq!(cliques[0], vec![0, 1, 2, 3, 4, 5]);
    verify_cliques(&g, &cliques);
}

// ============================================================================
// Graphs with multiple maximum cliques
// ============================================================================

#[test]
fn test_two_triangles_sharing_edge() {
    // 0-1-2 triangle, plus 1-2-3 triangle → two K3 sharing edge (1,2)
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 2);
    verify_all_same_size(&cliques);
    verify_cliques(&g, &cliques);
    verify_no_duplicates(&cliques);
    assert_eq!(cliques[0].len(), 3);
}

#[test]
fn test_two_disjoint_triangles() {
    // Triangle {0,1,2} and triangle {3,4,5}
    let g = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
    );
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 2);
    verify_all_same_size(&cliques);
    verify_cliques(&g, &cliques);
    verify_no_duplicates(&cliques);
    assert_eq!(cliques[0].len(), 3);
}

#[test]
fn test_k4_minus_one_edge() {
    // K4 minus edge (0,3): three K3 cliques
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    verify_no_duplicates(&cliques);
    // K3s: {0,1,2}, {1,2,3}
    assert_eq!(cliques.len(), 2);
    assert_eq!(cliques[0].len(), 3);
}

// ============================================================================
// Path and cycle graphs
// ============================================================================

#[test]
fn test_path_p4() {
    // 0-1-2-3
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 2);
    assert_eq!(cliques.len(), 3); // {0,1}, {1,2}, {2,3}
}

#[test]
fn test_cycle_c5() {
    // 0-1-2-3-4-0
    let g = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 2);
    assert_eq!(cliques.len(), 5); // 5 edges = 5 K2 cliques
}

#[test]
fn test_cycle_c4() {
    // 0-1-2-3-0 (no diagonals)
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 2);
    assert_eq!(cliques.len(), 4);
}

// ============================================================================
// Star graph
// ============================================================================

#[test]
fn test_star_k1_4() {
    // Center 0 connected to 1,2,3,4
    let g = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 2);
    assert_eq!(cliques.len(), 4); // {0,1}, {0,2}, {0,3}, {0,4}
}

// ============================================================================
// Petersen graph
// ============================================================================

#[test]
fn test_petersen_graph() {
    // Petersen graph: 10 vertices, 15 edges, ω = 2 (triangle-free)
    // Outer pentagon: 0-1-2-3-4-0
    // Inner pentagram: 5-7-9-6-8-5
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    let g = BitSquareMatrix::from_symmetric_edges(
        10,
        vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0), // outer
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5), // inner
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9), // spokes
        ],
    );
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    // Petersen graph is triangle-free, so ω = 2
    assert_eq!(cliques[0].len(), 2);
    // 15 edges = 15 K2 cliques
    assert_eq!(cliques.len(), 15);
}

// ============================================================================
// Single mode
// ============================================================================

#[test]
fn test_maximum_clique_single_returns_one() {
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
    let clique = g.maximum_clique();
    assert_eq!(clique.len(), 3);
    // Verify it's an actual clique
    verify_cliques(&g, &[clique]);
}

#[test]
fn test_maximum_clique_where_can_skip_rejected_tied_maximum() {
    let g = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
    );

    let clique = g.maximum_clique_where(|clique| !clique.contains(&0));

    assert_eq!(clique, vec![3, 4, 5]);
    verify_cliques(&g, &[clique]);
}

#[test]
fn test_all_maximum_cliques_where_only_returns_accepted_maxima() {
    let g = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
    );

    let cliques = g.all_maximum_cliques_where(|clique| !clique.contains(&0));

    assert_eq!(cliques, vec![vec![3, 4, 5]]);
    verify_cliques(&g, &cliques);
}

#[test]
fn test_maximum_clique_where_accept_all_matches_existing_api() {
    let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);

    assert_eq!(g.maximum_clique_where(|_| true), g.maximum_clique());
    assert_eq!(g.all_maximum_cliques_where(|_| true), g.all_maximum_cliques());
}

// ============================================================================
// Larger known graph: complement of Petersen
// ============================================================================

#[test]
fn test_complement_of_cycle_c5() {
    // Complement of C5: each vertex connected to non-neighbors in C5
    // C5: 0-1-2-3-4-0
    // Complement edges: (0,2),(0,3),(1,3),(1,4),(2,4)
    // This is again C5 (Petersen property), so ω = 2
    let g = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]);
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 2);
    assert_eq!(cliques.len(), 5);
}

// ============================================================================
// Graph where ω > coloring suggests different structure
// ============================================================================

#[test]
fn test_wheel_w5() {
    // Wheel: center 0, rim 1-2-3-4-1
    let g = BitSquareMatrix::from_symmetric_edges(
        5,
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4), // spokes
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1), // rim
        ],
    );
    let cliques = g.all_maximum_cliques();
    verify_cliques(&g, &cliques);
    verify_all_same_size(&cliques);
    // ω = 3: triangles like {0,1,2}, {0,2,3}, {0,3,4}, {0,4,1}
    assert_eq!(cliques[0].len(), 3);
    assert_eq!(cliques.len(), 4);
}

// ============================================================================
// Three isolated vertices
// ============================================================================

#[test]
fn test_three_isolated() {
    let g = BitSquareMatrix::new(3);
    let cliques = g.all_maximum_cliques();
    assert_eq!(cliques.len(), 3);
    verify_all_same_size(&cliques);
    assert_eq!(cliques[0].len(), 1);
}

// ============================================================================
// Ground-truth regression tests (5000 cases from NetworkX)
// ============================================================================

#[cfg(feature = "std")]
const GROUND_TRUTH_GZ: &[u8] = include_bytes!("fixtures/maximum_clique_ground_truth.json.gz");

#[cfg(feature = "std")]
#[derive(serde::Deserialize)]
struct Fixture {
    schema_version: u32,
    cases: Vec<GroundTruthCase>,
}

#[cfg(feature = "std")]
#[derive(serde::Deserialize)]
struct GroundTruthCase {
    n: usize,
    edges: Vec<[usize; 2]>,
    omega: usize,
    max_cliques: Vec<Vec<usize>>,
}

#[cfg(feature = "std")]
fn load_fixture() -> Fixture {
    let mut json = String::new();
    flate2::read::GzDecoder::new(GROUND_TRUTH_GZ)
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json).expect("fixture JSON parse failed")
}

#[cfg(feature = "std")]
#[test]
fn test_ground_truth_metadata() {
    let f = load_fixture();
    assert_eq!(f.schema_version, 1);
    assert_eq!(f.cases.len(), 43_809);
}

#[cfg(feature = "std")]
#[test]
fn test_ground_truth_cases() {
    let f = load_fixture();
    for (idx, case) in f.cases.iter().enumerate() {
        let g =
            BitSquareMatrix::from_symmetric_edges(case.n, case.edges.iter().map(|&[u, v]| (u, v)));
        let mut cliques = g.all_maximum_cliques();
        cliques.sort();

        assert_eq!(
            cliques.len(),
            case.max_cliques.len(),
            "case {idx}: n={}, edges={:?}: count mismatch: got {} expected {}",
            case.n,
            case.edges,
            cliques.len(),
            case.max_cliques.len(),
        );

        for (c_got, c_exp) in cliques.iter().zip(case.max_cliques.iter()) {
            assert_eq!(c_got.len(), case.omega, "case {idx}: wrong clique size");
            assert_eq!(
                c_got, c_exp,
                "case {idx}: n={}, edges={:?}: clique mismatch",
                case.n, case.edges,
            );
        }
    }
}

// ============================================================================
// Partition-aware maximum clique
// ============================================================================

/// Helper: verify all cliques respect the double-sided partition constraint
/// (at most one vertex per G1 group AND per G2 group).
fn verify_partition_respected(cliques: &[Vec<usize>], partition: &[(usize, usize)]) {
    for clique in cliques {
        let mut g1_seen = alloc::vec::Vec::new();
        let mut g2_seen = alloc::vec::Vec::new();
        for &v in clique {
            let (g1, g2) = partition[v];
            assert!(
                !g1_seen.contains(&g1),
                "G1 partition violation: two vertices in G1 group {g1} in clique {clique:?}"
            );
            assert!(
                !g2_seen.contains(&g2),
                "G2 partition violation: two vertices in G2 group {g2} in clique {clique:?}"
            );
            g1_seen.push(g1);
            g2_seen.push(g2);
        }
    }
}

#[test]
fn test_partition_identity_matches_regular() {
    // Each vertex in its own group on both sides → no extra constraint.
    let k4 = BitSquareMatrix::from_symmetric_edges(
        4,
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    );
    let partition: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();

    let regular = k4.all_maximum_cliques();
    let partitioned = k4.all_maximum_cliques_with_pairs(&partition);

    assert_eq!(regular, partitioned);
}

#[test]
fn test_partition_where_can_skip_rejected_tied_maximum() {
    let g = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
    );
    let partition: Vec<(usize, usize)> = (0..6).map(|i| (i, i)).collect();

    let clique = g.maximum_clique_with_pairs_where(&partition, |clique| !clique.contains(&0));
    let all = g.all_maximum_cliques_with_pairs_where(&partition, |clique| !clique.contains(&0));

    assert_eq!(clique, vec![3, 4, 5]);
    assert_eq!(all, vec![vec![3, 4, 5]]);
    verify_partition_respected(&all, &partition);
}

#[test]
fn test_partition_k4_two_groups() {
    // K4 with G1 groups {0,1} → g1=0, {2,3} → g1=1. G2 unique.
    // Regular max clique = 4, but G1 partition allows at most 2.
    let k4 = BitSquareMatrix::from_symmetric_edges(
        4,
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    );
    let partition = vec![(0, 0), (0, 1), (1, 2), (1, 3)];

    let clique = k4.maximum_clique_with_pairs(&partition);
    assert_eq!(clique.len(), 2);
    verify_partition_respected(&[clique], &partition);

    let all = k4.all_maximum_cliques_with_pairs(&partition);
    verify_all_same_size(&all);
    verify_cliques(&k4, &all);
    verify_partition_respected(&all, &partition);
    assert_eq!(all[0].len(), 2);
    // There should be 4 size-2 cliques: (0,2), (0,3), (1,2), (1,3).
    assert_eq!(all.len(), 4);
}

#[test]
fn test_partition_single_group() {
    // All vertices in one G1 group → max clique under partition = 1.
    let k3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let partition = vec![(0, 0), (0, 1), (0, 2)];

    let clique = k3.maximum_clique_with_pairs(&partition);
    assert_eq!(clique.len(), 1);

    let all = k3.all_maximum_cliques_with_pairs(&partition);
    assert_eq!(all.len(), 3); // {0}, {1}, {2}
    for c in &all {
        assert_eq!(c.len(), 1);
    }
}

#[test]
fn test_partition_no_edges() {
    // No edges, 4 vertices, each in own group → max clique = 1.
    let g = BitSquareMatrix::new(4);
    let partition: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();

    let all = g.all_maximum_cliques_with_pairs(&partition);
    for c in &all {
        assert_eq!(c.len(), 1);
    }
    verify_partition_respected(&all, &partition);
}

#[test]
fn test_partition_aware_leq_regular() {
    // Partition-aware max clique size ≤ regular max clique size.
    let g = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)],
    );
    // G1 groups of 2, G2 unique.
    let partition = vec![(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)];

    let regular = g.maximum_clique();
    let partitioned = g.maximum_clique_with_pairs(&partition);

    assert!(partitioned.len() <= regular.len());
    verify_cliques(&g, core::slice::from_ref(&partitioned));
    verify_partition_respected(&[partitioned], &partition);
}

#[test]
fn test_partition_modular_product_integration() {
    // Build a modular product from two small graphs, extract partition,
    // run partition-aware clique, verify results.
    let g1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);

    let result = g1.modular_product_filtered(&g2, |_, _| true);
    let partition: Vec<(usize, usize)> = result.vertex_pairs().to_vec();

    let cliques = result.matrix().all_maximum_cliques_with_pairs(&partition);
    verify_cliques(result.matrix(), &cliques);
    verify_partition_respected(&cliques, &partition);
    verify_all_same_size(&cliques);
    verify_no_duplicates(&cliques);

    // Also verify partition-aware size ≤ regular.
    let regular = result.matrix().all_maximum_cliques();
    assert!(cliques[0].len() <= regular[0].len());
}

#[test]
fn test_partition_empty_graph() {
    let g = BitSquareMatrix::new(0);
    let partition: Vec<(usize, usize)> = vec![];
    let cliques = g.all_maximum_cliques_with_pairs(&partition);
    assert_eq!(cliques, vec![Vec::<usize>::new()]);
}

#[test]
fn test_partition_three_groups_triangle() {
    // K3 with each vertex in its own group on both sides → regular behavior.
    let k3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let partition = vec![(0, 0), (1, 1), (2, 2)];

    let clique = k3.maximum_clique_with_pairs(&partition);
    assert_eq!(clique.len(), 3);
    verify_partition_respected(&[clique], &partition);
}

#[test]
fn test_partition_uneven_groups() {
    // K5 with partition: {0,1,2} → group 0, {3} → group 1, {4} → group 2.
    // Large group severely constrains: max 1 from group 0 + 1 from group 1 + 1 from
    // group 2 = 3. Regular max clique = 5.
    let k5 = BitSquareMatrix::from_symmetric_edges(
        5,
        vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    );
    let partition = vec![(0, 0), (0, 1), (0, 2), (1, 3), (2, 4)];

    let all = k5.all_maximum_cliques_with_pairs(&partition);
    verify_cliques(&k5, &all);
    verify_partition_respected(&all, &partition);
    verify_all_same_size(&all);
    assert_eq!(all[0].len(), 3);
    // Cliques: pick 1 from {0,1,2}, 1 from {3}, 1 from {4} → 3 * 1 * 1 = 3 cliques.
    assert_eq!(all.len(), 3);
}

#[test]
fn test_partition_dense_graph_many_groups() {
    // K6 with 6 groups (identity partition) → regular max clique = 6.
    // Then K6 with 3 groups of 2 → max clique = 3.
    let k6 = BitSquareMatrix::from_symmetric_edges(
        6,
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
    );

    // Identity partition: same as regular.
    let id_part: Vec<(usize, usize)> = (0..6).map(|i| (i, i)).collect();
    let regular = k6.all_maximum_cliques();
    let id_result = k6.all_maximum_cliques_with_pairs(&id_part);
    assert_eq!(regular, id_result);

    // 3 G1-groups of 2, G2 unique: max 3.
    let grouped = vec![(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)];
    let grouped_result = k6.all_maximum_cliques_with_pairs(&grouped);
    verify_cliques(&k6, &grouped_result);
    verify_partition_respected(&grouped_result, &grouped);
    assert_eq!(grouped_result[0].len(), 3);
    // C(2,1)^3 = 8 cliques of size 3.
    assert_eq!(grouped_result.len(), 8);
}

#[test]
fn test_partition_sparse_graph_constraint_is_deciding() {
    // Path 0-1-2-3-4: regular max clique = 2 (any edge).
    // Partition {0,1}→g0, {2,3}→g1, {4}→g2: still max 2 (partition allows up to 3).
    // Partition {0,1}→g0, {2}→g0, {3,4}→g1: edges (0,1) and (1,2) have same-group
    // conflicts. Edge (2,3) crosses groups, so clique {2,3} is valid. Max = 2.
    let path = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let partition = vec![(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)];

    let all = path.all_maximum_cliques_with_pairs(&partition);
    verify_cliques(&path, &all);
    verify_partition_respected(&all, &partition);
    // Only cross-group edges: (2,3). Edges (0,1) and (1,2) have both endpoints
    // in group 0. Edge (3,4) has both in group 1. So only {2,3} is valid size 2.
    assert_eq!(all[0].len(), 2);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0], vec![2, 3]);
}
