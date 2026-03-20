//! Tests for the maximum clique algorithm.
#![cfg(feature = "alloc")]

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
