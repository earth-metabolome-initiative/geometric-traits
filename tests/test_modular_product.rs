//! Tests for the modular product algorithm.
#![cfg(feature = "std")]

use std::io::Read as _;

use flate2::read::GzDecoder;
use geometric_traits::{impls::BitSquareMatrix, prelude::*};
use serde::Deserialize;

// ============================================================================
// Ground-truth regression tests (10 000 cases from NetworkX)
// ============================================================================

const GROUND_TRUTH_GZ: &[u8] = include_bytes!("fixtures/modular_product_ground_truth.json.gz");

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    n1: usize,
    n2: usize,
    e1: Vec<[usize; 2]>,
    e2: Vec<[usize; 2]>,
    mp: Vec<[usize; 2]>,
}

fn fixture() -> Fixture {
    let mut json = String::new();
    GzDecoder::new(GROUND_TRUTH_GZ).read_to_string(&mut json).expect("gzip decompression failed");
    serde_json::from_str(&json)
        .expect("`tests/fixtures/modular_product_ground_truth.json.gz` must contain valid JSON")
}

fn upper_triangle_edges(m: &BitSquareMatrix) -> Vec<[usize; 2]> {
    let n = m.order();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if m.has_entry(i, j) {
                edges.push([i, j]);
            }
        }
    }
    edges
}

#[test]
fn test_ground_truth_metadata() {
    let f = fixture();
    assert_eq!(f.schema_version, 1);
    assert_eq!(f.cases.len(), 10_000);
}

#[test]
fn test_ground_truth_cases() {
    let f = fixture();
    for (idx, case) in f.cases.iter().enumerate() {
        let g1 =
            BitSquareMatrix::from_symmetric_edges(case.n1, case.e1.iter().map(|&[u, v]| (u, v)));
        let g2 =
            BitSquareMatrix::from_symmetric_edges(case.n2, case.e2.iter().map(|&[u, v]| (u, v)));
        let pairs: Vec<(usize, usize)> =
            (0..case.n1).flat_map(|i| (0..case.n2).map(move |j| (i, j))).collect();
        let mp = g1.modular_product(&g2, &pairs);
        let got = upper_triangle_edges(&mp);
        assert_eq!(
            got, case.mp,
            "case {idx}: n1={}, n2={}, e1={:?}, e2={:?}",
            case.n1, case.n2, case.e1, case.e2,
        );
    }
}

// ============================================================================
// Structural invariant tests
// ============================================================================

#[test]
fn test_empty_pairs() {
    let g = BitSquareMatrix::new(3);
    let mp = g.modular_product(&g, &[]);
    assert_eq!(mp.order(), 0);
}

#[test]
fn test_single_pair() {
    let g = BitSquareMatrix::from_symmetric_edges(2, vec![(0, 1)]);
    let mp = g.modular_product(&g, &[(0, 0)]);
    assert_eq!(mp.order(), 1);
    assert!(!mp.has_entry(0, 0));
}

#[test]
fn test_k2_vs_k2() {
    // K2 = single edge (0,1)
    let k2 = BitSquareMatrix::from_symmetric_edges(2, vec![(0, 1)]);
    // All 4 pairs: (0,0),(0,1),(1,0),(1,1)
    let pairs = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
    let mp = k2.modular_product(&k2, &pairs);
    assert_eq!(mp.order(), 4);
    // Pairs where u1!=v1 and u2!=v2:
    //   (0,0)-(1,1): edge==edge -> edge     [0-3]
    //   (0,1)-(1,0): edge==edge -> edge     [1-2]
    assert!(mp.has_entry(0, 3)); // (0,0)-(1,1)
    assert!(mp.has_entry(1, 2)); // (0,1)-(1,0)
    assert_eq!(upper_triangle_edges(&mp).len(), 2);
}

#[test]
fn test_k3_vs_k3() {
    let k3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = k3.modular_product(&k3, &pairs);
    assert_eq!(mp.order(), 9);
    // In K3 vs K3, every pair (a,b) with u1!=v1 and u2!=v2 is an edge
    // because adj(u1,v1)==adj(u2,v2) is always true==true.
    let mut expected = 0;
    for a in 0..9 {
        for b in (a + 1)..9 {
            if a / 3 != b / 3 && a % 3 != b % 3 {
                expected += 1;
            }
        }
    }
    assert_eq!(upper_triangle_edges(&mp).len(), expected);
}

#[test]
fn test_symmetry() {
    let g1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.modular_product(&g2, &pairs);
    let n = mp.order();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(mp.has_entry(i, j), mp.has_entry(j, i), "symmetry violated at ({i}, {j})");
        }
    }
}

#[test]
fn test_no_self_loops() {
    let g1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3), (0, 3)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.modular_product(&g2, &pairs);
    for i in 0..mp.order() {
        assert!(!mp.has_entry(i, i), "self-loop at {i}");
    }
}

#[test]
fn test_commutativity() {
    let g1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
    let n1 = 4usize;
    let n2 = 3usize;

    let pairs_fwd: Vec<(usize, usize)> =
        (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect();
    let mp_fwd = g1.modular_product(&g2, &pairs_fwd);

    let pairs_rev: Vec<(usize, usize)> =
        (0..n2).flat_map(|j| (0..n1).map(move |i| (j, i))).collect();
    let mp_rev = g2.modular_product(&g1, &pairs_rev);

    // Map forward edges to reverse-pair indices and compare.
    let mut fwd_edges: Vec<[usize; 2]> = Vec::new();
    for (a, &(i_a, j_a)) in pairs_fwd.iter().enumerate() {
        for (b, &(i_b, j_b)) in pairs_fwd.iter().enumerate().skip(a + 1) {
            if mp_fwd.has_entry(a, b) {
                let ra = j_a * n1 + i_a;
                let rb = j_b * n1 + i_b;
                fwd_edges.push([ra.min(rb), ra.max(rb)]);
            }
        }
    }
    fwd_edges.sort_unstable();

    let rev_edges = upper_triangle_edges(&mp_rev);
    assert_eq!(fwd_edges, rev_edges);
}

#[test]
fn test_filtered_pairs_subset() {
    let g1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3), (0, 3)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);

    let all_pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp_full = g1.modular_product(&g2, &all_pairs);

    // Subset: only pairs where i < 3 and j < 2
    let sub_pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..2).map(move |j| (i, j))).collect();
    let mp_sub = g1.modular_product(&g2, &sub_pairs);

    // Every edge in the sub-product must exist in the full product.
    let n2 = 3usize;
    for a in 0..sub_pairs.len() {
        for b in (a + 1)..sub_pairs.len() {
            if mp_sub.has_entry(a, b) {
                let (i_a, j_a) = sub_pairs[a];
                let (i_b, j_b) = sub_pairs[b];
                let fa = i_a * n2 + j_a;
                let fb = i_b * n2 + j_b;
                assert!(
                    mp_full.has_entry(fa, fb),
                    "sub-product edge ({a},{b}) = ({i_a},{j_a})-({i_b},{j_b}) \
                     not found in full product at ({fa},{fb})"
                );
            }
        }
    }
}

#[test]
fn test_single_edge_vs_empty() {
    // G1 has one edge, G2 is empty.
    // For any pair (u1,u2)-(v1,v2) with u1!=v1 and u2!=v2:
    //   adj1(u1,v1) is true for exactly the one edge, adj2(u2,v2) is always false.
    //   So edge iff adj1==adj2, i.e. iff adj1(u1,v1) is false.
    //   Edges exist only between non-adjacent G1 nodes crossed with distinct G2
    // nodes.
    let g1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1)]);
    let g2 = BitSquareMatrix::new(3);
    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.modular_product(&g2, &pairs);

    // Verify no edge connects pairs whose G1 nodes are 0-1 (the edge in G1).
    for a in 0..pairs.len() {
        for b in (a + 1)..pairs.len() {
            let (u1, u2) = pairs[a];
            let (v1, v2) = pairs[b];
            if u1 != v1 && u2 != v2 {
                let g1_adjacent = g1.has_entry(u1, v1);
                // G2 is empty, so adj2 is always false.
                // Edge iff adj1 == adj2, i.e. iff adj1 is false.
                assert_eq!(mp.has_entry(a, b), !g1_adjacent);
            } else {
                assert!(!mp.has_entry(a, b));
            }
        }
    }
}

#[test]
fn test_number_of_edges_consistency() {
    let g1 = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
    let g2 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (0, 3)]);
    let pairs: Vec<(usize, usize)> = (0..5).flat_map(|i| (0..4).map(move |j| (i, j))).collect();
    let mp = g1.modular_product(&g2, &pairs);

    // Count edges from upper triangle and compare with number_of_defined_values().
    let upper = upper_triangle_edges(&mp).len();
    // number_of_defined_values() counts directed entries (both (a,b) and (b,a)).
    assert_eq!(mp.number_of_defined_values(), upper * 2);
}
