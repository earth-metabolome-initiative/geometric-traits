//! Fuzz harness for the modular product algorithm.
//!
//! Verifies symmetry, no self-loops, edge condition, result dimensions,
//! edge-count consistency, and commutativity for small random graph pairs.

use geometric_traits::{impls::BitSquareMatrix, prelude::*};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            if data.len() < 3 {
                return;
            }
            let n1 = (data[0] % 6) as usize + 2; // 2..7
            let n2 = (data[1] % 6) as usize + 2;
            let max_e1 = n1 * (n1 - 1) / 2;
            let max_e2 = n2 * (n2 - 1) / 2;
            // Third byte: low bit selects full vs partial pairs
            let use_partial_pairs = data[2] & 1 != 0;
            let needed = 3 + max_e1 + max_e2;
            if data.len() < needed {
                return;
            }

            // Build G1
            let mut edges1 = Vec::new();
            let mut idx = 3;
            for u in 0..n1 {
                for v in (u + 1)..n1 {
                    if data[idx] & 1 != 0 {
                        edges1.push((u, v));
                    }
                    idx += 1;
                }
            }
            let g1 = BitSquareMatrix::from_symmetric_edges(n1, edges1);

            // Build G2
            let mut edges2 = Vec::new();
            for u in 0..n2 {
                for v in (u + 1)..n2 {
                    if data[idx] & 1 != 0 {
                        edges2.push((u, v));
                    }
                    idx += 1;
                }
            }
            let g2 = BitSquareMatrix::from_symmetric_edges(n2, edges2);

            // Build vertex pairs (full or partial)
            let pairs: Vec<(usize, usize)> = if use_partial_pairs {
                // Use only pairs where both indices are even, or first half
                (0..n1)
                    .flat_map(|i| (0..n2).map(move |j| (i, j)))
                    .filter(|(i, j)| i % 2 == 0 || j % 2 == 0)
                    .collect()
            } else {
                (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect()
            };

            let mp = g1.modular_product(&g2, &pairs);
            let p = pairs.len();

            // Check result dimensions
            assert_eq!(mp.order(), p, "order mismatch");

            // Check invariants
            let mut edge_count = 0usize;
            for a in 0..p {
                assert!(!mp.has_entry(a, a), "self-loop");
                for b in (a + 1)..p {
                    assert_eq!(mp.has_entry(a, b), mp.has_entry(b, a), "asymmetric");
                    let (u1, u2) = pairs[a];
                    let (v1, v2) = pairs[b];
                    let expected =
                        u1 != v1 && u2 != v2 && g1.has_entry(u1, v1) == g2.has_entry(u2, v2);
                    assert_eq!(mp.has_entry(a, b), expected, "edge condition");
                    if mp.has_entry(a, b) {
                        edge_count += 2; // symmetric: counted in both
                                         // directions
                    }
                }
            }

            // Check number_of_edges consistency
            assert_eq!(mp.number_of_defined_values(), edge_count, "edge count mismatch");

            // Check commutativity (only for full pairs to keep mapping simple)
            if !use_partial_pairs {
                let pairs_rev: Vec<(usize, usize)> =
                    (0..n2).flat_map(|j| (0..n1).map(move |i| (j, i))).collect();
                let mp_rev = g2.modular_product(&g1, &pairs_rev);
                assert_eq!(mp_rev.order(), p, "reverse order mismatch");
                assert_eq!(
                    mp_rev.number_of_defined_values(),
                    mp.number_of_defined_values(),
                    "commutativity: edge count differs"
                );
            }
        });
    }
}
