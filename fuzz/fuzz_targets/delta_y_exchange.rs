//! Fuzz harness for Delta-Y exchange detection.
//!
//! For two random small graphs with random edge subsets, verifies:
//! 1. `edge_subgraph_degree_sequence` matches brute-force count.
//! 2. `has_delta_y_exchange` consistency: result equals `seq1 != seq2`.
//! 3. Symmetry: `g1.has_delta_y_exchange(e1, &g2, e2) ==
//!    g2.has_delta_y_exchange(e2, &g1, e1)`.

use geometric_traits::{impls::BitSquareMatrix, prelude::*};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            if data.is_empty() {
                return;
            }

            // Parse two graph sizes (1..8 each).
            let n1 = ((data[0] & 0x0F) % 8 + 1) as usize;
            let n2 = ((data[0] >> 4) % 8 + 1) as usize;
            let max_edges1 = n1 * (n1 - 1) / 2;
            let max_edges2 = n2 * (n2 - 1) / 2;

            // Need: 1 byte header + max_edges1 bytes (g1 edges) + max_edges2 bytes (g2
            // edges)
            //       + max_edges1 bytes (subset1) + max_edges2 bytes (subset2)
            let needed = 1 + max_edges1 + max_edges2 + max_edges1 + max_edges2;
            if data.len() < needed {
                return;
            }

            let mut offset = 1;

            // Build graph 1.
            let mut edges1 = Vec::new();
            for u in 0..n1 {
                for v in (u + 1)..n1 {
                    if data[offset] & 1 != 0 {
                        edges1.push((u, v));
                    }
                    offset += 1;
                }
            }
            let g1 = BitSquareMatrix::from_symmetric_edges(n1, edges1.clone());

            // Build graph 2.
            let mut edges2 = Vec::new();
            for u in 0..n2 {
                for v in (u + 1)..n2 {
                    if data[offset] & 1 != 0 {
                        edges2.push((u, v));
                    }
                    offset += 1;
                }
            }
            let g2 = BitSquareMatrix::from_symmetric_edges(n2, edges2.clone());

            // Select random subsets of edges.
            let sub1: Vec<(usize, usize)> = edges1
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let byte_idx = 1 + max_edges1 + max_edges2 + i;
                    byte_idx < data.len() && data[byte_idx] & 1 != 0
                })
                .map(|(_, &e)| e)
                .collect();

            let sub2: Vec<(usize, usize)> = edges2
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let byte_idx = 1 + max_edges1 + max_edges2 + max_edges1 + i;
                    byte_idx < data.len() && data[byte_idx] & 1 != 0
                })
                .map(|(_, &e)| e)
                .collect();

            // 1. Verify edge_subgraph_degree_sequence via brute-force.
            let seq1 = g1.edge_subgraph_degree_sequence(&sub1);
            let seq2 = g2.edge_subgraph_degree_sequence(&sub2);

            let bf1 = brute_force_degree_sequence(n1, &sub1);
            let bf2 = brute_force_degree_sequence(n2, &sub2);
            assert_eq!(seq1, bf1, "degree sequence mismatch for g1");
            assert_eq!(seq2, bf2, "degree sequence mismatch for g2");

            // 2. has_delta_y_exchange consistency.
            let has_exchange = g1.has_delta_y_exchange(&sub1, &g2, &sub2);
            assert_eq!(has_exchange, seq1 != seq2, "has_delta_y_exchange inconsistency");

            // 3. Symmetry.
            let has_exchange_rev = g2.has_delta_y_exchange(&sub2, &g1, &sub1);
            assert_eq!(has_exchange, has_exchange_rev, "symmetry violation");
        });
    }
}

fn brute_force_degree_sequence(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; n];
    for &(u, v) in edges {
        counts[u] += 1;
        counts[v] += 1;
    }
    let mut seq: Vec<usize> = counts.into_iter().filter(|&d| d > 0).collect();
    seq.sort_unstable();
    seq
}
