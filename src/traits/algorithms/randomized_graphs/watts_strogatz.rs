//! Generator for the Watts-Strogatz small-world model.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates a Watts-Strogatz small-world graph.
///
/// Starts with a ring lattice of `n` vertices each connected to `k` nearest
/// neighbors, then rewires each edge with probability `beta`.
///
/// # Panics
/// Panics if `k` is odd, `k < 2`, or `n <= k`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[must_use]
pub fn watts_strogatz(
    seed: u64,
    n: usize,
    k: usize,
    beta: f64,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(k % 2 == 0, "k must be even");
    assert!(k >= 2, "k must be at least 2");
    assert!(n > k, "n must be greater than k");

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));

    // Use a flat boolean adjacency matrix for O(1) edge lookup.
    let mut adj = alloc::vec![false; n * n];

    let set_edge = |adj: &mut Vec<bool>, u: usize, v: usize| {
        adj[u * n + v] = true;
        adj[v * n + u] = true;
    };

    let has_edge = |adj: &[bool], u: usize, v: usize| -> bool { adj[u * n + v] };

    let remove_edge = |adj: &mut Vec<bool>, u: usize, v: usize| {
        adj[u * n + v] = false;
        adj[v * n + u] = false;
    };

    // Build ring lattice: connect each vertex to k/2 neighbors on each side.
    let half_k = k / 2;
    for u in 0..n {
        for j in 1..=half_k {
            let v = (u + j) % n;
            set_edge(&mut adj, u, v);
        }
    }

    // Rewire: for each vertex u, consider its "right" neighbors (u+1..u+k/2).
    for u in 0..n {
        for j in 1..=half_k {
            let v = (u + j) % n;
            let uniform = (rng.next().unwrap() as f64) / (u64::MAX as f64);
            if uniform < beta {
                // Rewire (u, v) to (u, w) where w != u and not already connected.
                let mut w = (rng.next().unwrap() as usize) % n;
                let mut attempts = 0;
                while w == u || has_edge(&adj, u, w) {
                    w = (rng.next().unwrap() as usize) % n;
                    attempts += 1;
                    if attempts > n * 10 {
                        // Cannot find a valid target; keep original edge.
                        break;
                    }
                }
                if w != u && !has_edge(&adj, u, w) {
                    remove_edge(&mut adj, u, v);
                    set_edge(&mut adj, u, w);
                }
            }
        }
    }

    // Collect edges from adjacency matrix.
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if has_edge(&adj, u, v) {
                edges.push((u, v));
            }
        }
    }

    build_symmetric(n, edges)
}
