//! Generator for the Barabasi-Albert preferential attachment model.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::{builder_utils::build_symmetric, XorShift64};

/// Generates a Barabasi-Albert preferential attachment graph.
///
/// Starts with a clique of `m + 1` vertices, then adds vertices one at a time,
/// each connecting to `m` existing vertices chosen proportional to their degree.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn barabasi_albert(
    seed: u64,
    n: usize,
    m: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(m >= 1, "m must be at least 1");

    let initial_clique = m + 1;

    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }

    // If n fits in the initial clique, build a complete graph on n vertices.
    if n <= initial_clique {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        return build_symmetric(n, edges);
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));

    // Repeated stubs list for preferential attachment.
    // Each vertex appears once per edge endpoint.
    let clique_edges = initial_clique * (initial_clique - 1) / 2;
    let total_edges_estimate = clique_edges + (n - initial_clique) * m;
    let mut stubs: Vec<usize> = Vec::with_capacity(total_edges_estimate * 2);

    // Build the initial clique edges and populate stubs.
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(total_edges_estimate);
    for u in 0..initial_clique {
        for v in (u + 1)..initial_clique {
            edges.push((u, v));
            stubs.push(u);
            stubs.push(v);
        }
    }

    // Add vertices m+1 .. n-1
    for v in initial_clique..n {
        let mut targets: Vec<usize> = Vec::with_capacity(m);
        while targets.len() < m {
            let idx = (rng.next().unwrap() as usize) % stubs.len();
            let target = stubs[idx];
            // Ensure we don't pick duplicates for this vertex.
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        for &t in &targets {
            let (u, w) = if t < v { (t, v) } else { (v, t) };
            edges.push((u, w));
            stubs.push(v);
            stubs.push(t);
        }
    }

    edges.sort_unstable();
    edges.dedup();
    build_symmetric(n, edges)
}
