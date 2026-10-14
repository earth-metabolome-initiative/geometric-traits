//! Generator for windmill graphs.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the windmill graph formed by `num_cliques` copies of
/// `K_clique_size` sharing a single hub vertex `0`.
///
/// This crate uses `(num_cliques, clique_size)` for clarity, although the
/// literature often writes the family as `Wd(k, n)`.
///
/// Total vertices: `1 + num_cliques * (clique_size - 1)`.
/// Total edges: `num_cliques * clique_size * (clique_size - 1) / 2`.
///
/// Special cases:
/// - `windmill_graph(1, k) = K_k`
/// - `windmill_graph(n, 2)` is a star on `n + 1` vertices
#[must_use]
pub fn windmill_graph(
    num_cliques: usize,
    clique_size: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(num_cliques >= 1, "windmill_graph requires num_cliques >= 1");
    assert!(clique_size >= 2, "windmill_graph requires clique_size >= 2");

    let blade_vertices = clique_size - 1;
    let total = 1 + num_cliques * blade_vertices;
    let edges_per_blade = clique_size * blade_vertices / 2;
    let mut edges = Vec::with_capacity(num_cliques * edges_per_blade);

    for blade in 0..num_cliques {
        let start = 1 + blade * blade_vertices;
        let end = start + blade_vertices;

        for vertex in start..end {
            edges.push((0, vertex));
        }

        for i in start..end {
            for j in (i + 1)..end {
                edges.push((i, j));
            }
        }
    }

    edges.sort_unstable();
    build_symmetric(total, edges)
}
