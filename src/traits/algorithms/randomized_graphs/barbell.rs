//! Generator for barbell graphs B(k, p).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the barbell graph: two K_k cliques connected by a path of `p`
/// internal vertices.
///
/// Total vertices: 2k + p. The first clique uses vertices 0..k, the second uses
/// vertices k+p..2k+p, and the bridge path uses vertices k-1, k, k+1, ...,
/// k+p-1, k+p.
#[must_use]
pub fn barbell_graph(k: usize, p: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(k >= 1, "barbell_graph requires k >= 1");
    let n = 2 * k + p;
    let mut edges = Vec::new();
    // First clique: vertices 0..k
    for i in 0..k {
        for j in (i + 1)..k {
            edges.push((i, j));
        }
    }
    // Bridge path: k-1 -- k -- k+1 -- ... -- k+p-1 -- k+p
    // If p == 0, this is a single edge from k-1 to k.
    let bridge_start = k - 1;
    let bridge_end = k + p;
    for i in bridge_start..bridge_end {
        edges.push((i, i + 1));
    }
    // Second clique: vertices k+p..k+p+k
    let offset = k + p;
    for i in 0..k {
        for j in (i + 1)..k {
            edges.push((offset + i, offset + j));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    build_symmetric(n, edges)
}
