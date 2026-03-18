//! Generator for the Stochastic Block Model (SBM).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::{builder_utils::build_symmetric, XorShift64};

/// Generates a Stochastic Block Model graph.
///
/// Vertices are partitioned into communities defined by `sizes`. Edges between
/// vertices in the same community occur with probability `p_intra`, and edges
/// between different communities with probability `p_inter`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn stochastic_block_model(
    seed: u64,
    sizes: &[usize],
    p_intra: f64,
    p_inter: f64,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n: usize = sizes.iter().sum();

    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }

    // Build community assignment: community[v] = community index.
    let mut community: Vec<usize> = Vec::with_capacity(n);
    for (c, &size) in sizes.iter().enumerate() {
        for _ in 0..size {
            community.push(c);
        }
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));
    let mut edges = Vec::new();

    for u in 0..n {
        for v in (u + 1)..n {
            let p = if community[u] == community[v] {
                p_intra
            } else {
                p_inter
            };
            let uniform = (rng.next().unwrap() as f64) / (u64::MAX as f64);
            if uniform < p {
                edges.push((u, v));
            }
        }
    }

    build_symmetric(n, edges)
}
