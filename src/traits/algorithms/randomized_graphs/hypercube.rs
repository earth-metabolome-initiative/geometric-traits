//! Generator for hypercube graphs (Q_d).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the hypercube graph Q_d with 2^d vertices.
#[must_use]
pub fn hypercube_graph(d: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = 1usize << d;
    let mut edges = Vec::with_capacity(d * n / 2);
    for v in 0..n {
        for b in 0..d {
            let w = v ^ (1 << b);
            if v < w {
                edges.push((v, w));
            }
        }
    }
    edges.sort_unstable();
    build_symmetric(n, edges)
}
