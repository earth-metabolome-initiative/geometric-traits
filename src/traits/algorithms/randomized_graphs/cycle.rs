//! Generator for cycle graphs (C_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the cycle graph C_n on `n` vertices.
#[must_use]
pub fn cycle_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n < 3 {
        return build_symmetric(n, Vec::new());
    }
    let mut edges = Vec::with_capacity(n);
    // Wrap-around edge (0, n-1) comes first in sorted order.
    edges.push((0, n - 1));
    for i in 0..(n - 1) {
        edges.push((i, i + 1));
    }
    edges.sort_unstable();
    build_symmetric(n, edges)
}
