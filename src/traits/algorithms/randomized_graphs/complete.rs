//! Generator for complete graphs (K_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the complete graph K_n on `n` vertices.
#[must_use]
pub fn complete_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let mut edges = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }
    build_symmetric(n, edges)
}
