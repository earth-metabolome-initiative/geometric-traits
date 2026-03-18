//! Generator for star graphs (S_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the star graph on `n` vertices (hub vertex 0, n-1 leaves).
#[must_use]
pub fn star_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }
    let mut edges = Vec::with_capacity(n - 1);
    for i in 1..n {
        edges.push((0, i));
    }
    build_symmetric(n, edges)
}
