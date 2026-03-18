//! Generator for wheel graphs (W_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the wheel graph W_n: a hub vertex 0 connected to a rim cycle of n vertices.
///
/// Total vertices: n+1, total edges: 2n.
#[must_use]
pub fn wheel_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(n >= 3, "wheel_graph requires n >= 3");
    let total = n + 1;
    let mut edges = Vec::with_capacity(2 * n);
    // Hub edges: (0, i) for 1 <= i <= n
    for i in 1..=n {
        edges.push((0, i));
    }
    // Rim edges: (i, i+1) for 1 <= i < n, plus wrap (1, n)
    edges.push((1, n));
    for i in 1..n {
        edges.push((i, i + 1));
    }
    edges.sort_unstable();
    build_symmetric(total, edges)
}
