//! Generator for friendship graphs (F_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the friendship graph F_n: n triangles sharing a universal hub vertex
/// 0.
///
/// 2n+1 vertices, 3n edges.
#[must_use]
pub fn friendship_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let total = 2 * n + 1;
    let mut edges = Vec::with_capacity(3 * n);
    for k in 0..n {
        let a = 2 * k + 1;
        let b = 2 * k + 2;
        edges.push((0, a));
        edges.push((0, b));
        edges.push((a, b));
    }
    edges.sort_unstable();
    build_symmetric(total, edges)
}
