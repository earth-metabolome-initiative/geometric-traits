//! Generator for complete bipartite graphs (K_{m,n}).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the complete bipartite graph K_{m,n}.
///
/// Left vertices 0..m, right vertices m..m+n.
#[must_use]
pub fn complete_bipartite_graph(m: usize, n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let total = m + n;
    let mut edges = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            edges.push((i, m + j));
        }
    }
    build_symmetric(total, edges)
}
