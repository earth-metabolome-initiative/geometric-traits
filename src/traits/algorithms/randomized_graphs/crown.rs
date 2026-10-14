//! Generator for crown graphs (Cr_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the crown graph on 2n vertices: K_{n,n} minus a perfect matching.
///
/// Top row vertices 0..n, bottom row vertices n..2n.
/// Edge (i, n+j) exists for all i != j.
#[must_use]
pub fn crown_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(n >= 2, "crown_graph requires n >= 2");
    let total = 2 * n;
    let mut edges = Vec::with_capacity(n * (n - 1));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                edges.push((i, n + j));
            }
        }
    }
    edges.sort_unstable();
    build_symmetric(total, edges)
}
