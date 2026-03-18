//! Generator for path graphs (P_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the path graph P_n on `n` vertices.
#[must_use]
pub fn path_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }
    let mut edges = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        edges.push((i, i + 1));
    }
    build_symmetric(n, edges)
}
