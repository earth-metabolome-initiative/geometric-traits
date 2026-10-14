//! Generator for grid graphs (G_{rows x cols}).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the grid graph with `rows` rows and `cols` columns.
#[must_use]
pub fn grid_graph(rows: usize, cols: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                edges.push((v, v + 1));
            }
            if r + 1 < rows {
                edges.push((v, v + cols));
            }
        }
    }
    build_symmetric(n, edges)
}
