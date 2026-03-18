//! Generator for torus graphs (T_{rows x cols}).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the torus graph with `rows` rows and `cols` columns.
#[must_use]
pub fn torus_graph(rows: usize, cols: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            // Horizontal neighbor (with wraparound)
            let right = r * cols + (c + 1) % cols;
            if v < right {
                edges.push((v, right));
            } else if right < v {
                edges.push((right, v));
            }
            // Vertical neighbor (with wraparound)
            let down = ((r + 1) % rows) * cols + c;
            if v < down {
                edges.push((v, down));
            } else if down < v {
                edges.push((down, v));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    build_symmetric(n, edges)
}
