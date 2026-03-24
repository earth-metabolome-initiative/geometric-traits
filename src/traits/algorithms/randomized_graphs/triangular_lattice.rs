//! Generator for finite triangular lattice graphs.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns a finite triangular-lattice patch on a `rows x cols` vertex grid.
///
/// The patch contains the usual horizontal and vertical grid edges, plus a
/// consistent down-right diagonal in each unit cell. If either dimension is
/// zero, the result is the empty graph.
///
/// This is a reasonable graph family for lattice models of crystalline and
/// materials-style systems. It is generally less chemically natural for
/// ordinary molecular graphs than honeycomb / benzenoid families.
/// For examples in that direction, see
/// [Gunlycke and Tseng (2016)](https://doi.org/10.1039/C6CP00205F) and
/// [Tomeno et al. (2020)](https://doi.org/10.1021/acs.inorgchem.0c00880).
#[must_use]
pub fn triangular_lattice_graph(
    rows: usize,
    cols: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
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
            if r + 1 < rows && c + 1 < cols {
                edges.push((v, v + cols + 1));
            }
        }
    }

    build_symmetric(n, edges)
}
