//! Generator for finite hexagonal / honeycomb lattice graphs.
#![cfg(feature = "alloc")]

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

const CORNER_OFFSETS: [(i64, i64, i64); 6] =
    [(2, -1, -1), (1, 1, -2), (-1, 2, -1), (-2, 1, 1), (-1, -1, 2), (1, -2, 1)];

/// Returns a finite hexagonal lattice patch with `rows * cols` hexagonal faces.
///
/// The hexagons are arranged as an axial `rows x cols` parallelogram. If either
/// dimension is zero, the result is the empty graph. In particular,
/// `hexagonal_lattice_graph(1, 1)` is a single 6-cycle.
///
/// This family is also called the honeycomb lattice; in chemical graph theory,
/// finite connected patches are benzenoid / polyhex graphs.
/// For benzenoid and matching-polynomial context, see
/// [Gutman (1983)](https://doi.org/10.1039/F29837900337),
/// [Gutman, Vukičević, Graovac, and Randić (2004)](https://doi.org/10.1021/ci030417z),
/// and [Zhao et al. (2014)](https://doi.org/10.1371/journal.pone.0102043).
/// For a materials-facing honeycomb-lattice reference, see
/// [Lee et al. (2015)](https://doi.org/10.1038/srep11512).
#[must_use]
pub fn hexagonal_lattice_graph(
    rows: usize,
    cols: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if rows == 0 || cols == 0 {
        return build_symmetric(0, Vec::new());
    }

    let mut vertex_ids = BTreeMap::new();
    let mut edges = BTreeSet::new();
    let mut next_id = 0;

    for row in 0..rows {
        for col in 0..cols {
            let x = i64::try_from(col).expect("hexagonal_lattice_graph columns exceed i64");
            let z = i64::try_from(row).expect("hexagonal_lattice_graph rows exceed i64");
            let y = -x - z;
            let mut corners = [0; 6];

            for (index, (dx, dy, dz)) in CORNER_OFFSETS.into_iter().enumerate() {
                let corner = (3 * x + dx, 3 * y + dy, 3 * z + dz);
                let vertex = *vertex_ids.entry(corner).or_insert_with(|| {
                    let vertex = next_id;
                    next_id += 1;
                    vertex
                });
                corners[index] = vertex;
            }

            for index in 0..6 {
                let left = corners[index];
                let right = corners[(index + 1) % 6];
                edges.insert(if left < right { (left, right) } else { (right, left) });
            }
        }
    }

    build_symmetric(next_id, edges.into_iter().collect())
}
