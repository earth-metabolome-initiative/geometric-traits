//! Generator for Turan graphs T(n, r).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::builder_utils::build_symmetric;
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the Turan graph T(n, r): the complete r-partite graph with balanced
/// parts.
///
/// Vertices are partitioned into r groups as evenly as possible, with edges
/// between all pairs of vertices in different groups.
#[must_use]
pub fn turan_graph(n: usize, r: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!(r >= 1, "turan_graph requires r >= 1");
    // Assign each vertex to a group.
    let mut group = Vec::with_capacity(n);
    let large = n % r; // number of groups with ceil(n/r) members
    let small_size = n / r;
    let mut vertex = 0;
    for g in 0..r {
        let size = if g < large { small_size + 1 } else { small_size };
        for _ in 0..size {
            group.push(g);
            vertex += 1;
        }
    }
    let _ = vertex;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if group[i] != group[j] {
                edges.push((i, j));
            }
        }
    }
    build_symmetric(n, edges)
}
