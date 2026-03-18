//! Generator for the Erdos-Renyi G(n, m) random graph model.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]

use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashSet;

#[cfg(all(feature = "hashbrown", not(feature = "std")))]
use hashbrown::HashSet;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates an Erdos-Renyi G(n, m) random graph: `n` vertices and exactly `m`
/// distinct undirected edges chosen uniformly at random (no self-loops).
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn erdos_renyi_gnm(
    seed: u64,
    n: usize,
    m: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }

    let max_edges = n * (n - 1) / 2;
    let m = m.min(max_edges);

    if m == 0 {
        return build_symmetric(n, Vec::new());
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));
    let n_u64 = n as u64;

    let mut edge_set = HashSet::with_capacity(m);
    while edge_set.len() < m {
        let s1 = rng.next().unwrap();
        let s2 = rng.next().unwrap();
        let mut u = (s1 % n_u64) as usize;
        let mut v = (s2 % n_u64) as usize;
        if u == v {
            continue;
        }
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }
        edge_set.insert((u, v));
    }

    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    build_symmetric(n, edges)
}
