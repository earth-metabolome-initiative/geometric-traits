//! Generator for the Configuration Model.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]

use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::HashSet;

#[cfg(all(feature = "hashbrown", not(feature = "std")))]
use hashbrown::HashSet;

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::{builder_utils::build_symmetric, XorShift64};

/// Generates a graph from the given degree sequence using the configuration model.
///
/// Creates stubs according to the degree sequence, shuffles, and pairs them.
/// Self-loops and multi-edges are silently removed to produce a simple graph.
///
/// # Panics
/// Panics if the sum of degrees is odd.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
#[must_use]
pub fn configuration_model(
    seed: u64,
    degrees: &[usize],
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = degrees.len();
    let total_stubs: usize = degrees.iter().sum();
    assert!(
        total_stubs % 2 == 0,
        "sum of degrees must be even"
    );

    if n == 0 || total_stubs == 0 {
        return build_symmetric(n, Vec::new());
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));

    // Create stubs: vertex i appears degrees[i] times.
    let mut stubs: Vec<usize> = Vec::with_capacity(total_stubs);
    for (i, &deg) in degrees.iter().enumerate() {
        for _ in 0..deg {
            stubs.push(i);
        }
    }

    // Fisher-Yates shuffle.
    for i in (1..total_stubs).rev() {
        let j = (rng.next().unwrap() as usize) % (i + 1);
        stubs.swap(i, j);
    }

    // Pair consecutive stubs, skip self-loops and multi-edges.
    let num_pairs = total_stubs / 2;
    let mut edge_set = HashSet::with_capacity(num_pairs);
    for pair_idx in 0..num_pairs {
        let a = stubs[2 * pair_idx];
        let b = stubs[2 * pair_idx + 1];
        if a == b {
            continue; // self-loop
        }
        let (u, v) = if a < b { (a, b) } else { (b, a) };
        edge_set.insert((u, v));
    }

    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    build_symmetric(n, edges)
}
