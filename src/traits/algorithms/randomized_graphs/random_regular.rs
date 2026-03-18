//! Generator for random regular graphs via the configuration model with
//! repeated local retries.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]

use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashSet;

#[cfg(all(feature = "hashbrown", not(feature = "std")))]
use hashbrown::HashSet;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates a random `k`-regular graph on `n` vertices using the configuration
/// model with rejection sampling.
///
/// # Panics
/// Panics if `n * k` is odd, if `k >= n`, or if generation fails after 1000
/// attempts.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
#[must_use]
pub fn random_regular_graph(
    seed: u64,
    n: usize,
    k: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    assert!((n * k) % 2 == 0, "n * k must be even for a regular graph to exist");
    assert!(k < n || n == 0, "k must be strictly less than n");

    if n == 0 || k == 0 {
        return build_symmetric(n, Vec::new());
    }

    let num_stubs = n * k;
    let num_pairs = num_stubs / 2;

    for attempt in 0u64..1000 {
        let current_seed = seed.wrapping_add(attempt.wrapping_mul(0x9E37_79B9));
        let mut rng = XorShift64::from(XorShift64::normalize_seed(current_seed));

        // Create stubs: vertex i appears k times.
        let mut stubs: Vec<usize> = Vec::with_capacity(num_stubs);
        for i in 0..n {
            for _ in 0..k {
                stubs.push(i);
            }
        }

        // Fisher-Yates shuffle.
        for i in (1..num_stubs).rev() {
            let j = (rng.next().unwrap() as usize) % (i + 1);
            stubs.swap(i, j);
        }

        // Pair consecutive stubs and check for self-loops / multi-edges.
        let mut edge_set = HashSet::with_capacity(num_pairs);
        let mut valid = true;
        for pair_idx in 0..num_pairs {
            let a = stubs[2 * pair_idx];
            let b = stubs[2 * pair_idx + 1];
            if a == b {
                valid = false;
                break;
            }
            let (u, v) = if a < b { (a, b) } else { (b, a) };
            if !edge_set.insert((u, v)) {
                valid = false;
                break;
            }
        }

        if valid {
            let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
            edges.sort_unstable();
            return build_symmetric(n, edges);
        }
    }

    panic!("random_regular_graph: failed to generate a valid graph after 1000 attempts");
}
