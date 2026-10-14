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

/// Error type for random regular graph generation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RandomRegularGraphError {
    /// The total number of configuration-model stubs must be even.
    #[error("n * k must be even for a regular graph to exist (got n={n}, k={k})")]
    OddStubCount {
        /// Requested number of vertices.
        n: usize,
        /// Requested regular degree.
        k: usize,
    },
    /// A simple k-regular graph requires `k < n`, except for the empty graph.
    #[error("k must be strictly less than n for a simple regular graph (got n={n}, k={k})")]
    DegreeTooLarge {
        /// Requested number of vertices.
        n: usize,
        /// Requested regular degree.
        k: usize,
    },
    /// The requested stub count overflowed `usize`.
    #[error("n * k overflows usize for a regular graph request (got n={n}, k={k})")]
    StubCountOverflow {
        /// Requested number of vertices.
        n: usize,
        /// Requested regular degree.
        k: usize,
    },
    /// Rejection sampling failed to produce a simple graph in the retry budget.
    #[error(
        "failed to generate a valid random regular graph after {attempts} attempts (n={n}, k={k})"
    )]
    GenerationAttemptsExceeded {
        /// Requested number of vertices.
        n: usize,
        /// Requested regular degree.
        k: usize,
        /// Number of rejection-sampling attempts that were tried.
        attempts: usize,
    },
}

/// Generates a random `k`-regular graph on `n` vertices using the configuration
/// model with rejection sampling.
///
/// # Errors
/// Returns [`RandomRegularGraphError::OddStubCount`] if `n * k` is odd,
/// [`RandomRegularGraphError::DegreeTooLarge`] if `k >= n` for a non-empty
/// graph, [`RandomRegularGraphError::StubCountOverflow`] if `n * k` overflows
/// `usize`, or [`RandomRegularGraphError::GenerationAttemptsExceeded`] if
/// rejection sampling fails within the retry budget.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
pub fn random_regular_graph(
    seed: u64,
    n: usize,
    k: usize,
) -> Result<SymmetricCSR2D<CSR2D<usize, usize, usize>>, RandomRegularGraphError> {
    if n != 0 && k >= n {
        return Err(RandomRegularGraphError::DegreeTooLarge { n, k });
    }

    let num_stubs = n.checked_mul(k).ok_or(RandomRegularGraphError::StubCountOverflow { n, k })?;
    if num_stubs % 2 != 0 {
        return Err(RandomRegularGraphError::OddStubCount { n, k });
    }

    if n == 0 || k == 0 {
        return Ok(build_symmetric(n, Vec::new()));
    }

    let num_pairs = num_stubs / 2;
    let max_attempts = 1000usize;

    for attempt in 0u64..u64::try_from(max_attempts).expect("attempt budget fits in u64") {
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
            return Ok(build_symmetric(n, edges));
        }
    }

    Err(RandomRegularGraphError::GenerationAttemptsExceeded { n, k, attempts: max_attempts })
}
