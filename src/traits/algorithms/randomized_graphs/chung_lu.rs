//! Generator for the Chung-Lu random graph model.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates a Chung-Lu random graph from the given weight sequence.
///
/// Edge `(i, j)` exists with probability `w_i * w_j / sum_w`, capped at 1.0.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn chung_lu(seed: u64, weights: &[f64]) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = weights.len();

    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }

    let sum_w: f64 = weights.iter().sum();

    if sum_w <= 0.0 {
        return build_symmetric(n, Vec::new());
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let p = (weights[i] * weights[j] / sum_w).min(1.0);
            if p <= 0.0 {
                continue;
            }
            let uniform = (rng.next().unwrap() as f64) / (u64::MAX as f64);
            if uniform < p {
                edges.push((i, j));
            }
        }
    }

    build_symmetric(n, edges)
}
