//! Generator for the Erdos-Renyi G(n, p) random graph model.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates an Erdos-Renyi G(n, p) random graph: each possible edge exists
/// independently with probability `p`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn erdos_renyi_gnp(seed: u64, n: usize, p: f64) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 || p <= 0.0 {
        return build_symmetric(n, Vec::new());
    }

    // Complete graph when p >= 1.0
    if p >= 1.0 {
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        return build_symmetric(n, edges);
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));
    let total_pairs = n * (n - 1) / 2;
    let ln_1_minus_p = (1.0 - p).ln();

    let mut edges = Vec::new();
    let mut pos: isize = -1;

    loop {
        let uniform = (rng.next().unwrap() as f64) / (u64::MAX as f64);
        // Avoid log(0)
        let u_clamped = if uniform <= 0.0 { f64::MIN_POSITIVE } else { uniform };
        let skip = (u_clamped.ln() / ln_1_minus_p).floor() as isize;
        pos += 1 + skip;
        if pos >= total_pairs as isize {
            break;
        }
        let k = pos as usize;
        // Convert linear index k to (u, v) with u < v.
        // Row u is the largest integer such that u*(u-1)/2 <= k.
        #[allow(clippy::manual_midpoint)]
        let row = ((1.0 + (1.0 + 8.0 * k as f64).sqrt()) / 2.0).floor() as usize;
        // Adjust: row might overshoot by 1 due to floating-point.
        let row = if row * (row - 1) / 2 > k { row - 1 } else { row };
        let col = k - row * (row - 1) / 2;
        edges.push((col, row));
    }

    edges.sort_unstable();
    build_symmetric(n, edges)
}
