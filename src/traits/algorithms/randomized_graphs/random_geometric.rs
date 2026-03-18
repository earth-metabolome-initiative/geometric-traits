//! Generator for Random Geometric Graphs (RGG).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates a random geometric graph by placing `n` points uniformly in the
/// unit square and connecting pairs within Euclidean distance `radius`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn random_geometric_graph(
    seed: u64,
    n: usize,
    radius: f64,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 || radius <= 0.0 {
        return build_symmetric(n, Vec::new());
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));

    // Generate positions in [0, 1) x [0, 1).
    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = (rng.next().unwrap() as f64) / (u64::MAX as f64);
        let y = (rng.next().unwrap() as f64) / (u64::MAX as f64);
        positions.push((x, y));
    }

    let r_sq = radius * radius;
    let mut edges = Vec::new();

    for (i, &(xi, yi)) in positions.iter().enumerate() {
        for (j, &(xj, yj)) in positions.iter().enumerate().skip(i + 1) {
            let dx = xi - xj;
            let dy = yi - yj;
            if dx * dx + dy * dy <= r_sq {
                edges.push((i, j));
            }
        }
    }

    build_symmetric(n, edges)
}
