//! The scaling layout and the modular multilevel mixer.
//!
//! [`multilevel_layout`] is the OGDF `ModularMultilevelMixer` walk: build the
//! coarsening hierarchy, lay out the coarsest level, then from coarsest to
//! finest run the per-level layout, recenter, and prolong. The per-level layout
//! is [`scaling_layout`] (OGDF `ScalingLayout`, `min = max = 1.0`,
//! `RelativeToDrawing`): three passes, each pinning the mean edge length back
//! to its start before one full FME solve.

// Per-axis loops over `0..2` index flat buffers in lockstep.
#![allow(clippy::needless_range_loop)]

use alloc::{vec, vec::Vec};

use num_traits::Float;

use super::{
    super::randomized_graphs::XorShift64,
    fme::FastMultipoleEmbedder,
    multilevel::{LevelGraph, build_hierarchy},
};

/// Configuration for the multilevel layout, pinned to the TMAP defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct MixerConfig<F> {
    /// Maximum FME iterations per per-level layout pass (`fme_iterations`,
    /// 1000).
    pub iterations: usize,
    /// Barnes-Hut acceptance for the FME multipole path (`fme_precision` maps
    /// to this in spirit; the port uses a `theta` cutoff, default 0.5).
    pub theta: F,
    /// Extra scaling steps of the scaling layout (`sl_extra_scaling_steps`, 2),
    /// so each level runs `extra_scaling_steps + 1` FME passes.
    pub extra_scaling_steps: usize,
    /// Per-node width and height (`node_size`, `1/65`). The FME node size is
    /// `0.5 * sqrt(2) * node_size`.
    pub node_size: F,
    /// Multipole-path repulsion factor (OGDF `repForceFactor`, 2.0).
    pub rep_factor: F,
    /// Multipole-path main-loop edge factor (OGDF, 0.5).
    pub edge_factor: F,
    /// Scale of the prolongation jitter (OGDF `randomDouble(-1, 1)`, so 1.0).
    pub jitter: F,
    /// Seed for the coarsening randomness and the prolongation jitter.
    pub seed: u64,
}

impl<F: Float> Default for MixerConfig<F> {
    fn default() -> Self {
        Self {
            iterations: 1000,
            theta: F::from(0.5).unwrap(),
            extra_scaling_steps: 2,
            node_size: F::from(1.0 / 65.0).unwrap(),
            rep_factor: F::from(2.0).unwrap(),
            edge_factor: F::from(0.5).unwrap(),
            jitter: F::one(),
            seed: 0x1234_5678_9abc_def0,
        }
    }
}

/// Lays out a single connected graph on `n` nodes with `edges` (relabeled to
/// `0..n`), returning a flat `n * 2` buffer. Use the component splitter for
/// disconnected inputs.
pub(super) fn multilevel_layout<F>(
    n: usize,
    edges: &[(usize, usize)],
    config: &MixerConfig<F>,
) -> Vec<F>
where
    F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
{
    // FME node size s = 0.5 * sqrt(w^2 + h^2) with w = h = node_size.
    let s = config.node_size * F::from(2.0).unwrap().sqrt() / F::from(2.0).unwrap();
    let mut rng = XorShift64::from(XorShift64::normalize_seed(config.seed));
    let (levels, prolongs) = build_hierarchy(n, edges, &mut rng);
    let mut fme = FastMultipoleEmbedder::<F>::new();

    let coarsest = levels.len() - 1;
    // The coarsest level starts at the origin (TMAP disables randomize). The
    // prolongation jitter breaks the degeneracy on the way down.
    let mut positions = vec![F::zero(); levels[coarsest].n * 2];

    for level in (1..=coarsest).rev() {
        run_level(&mut fme, &mut positions, &levels[level], s, config);
        move_to_zero::<F>(&mut positions, levels[level].n);
        positions = prolongs[level - 1].apply::<F>(&positions, &mut rng, config.jitter);
    }
    run_level(&mut fme, &mut positions, &levels[0], s, config);

    positions
}

/// Runs the per-level scaling layout on `positions` for `level`.
fn run_level<F>(
    fme: &mut FastMultipoleEmbedder<F>,
    positions: &mut [F],
    level: &LevelGraph,
    node_size: F,
    config: &MixerConfig<F>,
) where
    F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
{
    let sizes = vec![node_size; level.n];
    scaling_layout::<F>(
        fme,
        positions,
        &level.edges,
        &sizes,
        config.iterations,
        config.theta,
        config.extra_scaling_steps,
        config.rep_factor,
        config.edge_factor,
    );
}

/// OGDF `ScalingLayout` (`RelativeToDrawing`, `min = max = 1.0`). Runs
/// `extra_scaling_steps + 1` passes: each recenters, scales so the mean edge
/// length returns to the first pass's value, then runs one full FME solve.
#[allow(clippy::too_many_arguments)]
fn scaling_layout<F>(
    fme: &mut FastMultipoleEmbedder<F>,
    positions: &mut [F],
    edges: &[(usize, usize)],
    node_sizes: &[F],
    iterations: usize,
    theta: F,
    extra_scaling_steps: usize,
    rep_factor: F,
    edge_factor: F,
) where
    F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
{
    let n = node_sizes.len();
    let mut avg_start = F::zero();
    let mut have_start = false;
    for _ in 0..=extra_scaling_steps {
        let avg = avg_edge_length::<F>(positions, edges, n);
        if avg <= F::zero() {
            move_to_zero::<F>(positions, n);
        } else {
            if !have_start {
                avg_start = avg;
                have_start = true;
            }
            let scaling = avg_start / avg;
            move_to_zero::<F>(positions, n);
            for value in positions.iter_mut() {
                *value = *value * scaling;
            }
        }
        fme.run_with_factors(
            positions,
            edges,
            node_sizes,
            iterations,
            theta,
            rep_factor,
            edge_factor,
        );
    }
}

/// Mean edge length: the summed Euclidean edge length divided by the node count
/// (OGDF divides by node count, not edge count).
fn avg_edge_length<F: Float>(positions: &[F], edges: &[(usize, usize)], n: usize) -> F {
    if n == 0 {
        return F::zero();
    }
    let mut total = F::zero();
    for &(a, b) in edges {
        let mut dsq = F::zero();
        for d in 0..2 {
            let delta = positions[a * 2 + d] - positions[b * 2 + d];
            dsq = dsq + delta * delta;
        }
        total = total + dsq.sqrt();
    }
    total / F::from(n).unwrap()
}

/// Recenters `positions` (the first `n` points) so the centroid is at the
/// origin.
fn move_to_zero<F: Float>(positions: &mut [F], n: usize) {
    if n == 0 {
        return;
    }
    let mut center = [F::zero(); 2];
    for i in 0..n {
        for d in 0..2 {
            center[d] = center[d] + positions[i * 2 + d];
        }
    }
    let inv = F::one() / F::from(n).unwrap();
    for d in 0..2 {
        center[d] = center[d] * inv;
    }
    for i in 0..n {
        for d in 0..2 {
            positions[i * 2 + d] = positions[i * 2 + d] - center[d];
        }
    }
}
