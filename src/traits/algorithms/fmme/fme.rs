//! Single-level FastMultipoleEmbedder, ported from OGDF.
//!
//! Lays out one graph in place, refining the positions it is handed (TMAP
//! disables the randomize step). Each iteration sums two force families and
//! takes one explicit Euler step:
//!
//! - Repulsion between every pair, `(s_i + s_j) / distance`, the gradient of
//!   the 2D log potential OGDF uses. A Coulomb gas under it fills a disk with
//!   near-uniform density (the space-filling look). The squared distance is
//!   floored at `0.25 * s_sum` near contact.
//! - Attraction along each edge, `0.25 * log(distance / e)` split by endpoint
//!   degree, where `e = s_a + s_b` is the desired edge length.
//!
//! [`FastMultipoleEmbedder::run`] dispatches on node count: an exact O(N^2)
//! direct sum under 100 nodes, a Barnes-Hut multipole approximation at or
//! above. 2D only, matching OGDF.

#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use barnes_hut_tree::BarnesHutTree;
use num_traits::Float;

/// OGDF `FMEGlobalOptions` physics constants. Fixed in OGDF, not exposed as
/// knobs (only the iteration count and multipole acceptance are).
mod constants {
    /// Explicit Euler step for the exact path and the multipole base step.
    pub const TIME_STEP: f64 = 0.25;
    /// Step used during the edge-only preprocessing iterations.
    pub const PREPROC_TIME_STEP: f64 = 0.5;
    /// Number of edge-only preprocessing iterations run before the main loop.
    pub const PREPROC_ITERATIONS: usize = 20;
    /// Near-contact protection factor in the repulsion denominator.
    pub const PROTECTION: f64 = 0.25;
    /// Geometric cool down applied once per multipole main iteration.
    pub const COOL_DOWN: f64 = 0.999;
    /// Minimum iterations before the stopping criterion can fire.
    pub const MIN_ITERATIONS: usize = 4;
    /// Denominator of the stopping-force threshold.
    pub const STOP_CRIT_CONST_SQ: f64 = 2_000_400.0;
    /// Node count at or above which OGDF switches to the multipole path.
    pub const MULTIPOLE_THRESHOLD: usize = 100;
    /// High-degree guard: repulsion on a node of degree above this is divided
    /// by its degree before the factor is applied (multipole path only).
    pub const HIGH_DEGREE_GUARD: usize = 100;
}

/// A precomputed edge: endpoints and desired length `e = s_a + s_b` (OGDF
/// `ArrayGraph`), stored so the force loop does not recompute it.
#[derive(Debug, Clone, Copy)]
struct EdgeInfo<F> {
    a: usize,
    b: usize,
    e: F,
}

/// The single-level FastMultipoleEmbedder. Holds the reusable [`BarnesHutTree`]
/// and a force scratch buffer so the per-iteration loop never reallocates.
#[derive(Debug)]
pub struct FastMultipoleEmbedder<F> {
    tree: BarnesHutTree<F, u64, 2>,
    forces: Vec<F>,
    rep: Vec<F>,
    edges: Vec<EdgeInfo<F>>,
    degree: Vec<F>,
}

impl<F> Default for FastMultipoleEmbedder<F>
where
    F: Float
        + num_traits::float::FloatCore
        + Send
        + Sync
        + core::ops::AddAssign
        + core::ops::SubAssign,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> FastMultipoleEmbedder<F>
where
    F: Float
        + num_traits::float::FloatCore
        + Send
        + Sync
        + core::ops::AddAssign
        + core::ops::SubAssign,
{
    /// Creates an embedder with empty scratch buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: BarnesHutTree::empty(),
            forces: Vec::new(),
            rep: Vec::new(),
            edges: Vec::new(),
            degree: Vec::new(),
        }
    }

    /// Lays out `positions` (flat `n * 2`) in place. `edges` are `(a, b)` index
    /// pairs, `node_sizes` gives each node's size `s_i` (positive),
    /// `iterations` caps the main loop, `theta` is the Barnes-Hut
    /// acceptance (multipole path only), and `rep_factor`/`edge_factor`
    /// scale the multipole-path repulsion and edge forces (OGDF
    /// `repForceFactor` and the main-loop edge factor). The desired length
    /// of edge `(a, b)` is `s_a + s_b`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_factors(
        &mut self,
        positions: &mut [F],
        edges: &[(usize, usize)],
        node_sizes: &[F],
        iterations: usize,
        theta: F,
        rep_factor: F,
        edge_factor: F,
    ) {
        let n = node_sizes.len();
        debug_assert_eq!(positions.len(), n * 2, "positions must be n * 2 long");
        if n == 0 {
            return;
        }
        if n == 1 {
            for value in positions.iter_mut().take(2) {
                *value = F::zero();
            }
            return;
        }

        // Per-edge desired length and per-node degree (OGDF ArrayGraph).
        self.edges.clear();
        self.edges.reserve(edges.len());
        self.degree.clear();
        self.degree.resize(n, F::zero());
        for &(a, b) in edges {
            let e = node_sizes[a] + node_sizes[b];
            self.edges.push(EdgeInfo { a, b, e });
            self.degree[a] += F::one();
            self.degree[b] += F::one();
        }

        self.forces.clear();
        self.forces.resize(n * 2, F::zero());

        // stopCritForce = n^2 * avgNodeSize / stopCritConstSq.
        let avg_node_size =
            node_sizes.iter().copied().fold(F::zero(), |acc, s| acc + s) / F::from(n).unwrap();
        let stop_force = F::from(n).unwrap() * F::from(n).unwrap() * avg_node_size
            / F::from(constants::STOP_CRIT_CONST_SQ).unwrap();

        if n < constants::MULTIPOLE_THRESHOLD {
            self.run_exact(positions, node_sizes, n, iterations, stop_force);
        } else {
            self.run_multipole(
                positions,
                node_sizes,
                n,
                iterations,
                theta,
                stop_force,
                rep_factor,
                edge_factor,
            );
        }
    }

    /// Exact O(N^2) path (OGDF `runSingle`). One step (0.25) for preprocessing
    /// and the main loop, with no factors and no cooling.
    fn run_exact(
        &mut self,
        positions: &mut [F],
        node_sizes: &[F],
        n: usize,
        max_iterations: usize,
        stop_force: F,
    ) {
        let t = F::from(constants::TIME_STEP).unwrap();

        // Preprocessing: 20 edge-only iterations.
        for _ in 0..constants::PREPROC_ITERATIONS {
            self.zero_forces(n);
            self.edge_forces(positions, F::one());
            move_nodes::<F>(positions, &self.forces, t, F::one(), n);
        }

        // Main loop: repulsion, attraction, step. Tracks the raw squared force
        // for the stopping criterion (metric = 1).
        for i in 0..max_iterations {
            self.zero_forces(n);
            self.repulsion_exact(positions, node_sizes, n, F::one());
            self.edge_forces(positions, F::one());
            let max_force_sq = move_nodes::<F>(positions, &self.forces, t, F::one(), n);
            if max_force_sq < stop_force && i > constants::MIN_ITERATIONS {
                break;
            }
        }
    }

    /// Multipole path (OGDF `runMultipole`). Barnes-Hut repulsion (factor 2.0),
    /// edge factor 0.5, step `0.25 * coolDown`, `coolDown` decaying 0.999 per
    /// iteration from 1.0.
    #[allow(clippy::too_many_arguments)]
    fn run_multipole(
        &mut self,
        positions: &mut [F],
        node_sizes: &[F],
        n: usize,
        max_iterations: usize,
        theta: F,
        stop_force: F,
        rep_factor: F,
        edge_factor: F,
    ) {
        let preproc_t = F::from(constants::PREPROC_TIME_STEP).unwrap();
        let base_t = F::from(constants::TIME_STEP).unwrap();
        let cool_rate = F::from(constants::COOL_DOWN).unwrap();

        // Preprocessing: 20 edge-only iterations, edge factor 0.5, step 0.5.
        for _ in 0..constants::PREPROC_ITERATIONS {
            self.zero_forces(n);
            self.edge_forces(positions, edge_factor);
            move_nodes::<F>(positions, &self.forces, preproc_t, preproc_t, n);
        }

        // Main loop. Tracks the squared displacement (force times the cooling
        // step), so the criterion fires as the cool down shrinks the step.
        let mut cool_down = F::one();
        for i in 0..max_iterations {
            cool_down = cool_down * cool_rate;
            self.zero_forces(n);
            self.repulsion_multipole(positions, node_sizes, n, theta, rep_factor);
            self.edge_forces(positions, edge_factor);
            let step = base_t * cool_down;
            let max_disp_sq = move_nodes::<F>(positions, &self.forces, step, step, n);
            // OGDF exits at `currNumIteration >= minNumIterations`, one
            // iteration before the exact path.
            if max_disp_sq < stop_force && i >= constants::MIN_ITERATIONS {
                break;
            }
        }
    }

    /// Zeros the force buffer for `n` nodes.
    fn zero_forces(&mut self, n: usize) {
        for value in self.forces.iter_mut().take(n * 2) {
            *value = F::zero();
        }
    }

    /// Exact O(N^2) repulsion, `s_sum / distance` (the 2D log-potential
    /// gradient). Per-pair scalar `s_sum / max(0.25 * s_sum, dsq)`, exactly
    /// OGDF `eval_direct`.
    fn repulsion_exact(&mut self, positions: &[F], node_sizes: &[F], n: usize, factor: F) {
        let protection = F::from(constants::PROTECTION).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut disp = [F::zero(); 2];
                let mut dsq = F::zero();
                for d in 0..2 {
                    let value = positions[i * 2 + d] - positions[j * 2 + d];
                    disp[d] = value;
                    dsq += value * value;
                }
                let s_sum = node_sizes[i] + node_sizes[j];
                let denom = Float::max(s_sum * protection, dsq);
                let f = (s_sum / denom) * factor;
                for d in 0..2 {
                    let push = disp[d] * f;
                    self.forces[i * 2 + d] += push;
                    self.forces[j * 2 + d] -= push;
                }
            }
        }
    }

    /// Multipole repulsion through the Barnes-Hut tree, masses being summed
    /// node sizes. The far-field kernel adds `mass * displacement / distance^2`
    /// (2D log-potential gradient), scaled by `factor`, with a high-degree
    /// node's repulsion divided by its degree first, matching OGDF.
    ///
    /// `distance^2` is floored at `0.5 * mean_node_size` (OGDF's near-contact
    /// cap `0.25 * (s_i + s_j)` for the uniform sizes TMAP uses). Without it a
    /// close pair gets unbounded repulsion and the layout diverges.
    fn repulsion_multipole(
        &mut self,
        positions: &[F],
        node_sizes: &[F],
        n: usize,
        theta: F,
        factor: F,
    ) {
        self.tree.rebuild(positions, node_sizes);
        let mean_size =
            node_sizes.iter().copied().fold(F::zero(), |acc, s| acc + s) / F::from(n).unwrap();
        let floor = F::from(constants::PROTECTION).unwrap() * (mean_size + mean_size);
        // OGDF charges a near pair with `s_i + s_j` but a far cell with only its
        // summed charge. For uniform sizes a leaf's near-field is twice its
        // center-of-mass charge, so leaf contributions are doubled to match,
        // setting packed-cluster density relative to the sparse periphery.
        let two = F::from(2.0).unwrap();
        let kernel = |displacement: &[F; 2],
                      distance_squared: F,
                      mass: F,
                      is_leaf: bool,
                      force: &mut [F; 2]| {
            let charge = if is_leaf { mass * two } else { mass };
            let f = charge / Float::max(distance_squared, floor);
            for d in 0..2 {
                force[d] += displacement[d] * f;
            }
        };
        // `accumulate_all` overwrites its output, so the scratch buffer only
        // needs the right length, not zeroing. A matching resize never allocates.
        if self.rep.len() != n * 2 {
            self.rep.resize(n * 2, F::zero());
        }
        self.tree.accumulate_all(positions, theta, &kernel, &mut self.rep);
        let guard = F::from(constants::HIGH_DEGREE_GUARD).unwrap();
        for i in 0..n {
            let scale = if self.degree[i] > guard { factor / self.degree[i] } else { factor };
            for d in 0..2 {
                self.forces[i * 2 + d] += self.rep[i * 2 + d] * scale;
            }
        }
    }

    /// Attractive edge forces (OGDF `eval_edges`), scaled by `factor`. For each
    /// edge the scalar is `0.25 * (log(distance) - log(e))`, split by endpoint
    /// degree, pulling the endpoints together when farther apart than `e`.
    fn edge_forces(&mut self, positions: &[F], factor: F) {
        let quarter = F::from(0.25).unwrap();
        let half = F::from(0.5).unwrap();
        for edge in &self.edges {
            let a = edge.a;
            let b = edge.b;
            let mut disp = [F::zero(); 2];
            let mut dsq = F::zero();
            for d in 0..2 {
                let value = positions[a * 2 + d] - positions[b * 2 + d];
                disp[d] = value;
                dsq += value * value;
            }
            let f = if dsq == F::zero() {
                F::zero()
            } else {
                (dsq.ln() * half - edge.e.ln()) * quarter * factor
            };
            let fa = f / self.degree[a];
            let fb = f / self.degree[b];
            for d in 0..2 {
                self.forces[a * 2 + d] -= disp[d] * fa;
                self.forces[b * 2 + d] += disp[d] * fb;
            }
        }
    }
}

/// Explicit Euler step. Moves every node by `force * step` and returns the
/// largest squared `force * metric`, for the stopping criterion. Pass
/// `metric = 1` on the exact path (tracks raw `f^2`, OGDF `move_nodes`) and
/// `metric = step` on the multipole path (tracks squared displacement, OGDF
/// `NodeMoveFunctor`), so the cooling step lets the criterion fire.
fn move_nodes<F: Float>(positions: &mut [F], forces: &[F], step: F, metric: F, n: usize) -> F {
    let mut max_sq = F::zero();
    for i in 0..n {
        let mut measured_sq = F::zero();
        for d in 0..2 {
            let f = forces[i * 2 + d];
            let measured = f * metric;
            measured_sq = measured_sq + measured * measured;
            positions[i * 2 + d] = positions[i * 2 + d] + f * step;
        }
        if measured_sq > max_sq {
            max_sq = measured_sq;
        }
    }
    max_sq
}

#[cfg(test)]
mod tests {
    // Test-only numeric casts between integer generators and floating-point
    // coordinates.
    #![allow(clippy::cast_precision_loss)]
    use alloc::vec;

    use super::*;

    /// OGDF node size for a `1/65` square node: `s = 0.5 * sqrt(w^2 + h^2)`.
    fn node_size() -> f64 {
        (1.0 / 65.0) * (2.0f64).sqrt() / 2.0
    }

    /// Regression against the OGDF `FastMultipoleEmbedder` single-level oracle
    /// (`ogdf-harness/fme_single`). A 4-leaf star with these exact initial
    /// positions and node sizes, run for 5 iterations, must reproduce the OGDF
    /// coordinates to within f32 rounding (OGDF computes in `float`). This pins
    /// the exact O(N^2) force model: repulsion, degree-split attraction, the 20
    /// preprocessing iterations, and the move rule.
    #[test]
    fn matches_ogdf_star_oracle() {
        let s = node_size();
        let mut positions = vec![0.05, 0.05, 1.0, 0.0, -1.0, 0.1, 0.1, 1.0, 0.0, -1.0];
        let edges = [(0, 1), (0, 2), (0, 3), (0, 4)];
        let sizes = vec![s; 5];
        let mut fme = FastMultipoleEmbedder::<f64>::new();
        fme.run_with_factors(&mut positions, &edges, &sizes, 5, 0.5, 2.0, 0.5);

        let expected = [
            0.035439838,
            0.035867669,
            0.320673585,
            0.024086071,
            -0.248787373,
            0.052644137,
            0.052533429,
            0.320791483,
            0.024114916,
            -0.248680934,
        ];
        for (got, want) in positions.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "FME diverged from OGDF oracle: got {got}, want {want}"
            );
        }
    }

    /// A path settles onto a nearly straight, evenly spaced line: the two ends
    /// spread apart and the interior gaps equalize.
    #[test]
    fn path_unfolds_to_a_line() {
        let s = node_size();
        let n = 8;
        let mut positions = Vec::new();
        for i in 0..n {
            positions.push(i as f64 * 0.4);
            positions.push(if i % 2 == 0 { -0.2 } else { 0.2 });
        }
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let sizes = vec![s; n];
        let mut fme = FastMultipoleEmbedder::<f64>::new();
        fme.run_with_factors(&mut positions, &edges, &sizes, 60, 0.5, 2.0, 0.5);

        // The layout is nearly one-dimensional: the perpendicular spread is far
        // smaller than the extent along the backbone.
        let xs: Vec<f64> = (0..n).map(|i| positions[i * 2]).collect();
        let ys: Vec<f64> = (0..n).map(|i| positions[i * 2 + 1]).collect();
        let x_range = xs.iter().copied().fold(f64::MIN, f64::max)
            - xs.iter().copied().fold(f64::MAX, f64::min);
        let y_range = ys.iter().copied().fold(f64::MIN, f64::max)
            - ys.iter().copied().fold(f64::MAX, f64::min);
        assert!(x_range > 10.0 * y_range, "path did not flatten: x={x_range}, y={y_range}");
    }

    /// Every produced coordinate is finite for a mid-size graph that takes the
    /// multipole path (at or above 100 nodes).
    #[test]
    fn multipole_path_stays_finite() {
        let s = node_size();
        let n = 200;
        // A caterpillar: a backbone with one leaf per spine node.
        let spine = n / 2;
        let mut positions = Vec::new();
        let mut seed = 0x1234_5678u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n {
            positions.push(rng() * 4.0 - 2.0);
            positions.push(rng() * 4.0 - 2.0);
        }
        let mut edges = Vec::new();
        for i in 0..spine - 1 {
            edges.push((i, i + 1));
        }
        for i in 0..spine {
            edges.push((i, spine + i));
        }
        let sizes = vec![s; n];
        let mut fme = FastMultipoleEmbedder::<f64>::new();
        fme.run_with_factors(&mut positions, &edges, &sizes, 100, 0.3, 2.0, 0.5);
        assert!(positions.iter().all(|value| value.is_finite()));
    }
}
