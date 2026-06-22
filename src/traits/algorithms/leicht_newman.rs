//! [`LeichtNewman`] directed community detector: the leading-eigenvector
//! spectral method of Leicht and Newman (2008, arXiv:0709.4500).
//!
//! Louvain and Leiden greedily maximize the undirected modularity. This
//! detector maximizes the directed modularity (see
//! [`DirectedModularity`](super::DirectedModularity)) by recursively bisecting
//! the graph along the leading eigenvector of the symmetric directed modularity
//! matrix `C = B + B^T`, where `B_ij = A_ij - gamma * k_i^out * k_j^in / m`.
//!
//! For a two-way split with `s_i in {-1, +1}` the modularity contribution is
//! `Q = (1 / 4m) * s^T C s`, so the most positive eigenvector of `C` gives the
//! best split. A non-positive leading eigenvalue or a non-positive gain leaves
//! the subgroup whole. An optional greedy vertex-moving pass refines each
//! split. The eigenvector comes from shifted power iteration.
//!
//! Edges are read from the graph through [`SparseValuedMatrix2D`]. The detector
//! caches only per-node strengths and partition state.

use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use num_traits::Float;
use num_traits::{AsPrimitive, ToPrimitive};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::{
    directed_modularity::DirectedView,
    modularity::{ModularityError, marker_partition, renumber_partition},
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

#[derive(Debug, Clone, PartialEq)]
/// Configuration options for the [`LeichtNewman`] directed community detector.
pub struct LeichtNewmanConfig {
    /// Resolution parameter (`gamma`) used in the directed modularity.
    pub resolution: f64,
    /// Maximum number of power iterations per leading-eigenvector solve.
    pub max_power_iterations: usize,
    /// Convergence tolerance (L1 change) for power iteration.
    pub power_tolerance: f64,
    /// Whether to fine-tune each split with greedy vertex moves.
    pub refine: bool,
    /// Maximum number of greedy refinement moves per split.
    pub max_refinement_passes: usize,
    /// Random seed used to initialize each power iteration deterministically.
    pub seed: u64,
}

impl Default for LeichtNewmanConfig {
    #[inline]
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_power_iterations: 300,
            power_tolerance: 1.0e-8,
            refine: true,
            max_refinement_passes: 50,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of the [`LeichtNewman`] directed community detector.
pub struct LeichtNewmanResult<Marker> {
    partition: Vec<Marker>,
    modularity: f64,
    number_of_communities: usize,
}

impl<Marker> LeichtNewmanResult<Marker> {
    /// Returns the community id of each node.
    #[must_use]
    #[inline]
    pub fn partition(&self) -> &[Marker] {
        &self.partition
    }

    /// Returns the directed modularity of the detected partition.
    #[must_use]
    #[inline]
    pub fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Returns the number of detected communities.
    #[must_use]
    #[inline]
    pub fn number_of_communities(&self) -> usize {
        self.number_of_communities
    }
}

/// Detects communities in a weighted directed graph with the Leicht-Newman
/// leading-eigenvector spectral method.
///
/// Entry `(i, j)` of the matrix is the weight of the directed edge from `i` to
/// `j`. Weights must be finite and strictly positive. Unlike
/// [`Louvain`](super::Louvain), the matrix need not be symmetric: edge
/// direction informs the detected communities.
pub trait LeichtNewman<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Detects communities by recursively bisecting the graph along the leading
    /// eigenvector of the directed modularity matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is invalid, the matrix is not
    /// square, a weight is non-finite, unrepresentable, or non-positive, or the
    /// number of communities does not fit into `Marker`.
    ///
    /// # Complexity
    ///
    /// O(C * I * (V + E)) time and O(V + E) space, where C is the number of
    /// accepted splits (at most V - 1), I the number of power iterations per
    /// solve, V the number of nodes and E the number of edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LeichtNewmanConfig};
    ///
    /// // Two disjoint directed 2-cycles: {0,1} and {2,3}.
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///         .expected_number_of_edges(4)
    ///         .expected_shape((4, 4))
    ///         .edges(vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)].into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    /// let result =
    ///     LeichtNewman::<usize>::leicht_newman(&edges, &LeichtNewmanConfig::default()).unwrap();
    /// assert_eq!(result.number_of_communities(), 2);
    /// assert!((result.modularity() - 0.5).abs() < 1e-9);
    /// ```
    #[inline]
    fn leicht_newman(
        &self,
        config: &LeichtNewmanConfig,
    ) -> Result<LeichtNewmanResult<Marker>, ModularityError> {
        if !config.resolution.is_finite() || config.resolution <= 0.0 {
            return Err(ModularityError::InvalidResolution);
        }
        if config.max_power_iterations == 0 {
            return Err(ModularityError::InvalidMaxPowerIterations);
        }
        if !config.power_tolerance.is_finite() || config.power_tolerance < 0.0 {
            return Err(ModularityError::InvalidPowerTolerance);
        }
        if config.refine && config.max_refinement_passes == 0 {
            return Err(ModularityError::InvalidMaxRefinementPasses);
        }

        let view = DirectedView::new(self)?;
        let mut partition = view.detect(config);
        let number_of_communities = renumber_partition(&mut partition);
        let modularity = view.modularity(&partition, config.resolution);
        let marker = marker_partition::<Marker>(&partition)?;

        Ok(LeichtNewmanResult { partition: marker, modularity, number_of_communities })
    }
}

impl<G, Marker> LeichtNewman<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}

impl<M> DirectedView<'_, M>
where
    M: SparseValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: ToPrimitive + Finite,
{
    /// Recursive spectral bisection producing a `0..number_of_communities`
    /// partition of the nodes.
    fn detect(&self, config: &LeichtNewmanConfig) -> Vec<usize> {
        let number_of_nodes = self.number_of_nodes();
        let mut partition = vec![0usize; number_of_nodes];
        if number_of_nodes == 0 {
            return partition;
        }

        let mut local_of = vec![usize::MAX; number_of_nodes];
        let mut communities: Vec<Vec<usize>> = Vec::new();
        let mut stack: Vec<Vec<usize>> = alloc::vec![(0..number_of_nodes).collect()];

        while let Some(group) = stack.pop() {
            match self.try_split(&group, config, &mut local_of) {
                Some((left, right)) => {
                    stack.push(left);
                    stack.push(right);
                }
                None => communities.push(group),
            }
        }

        for (community_id, members) in communities.into_iter().enumerate() {
            for node in members {
                partition[node] = community_id;
            }
        }
        partition
    }

    /// Attempts to bisect `group`, returning the two non-empty sub-groups when
    /// the split strictly increases the directed modularity.
    ///
    /// Sets the `local_of` markers for `group` on entry and clears them on
    /// exit.
    fn try_split(
        &self,
        group: &[usize],
        config: &LeichtNewmanConfig,
        local_of: &mut [usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let length = group.len();
        if length <= 1 || self.total_weight <= 0.0 || !self.total_weight.is_normal() {
            return None;
        }

        for (local, &node) in group.iter().enumerate() {
            local_of[node] = local;
        }
        let result = self.split_core(group, config, local_of);
        for &node in group {
            local_of[node] = usize::MAX;
        }
        result
    }

    fn split_core(
        &self,
        group: &[usize],
        config: &LeichtNewmanConfig,
        local_of: &[usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let length = group.len();

        let shift = self.spectral_shift(group, local_of);
        if shift <= 0.0 || !shift.is_finite() {
            return None;
        }

        // Correction term f_i = sum_{k in g} C_ik = (C 1_g)_i.
        let ones = alloc::vec![1.0_f64; length];
        let mut correction = vec![0.0; length];
        self.c_matvec(group, local_of, &ones, &mut correction);

        let eigenvector = self.leading_eigenvector(group, local_of, &correction, shift, config);

        let mut product = vec![0.0; length];
        self.cg_matvec(group, local_of, &correction, &eigenvector, &mut product);
        let beta: f64 = eigenvector.iter().zip(product.iter()).map(|(v, p)| *v * *p).sum();
        if beta <= 1.0e-9 {
            return None;
        }

        let mut signs: Vec<f64> =
            eigenvector.iter().map(|value| if *value >= 0.0 { 1.0 } else { -1.0 }).collect();
        if signs.iter().all(|s| *s > 0.0) || signs.iter().all(|s| *s < 0.0) {
            return None;
        }

        if config.refine {
            self.refine(group, local_of, &correction, &mut signs, config.max_refinement_passes);
        }

        let positive = signs.iter().filter(|s| **s > 0.0).count();
        if positive == 0 || positive == length {
            return None;
        }

        self.cg_matvec(group, local_of, &correction, &signs, &mut product);
        let delta: f64 = signs.iter().zip(product.iter()).map(|(s, p)| *s * *p).sum();
        if delta / (4.0 * self.total_weight) <= 1.0e-12 {
            return None;
        }

        let mut left = Vec::new();
        let mut right = Vec::new();
        for (local, &node) in group.iter().enumerate() {
            if signs[local] > 0.0 {
                left.push(node);
            } else {
                right.push(node);
            }
        }
        Some((left, right))
    }

    /// Applies the symmetric directed modularity matrix `C = B + B^T`
    /// restricted to `members` to the local vector `x`, writing `C x` into
    /// `out`.
    ///
    /// `local_of[node]` maps a node to its index within `members`, or
    /// `usize::MAX` when the node is outside the group.
    fn c_matvec(&self, members: &[usize], local_of: &[usize], x: &[f64], out: &mut [f64]) {
        let inverse_total_weight = 1.0 / self.total_weight;
        let mut dot_in = 0.0;
        let mut dot_out = 0.0;
        for (local, &node) in members.iter().enumerate() {
            dot_in += self.in_degree[node] * x[local];
            dot_out += self.out_degree[node] * x[local];
        }

        for value in out.iter_mut() {
            *value = 0.0;
        }

        // A x (out-edges) and A^T x (in-edges) accumulated together.
        for (local, &node) in members.iter().enumerate() {
            let source_value = x[local];
            self.for_each_out_edge(node, |destination, weight| {
                let destination_local = local_of[destination];
                if destination_local != usize::MAX {
                    out[local] += weight * x[destination_local];
                    out[destination_local] += weight * source_value;
                }
            });
        }

        // Subtract the rank-one (degree product) terms.
        for (local, &node) in members.iter().enumerate() {
            out[local] -= self.out_degree[node] * dot_in * inverse_total_weight
                + self.in_degree[node] * dot_out * inverse_total_weight;
        }
    }

    /// Applies the subgroup-corrected modularity matrix `C^(g) x = C x - f .*
    /// x`.
    fn cg_matvec(
        &self,
        members: &[usize],
        local_of: &[usize],
        correction: &[f64],
        x: &[f64],
        out: &mut [f64],
    ) {
        self.c_matvec(members, local_of, x, out);
        for (value, (f, xi)) in out.iter_mut().zip(correction.iter().zip(x.iter())) {
            *value -= *f * *xi;
        }
    }

    /// Gershgorin upper bound on the spectral radius of `C^(g)`, used to shift
    /// the power iteration so the most positive eigenvalue dominates.
    fn spectral_shift(&self, members: &[usize], local_of: &[usize]) -> f64 {
        let inverse_total_weight = 1.0 / self.total_weight;
        let strength_in: f64 = members.iter().map(|&node| self.in_degree[node]).sum();
        let strength_out: f64 = members.iter().map(|&node| self.out_degree[node]).sum();

        let mut adjacency_abs = vec![0.0; members.len()];
        for (local, &node) in members.iter().enumerate() {
            self.for_each_out_edge(node, |destination, weight| {
                let destination_local = local_of[destination];
                if destination_local != usize::MAX {
                    adjacency_abs[local] += weight;
                    adjacency_abs[destination_local] += weight;
                }
            });
        }

        let mut max_row_abs = 0.0_f64;
        for (local, &node) in members.iter().enumerate() {
            let rank = (self.out_degree[node] * strength_in + self.in_degree[node] * strength_out)
                * inverse_total_weight;
            max_row_abs = max_row_abs.max(adjacency_abs[local] + rank);
        }

        // The factor two covers the diagonal correction term |f_i| <= row_abs_i.
        2.0 * max_row_abs * (1.0 + 1.0e-9) + 1.0e-12
    }

    /// Leading (most positive) eigenvector of `C^(g)` via shifted power
    /// iteration, kept orthogonal to the all-ones null vector by mean
    /// subtraction.
    fn leading_eigenvector(
        &self,
        group: &[usize],
        local_of: &[usize],
        correction: &[f64],
        shift: f64,
        config: &LeichtNewmanConfig,
    ) -> Vec<f64> {
        let length = group.len();
        let mut rng = SmallRng::seed_from_u64(
            config
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(group[0] as u64)
                .wrapping_add((length as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)),
        );

        let mut current: Vec<f64> = (0..length).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        subtract_mean(&mut current);
        if !normalize(&mut current) {
            // Fall back to a deterministic non-uniform vector if the random draw
            // collapsed onto the null space.
            for (index, value) in current.iter_mut().enumerate() {
                *value = if index % 2 == 0 { 1.0 } else { -1.0 };
            }
            subtract_mean(&mut current);
            let _ = normalize(&mut current);
        }

        let mut next = vec![0.0; length];
        let convergence_threshold = config.power_tolerance * usize_to_f64(length);

        for _ in 0..config.max_power_iterations {
            self.cg_matvec(group, local_of, correction, &current, &mut next);
            for (value, x) in next.iter_mut().zip(current.iter()) {
                *value += shift * *x;
            }
            subtract_mean(&mut next);
            if !normalize(&mut next) {
                break;
            }

            let error: f64 = next.iter().zip(current.iter()).map(|(n, c)| (*n - *c).abs()).sum();
            core::mem::swap(&mut current, &mut next);
            if error < convergence_threshold {
                break;
            }
        }

        current
    }

    /// Greedy Kernighan-Lin style fine-tuning: repeatedly flip the single
    /// vertex whose move most increases the split modularity, keeping both
    /// sides non-empty. Each move strictly increases `s^T C^(g) s`, so this
    /// terminates.
    fn refine(
        &self,
        group: &[usize],
        local_of: &[usize],
        correction: &[f64],
        signs: &mut [f64],
        max_moves: usize,
    ) {
        let length = group.len();
        let inverse_total_weight = 1.0 / self.total_weight;

        // Diagonal of C^(g): (C^(g))_ii = 2 (A_ii - k_i^out k_i^in / m) - f_i.
        let mut diagonal = vec![0.0; length];
        for (local, &node) in group.iter().enumerate() {
            let mut self_loop = 0.0;
            self.for_each_out_edge(node, |destination, weight| {
                if destination == node {
                    self_loop += weight;
                }
            });
            let c_ii = 2.0
                * (self_loop - self.out_degree[node] * self.in_degree[node] * inverse_total_weight);
            diagonal[local] = c_ii - correction[local];
        }

        let mut product = vec![0.0; length];
        let mut positive = signs.iter().filter(|s| **s > 0.0).count();

        for _ in 0..max_moves {
            self.cg_matvec(group, local_of, correction, signs, &mut product);

            let mut best_gain = 1.0e-12;
            let mut best_local: Option<usize> = None;
            for local in 0..length {
                let would_empty = (signs[local] > 0.0 && positive == 1)
                    || (signs[local] < 0.0 && positive == length - 1);
                if would_empty {
                    continue;
                }
                let gain = (-4.0) * signs[local] * product[local] + 4.0 * diagonal[local];
                if gain > best_gain {
                    best_gain = gain;
                    best_local = Some(local);
                }
            }

            match best_local {
                Some(local) => {
                    if signs[local] > 0.0 {
                        positive -= 1;
                    } else {
                        positive += 1;
                    }
                    signs[local] = -signs[local];
                }
                None => break,
            }
        }
    }
}

fn subtract_mean(vector: &mut [f64]) {
    if vector.is_empty() {
        return;
    }
    let mean = vector.iter().sum::<f64>() / usize_to_f64(vector.len());
    for value in vector.iter_mut() {
        *value -= mean;
    }
}

fn normalize(vector: &mut [f64]) -> bool {
    let norm_squared =
        vector.iter().fold(0.0, |accumulator, value| value.mul_add(*value, accumulator));
    if norm_squared <= 0.0 {
        return false;
    }
    let inverse_norm = norm_squared.sqrt().recip();
    for value in vector.iter_mut() {
        *value *= inverse_norm;
    }
    true
}

/// Converts a graph size to `f64`. Graph sizes always fit.
#[inline]
fn usize_to_f64(value: usize) -> f64 {
    num_traits::NumCast::from(value).expect("graph sizes must fit into f64")
}
