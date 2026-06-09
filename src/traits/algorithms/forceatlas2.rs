//! ForceAtlas2 force-directed graph layout.
//!
//! ForceAtlas2 (FA2) is the continuous force-directed layout algorithm
//! designed for the Gephi software. Connected nodes attract each other, all
//! node pairs repulse proportionally to the product of their degree-based
//! masses, and a gravity force keeps disconnected components from drifting
//! away. An adaptive global speed regulates convergence by trading
//! oscillation ("swinging") against useful movement ("traction").
//!
//! This implementation follows the semantics of the current Gephi reference
//! implementation (`ForceAtlas2.java` and `ForceFactory.java` in the
//! gephi/gephi repository) rather than the published equations wherever the
//! two disagree, and is cross-validated against snapshots produced by the
//! gephi-toolkit and by the Python `fa2` package (see
//! `tests/test_forceatlas2_ground_truth.rs`). Unlike the Java reference, all
//! arithmetic is performed in pure `f64` (no intermediate `f32` truncation)
//! and divisions are guarded so that the layout never produces a non-finite
//! coordinate.
//!
//! Gephi options that are intentionally not implemented: fixed (pinned)
//! nodes, edge weight normalization, inverted edge weights, and dynamic
//! (time-interval) weights.
//!
//! # Complexity
//!
//! O(I * (V^2 + E)) time with exact repulsion and O(I * (V log V + E))
//! with the Barnes-Hut approximation, where I is the number of iterations,
//! and O(V + E) space. The crossover where Barnes-Hut wins is around 1000
//! nodes, which is also Gephi's auto-enable threshold.
//!
//! # Reference
//!
//! Jacomy M, Venturini T, Heymann S, Bastian M (2014). ForceAtlas2, a
//! Continuous Graph Layout Algorithm for Handy Network Visualization
//! Designed for the Gephi Software. PLoS ONE 9(6): e98679.

mod forces;
mod quadtree;
mod speed;
mod state;

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};
use state::LayoutState;

use crate::traits::{Finite, SparseValuedMatrix2D};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the ForceAtlas2 layout.
///
/// The defaults mirror the constants of the Gephi reference implementation
/// for graphs with at least 100 nodes. Gephi adaptively raises the scaling
/// ratio to 10.0 for graphs under 100 nodes, while this implementation
/// always defaults to 2.0 and leaves such tuning to the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceAtlas2Config {
    /// Number of iterations to run (default: 100).
    ///
    /// ForceAtlas2 has no convergence criterion by design ("our layout stops
    /// exclusively at the user's request"). A value of 0 is allowed and
    /// returns the initial positions untouched, which is useful to obtain
    /// the seeded random initialization. Callers wanting custom stopping
    /// logic can run in chunks and inspect
    /// [`final_swinging`](ForceAtlas2Result::final_swinging) and
    /// [`final_traction`](ForceAtlas2Result::final_traction).
    pub iterations: usize,
    /// Repulsion scaling constant `kr` (default: 2.0).
    ///
    /// Larger values produce a sparser layout. There is deliberately no
    /// attraction constant in ForceAtlas2. The repulsion scaling is the
    /// only spatialization constant.
    pub scaling_ratio: f64,
    /// Gravity constant `kg` (default: 1.0).
    ///
    /// Attracts every node toward the origin with force `kg * mass`,
    /// independent of distance, preventing disconnected components from
    /// drifting away. Must be finite and non-negative.
    pub gravity: f64,
    /// Strong gravity mode (default: false).
    ///
    /// When enabled, the gravity force becomes `kg * mass * distance`,
    /// growing linearly with the distance from the origin. Produces a very
    /// compact layout.
    pub strong_gravity: bool,
    /// Tolerance to swinging, called "Tolerance (speed)" in Gephi
    /// (default: 1.0).
    ///
    /// Larger values allow more oscillation in exchange for faster
    /// convergence. Must be finite and strictly positive.
    pub jitter_tolerance: f64,
    /// Edge weight influence exponent `delta` (default: 1.0).
    ///
    /// Attraction along an edge of weight `w` is multiplied by `w^delta`.
    /// A value of 0 ignores weights entirely and 1 uses the raw weight.
    /// Must be finite and non-negative.
    pub edge_weight_influence: f64,
    /// LinLog mode (default: false).
    ///
    /// Uses the logarithmic attraction `log(1 + d)` instead of the linear
    /// `d`, which produces tighter clusters. Requires re-tuning the scaling
    /// ratio.
    pub lin_log: bool,
    /// Dissuade hubs mode, called "outbound attraction distribution" in
    /// Gephi (default: false).
    ///
    /// Divides the attraction force exerted on each edge by the mass of its
    /// source node, compensated globally by the mean node mass. Pushes hubs
    /// to the periphery and keeps authorities central.
    pub dissuade_hubs: bool,
    /// Barnes-Hut approximation for the repulsion force (default: false).
    ///
    /// Replaces the exact O(V^2) pairwise repulsion with an O(V log V)
    /// quadtree approximation. Gephi enables it automatically for graphs
    /// with at least 1000 nodes. The approximation slightly perturbs the
    /// layout but not its overall shape.
    pub barnes_hut: bool,
    /// Barnes-Hut precision parameter theta (default: 1.2).
    ///
    /// A region is approximated as a single super-node when `distance *
    /// theta > size`, so larger values mean more approximation (this is
    /// the inverse of the textbook `s/d < theta` convention, hence the
    /// unusual default). Must be finite and strictly positive, and is
    /// validated even when [`barnes_hut`](Self::barnes_hut) is off.
    pub barnes_hut_theta: f64,
    /// Node sizes (radii) enabling the anti-collision "prevent overlap"
    /// mode, called adjustSizes in Gephi (default: `None`).
    ///
    /// When provided, the length must equal the number of nodes and every
    /// size must be finite and non-negative. Repulsion and attraction then
    /// gate on the border-to-border distance `d - size1 - size2`,
    /// overlapping nodes get a strong constant repulsion kick and no
    /// attraction, the per-node speed is divided by 10 and the per-node
    /// displacement is capped at 10 units per iteration. The Barnes-Hut
    /// region approximation ignores sizes (Gephi parity), only exact
    /// node-to-node interactions use them. This mode adds considerable
    /// friction. The reference guidance is to enable it only after the
    /// layout has converged.
    pub node_sizes: Option<Vec<f64>>,
    /// Seed for the random initial positions (default: 42).
    ///
    /// Ignored when [`initial_positions`](Self::initial_positions) is
    /// provided. Given the same seed, graph and configuration, the layout is
    /// fully deterministic.
    pub seed: u64,
    /// Explicit initial positions, one `[x, y]` pair per node (default:
    /// `None`).
    ///
    /// When `None`, positions are drawn uniformly from `[-0.5, 0.5)` per
    /// axis with a deterministic RNG seeded by [`seed`](Self::seed). When
    /// provided, the length must equal the number of nodes and every
    /// coordinate must be finite.
    pub initial_positions: Option<Vec<[f64; 2]>>,
}

impl Default for ForceAtlas2Config {
    #[inline]
    fn default() -> Self {
        Self {
            iterations: 100,
            scaling_ratio: 2.0,
            gravity: 1.0,
            strong_gravity: false,
            jitter_tolerance: 1.0,
            edge_weight_influence: 1.0,
            lin_log: false,
            dissuade_hubs: false,
            barnes_hut: false,
            barnes_hut_theta: 1.2,
            node_sizes: None,
            seed: 42,
            initial_positions: None,
        }
    }
}

/// Validates the scalar configuration values.
fn validate_config(config: &ForceAtlas2Config) -> Result<(), ForceAtlas2Error> {
    if !config.scaling_ratio.is_finite() || config.scaling_ratio <= 0.0 {
        return Err(ForceAtlas2Error::InvalidScalingRatio);
    }
    if !config.gravity.is_finite() || config.gravity < 0.0 {
        return Err(ForceAtlas2Error::InvalidGravity);
    }
    if !config.jitter_tolerance.is_finite() || config.jitter_tolerance <= 0.0 {
        return Err(ForceAtlas2Error::InvalidJitterTolerance);
    }
    if !config.edge_weight_influence.is_finite() || config.edge_weight_influence < 0.0 {
        return Err(ForceAtlas2Error::InvalidEdgeWeightInfluence);
    }
    if !config.barnes_hut_theta.is_finite() || config.barnes_hut_theta <= 0.0 {
        return Err(ForceAtlas2Error::InvalidBarnesHutTheta);
    }
    Ok(())
}

// ============================================================================
// Result
// ============================================================================

/// Result of a ForceAtlas2 layout.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceAtlas2Result {
    /// Flat coordinate storage: node *i*, dimension *d* maps to `coords[i *
    /// dims + d]`.
    coordinates: Vec<f64>,
    /// Number of nodes.
    n: usize,
    /// Number of dimensions.
    dims: usize,
    /// Mass-weighted global swinging of the last iteration.
    final_swinging: f64,
    /// Mass-weighted global effective traction of the last iteration.
    final_traction: f64,
    /// Number of iterations actually run.
    iterations_run: usize,
}

impl ForceAtlas2Result {
    /// Returns the coordinates of node *i* (length =
    /// [`dimensions`](Self::dimensions)).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_points()`.
    #[must_use]
    #[inline]
    pub fn point(&self, i: usize) -> &[f64] {
        let start = i * self.dims;
        &self.coordinates[start..start + self.dims]
    }

    /// Returns the flat coordinate storage (length = `num_points() *
    /// dimensions()`).
    #[must_use]
    #[inline]
    pub fn coordinates_flat(&self) -> &[f64] {
        &self.coordinates
    }

    /// Returns the number of embedding dimensions (currently always 2).
    #[must_use]
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dims
    }

    /// Returns the number of laid-out nodes.
    #[must_use]
    #[inline]
    pub fn num_points(&self) -> usize {
        self.n
    }

    /// Returns the mass-weighted global swinging of the last iteration.
    ///
    /// Swinging measures erratic oscillation. Together with
    /// [`final_traction`](Self::final_traction) it can drive caller-side
    /// convergence detection. Zero when no iterations were run.
    #[must_use]
    #[inline]
    pub fn final_swinging(&self) -> f64 {
        self.final_swinging
    }

    /// Returns the mass-weighted global effective traction of the last
    /// iteration.
    ///
    /// Traction measures useful, convergent movement. Zero when no
    /// iterations were run.
    #[must_use]
    #[inline]
    pub fn final_traction(&self) -> f64 {
        self.final_traction
    }

    /// Returns the number of iterations actually run.
    #[must_use]
    #[inline]
    pub fn iterations_run(&self) -> usize {
        self.iterations_run
    }
}

// ============================================================================
// Error
// ============================================================================

/// Errors that can occur during a ForceAtlas2 layout.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ForceAtlas2Error {
    /// The adjacency matrix must be square.
    #[error("The adjacency matrix must be square, but has {rows} rows and {columns} columns.")]
    NonSquareMatrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// The graph has no nodes.
    #[error("The graph has no nodes.")]
    EmptyGraph,
    /// The matrix does not represent an undirected graph.
    #[error(
        "The matrix is not symmetric: edge ({source_id}, {destination_id}) has no matching reverse edge."
    )]
    NonSymmetricEdge {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// The edge weight cannot be represented as `f64`.
    #[error(
        "Found an edge weight on ({source_id}, {destination_id}) that cannot be represented as f64."
    )]
    UnrepresentableWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// The edge weight is not finite.
    #[error("Found a non-finite edge weight on ({source_id}, {destination_id}).")]
    NonFiniteWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// The edge weight is negative.
    #[error("Found a negative edge weight on ({source_id}, {destination_id}).")]
    NegativeWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// The scaling ratio must be finite and strictly positive.
    #[error("The scaling ratio must be finite and strictly positive.")]
    InvalidScalingRatio,
    /// The gravity constant must be finite and non-negative.
    #[error("The gravity constant must be finite and non-negative.")]
    InvalidGravity,
    /// The jitter tolerance must be finite and strictly positive.
    #[error("The jitter tolerance must be finite and strictly positive.")]
    InvalidJitterTolerance,
    /// The edge weight influence must be finite and non-negative.
    #[error("The edge weight influence must be finite and non-negative.")]
    InvalidEdgeWeightInfluence,
    /// The Barnes-Hut theta must be finite and strictly positive.
    #[error("The Barnes-Hut theta must be finite and strictly positive.")]
    InvalidBarnesHutTheta,
    /// The provided initial positions have the wrong length.
    #[error("Expected {expected} initial positions, but {actual} were provided.")]
    InitialPositionsLengthMismatch {
        /// Expected number of positions (one per node).
        expected: usize,
        /// Provided number of positions.
        actual: usize,
    },
    /// A provided initial position is not finite.
    #[error("The initial position of node {index} is not finite.")]
    NonFiniteInitialPosition {
        /// Node index with the offending position.
        index: usize,
    },
    /// The provided node sizes have the wrong length.
    #[error("Expected {expected} node sizes, but {actual} were provided.")]
    NodeSizesLengthMismatch {
        /// Expected number of sizes (one per node).
        expected: usize,
        /// Provided number of sizes.
        actual: usize,
    },
    /// A provided node size is not finite or is negative.
    #[error("The size of node {index} must be finite and non-negative.")]
    InvalidNodeSize {
        /// Node index with the offending size.
        index: usize,
    },
}

// ============================================================================
// Trait
// ============================================================================

/// Trait providing the ForceAtlas2 force-directed layout.
///
/// The matrix is interpreted as the weighted adjacency matrix of an
/// undirected graph and must therefore be square and symmetric, with finite,
/// non-negative weights. Self-loops are ignored entirely (they contribute
/// neither mass nor attraction). Explicitly stored zero-weight edges are
/// structural: they count toward the degree-based mass but exert no
/// attraction at the default edge weight influence. This differs from the
/// reference implementations, which derive edges from matrix nonzeros and
/// cannot represent them.
///
/// # Examples
///
/// ```
/// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::ForceAtlas2Config};
///
/// // A weighted triangle.
/// let edges = vec![(0, 1, 1.0), (0, 2, 2.0), (1, 0, 1.0), (1, 2, 1.0), (2, 0, 2.0), (2, 1, 1.0)];
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(6)
///         .expected_shape((3, 3))
///         .edges(edges.into_iter())
///         .build()
///         .unwrap();
///
/// let result = csr.force_atlas2(&ForceAtlas2Config::default()).unwrap();
///
/// assert_eq!(result.num_points(), 3);
/// assert_eq!(result.dimensions(), 2);
/// assert!(result.coordinates_flat().iter().all(|value| value.is_finite()));
/// ```
pub trait ForceAtlas2: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: ToPrimitive + Finite,
{
    /// Computes a ForceAtlas2 layout of the graph.
    ///
    /// # Arguments
    ///
    /// * `config`: the layout configuration.
    ///
    /// # Returns
    ///
    /// A [`ForceAtlas2Result`] with the final node coordinates.
    ///
    /// # Errors
    ///
    /// Returns a [`ForceAtlas2Error`] when the configuration is invalid or
    /// the matrix is not a finite, non-negative, symmetric square matrix.
    fn force_atlas2(
        &self,
        config: &ForceAtlas2Config,
    ) -> Result<ForceAtlas2Result, ForceAtlas2Error> {
        self.force_atlas2_with_progress(config, &mut |_, _| {})
    }

    /// Like [`force_atlas2`](Self::force_atlas2), but calls
    /// `on_progress(done, config.iterations)` after each iteration, with
    /// `done` running from `1` to `config.iterations`. The callback is
    /// unthrottled, so coalesce updates caller-side if needed.
    ///
    /// # Errors
    ///
    /// As [`force_atlas2`](Self::force_atlas2).
    fn force_atlas2_with_progress(
        &self,
        config: &ForceAtlas2Config,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<ForceAtlas2Result, ForceAtlas2Error> {
        validate_config(config)?;
        let mut state = LayoutState::from_matrix(self, config)?;
        let n = state.masses.len();

        let (final_swinging, final_traction) = state.run_with_progress(config, on_progress);

        let mut coordinates = Vec::with_capacity(n * 2);
        for position in &state.positions {
            coordinates.extend_from_slice(position);
        }

        Ok(ForceAtlas2Result {
            coordinates,
            n,
            dims: 2,
            final_swinging,
            final_traction,
            iterations_run: config.iterations,
        })
    }
}

impl<M: SparseValuedMatrix2D> ForceAtlas2 for M
where
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: ToPrimitive + Finite,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{ForceAtlas2Config, ForceAtlas2Error, state::LayoutState};
    use crate::{impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder};

    type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

    fn build_matrix(node_count: usize, undirected_edges: &[(usize, usize, f64)]) -> WeightedMatrix {
        let mut directed_edges = Vec::with_capacity(undirected_edges.len() * 2);
        for &(source, destination, weight) in undirected_edges {
            directed_edges.push((source, destination, weight));
            if source != destination {
                directed_edges.push((destination, source, weight));
            }
        }
        directed_edges.sort_unstable_by(|(s1, d1, _), (s2, d2, _)| (s1, d1).cmp(&(s2, d2)));

        GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(directed_edges.len())
            .expected_shape((node_count, node_count))
            .edges(directed_edges.into_iter())
            .build()
            .unwrap()
    }

    #[test]
    fn masses_are_degree_plus_one() {
        // Path 0 - 1 - 2 plus the isolated node 3.
        let matrix = build_matrix(4, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let config = ForceAtlas2Config::default();
        let state = LayoutState::from_matrix(&matrix, &config).unwrap();
        assert_eq!(state.masses, vec![2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn self_loops_are_ignored() {
        let matrix = build_matrix(2, &[(0, 0, 5.0), (0, 1, 1.0)]);
        let config = ForceAtlas2Config::default();
        let state = LayoutState::from_matrix(&matrix, &config).unwrap();
        // The self-loop contributes neither mass nor an edge.
        assert_eq!(state.masses, vec![2.0, 2.0]);
        assert_eq!(state.edges, vec![(0, 1, 1.0)]);
    }

    #[test]
    fn edges_are_upper_triangular() {
        let matrix = build_matrix(3, &[(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
        let config = ForceAtlas2Config::default();
        let state = LayoutState::from_matrix(&matrix, &config).unwrap();
        assert_eq!(state.edges, vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
    }

    #[test]
    fn zero_weight_edges_are_kept() {
        let matrix = build_matrix(2, &[(0, 1, 0.0)]);
        let config = ForceAtlas2Config::default();
        let state = LayoutState::from_matrix(&matrix, &config).unwrap();
        // Zero-weight edges are structural: they count toward the degree
        // and stay in the edge list (their attraction is scaled by w^delta).
        assert_eq!(state.masses, vec![2.0, 2.0]);
        assert_eq!(state.edges, vec![(0, 1, 0.0)]);
    }

    #[test]
    fn asymmetric_matrix_is_rejected() {
        let directed_edges = vec![(0, 1, 1.0)];
        let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(directed_edges.len())
            .expected_shape((2, 2))
            .edges(directed_edges.into_iter())
            .build()
            .unwrap();
        let config = ForceAtlas2Config::default();
        let error = LayoutState::from_matrix(&matrix, &config).unwrap_err();
        assert_eq!(error, ForceAtlas2Error::NonSymmetricEdge { source_id: 0, destination_id: 1 });
    }
}
