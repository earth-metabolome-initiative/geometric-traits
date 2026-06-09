//! Submodule providing the `DirectedLouvain` trait and its blanket
//! implementation for weighted directed graphs.
//!
//! Directed Louvain runs the same greedy multi-level skeleton as the undirected
//! [`Louvain`](super::Louvain) but optimizes the directed (Leicht-Newman /
//! Dugue-Perez) modularity scored by
//! [`DirectedModularity`](super::DirectedModularity). It therefore accepts an
//! asymmetric matrix: arc direction informs the detected communities.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::{
    directed_community::DirectedWorkingGraph,
    modularity::{
        LocalMovingConfig, ModularityError, marker_partition, project_partition, regroup_members,
        renumber_partition, validate_common_config,
    },
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

#[derive(Debug, Clone, PartialEq)]
/// Configuration options for the directed Louvain community detection
/// algorithm.
pub struct DirectedLouvainConfig {
    /// Resolution parameter (`gamma`) used in modularity optimization.
    ///
    /// Larger values tend to produce more communities.
    pub resolution: f64,
    /// Minimal modularity improvement required to continue to the next level.
    pub modularity_threshold: f64,
    /// Maximum number of coarsening levels.
    pub max_levels: usize,
    /// Maximum local-moving passes per level.
    pub max_local_passes: usize,
    /// Random seed used for node-order shuffling.
    pub seed: u64,
}

impl Default for DirectedLouvainConfig {
    #[inline]
    fn default() -> Self {
        Self {
            resolution: 1.0,
            modularity_threshold: 1.0e-7,
            max_levels: 100,
            max_local_passes: 100,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Partition information captured at one directed Louvain level.
pub struct DirectedLouvainLevel<Marker> {
    /// Community identifier for each original node.
    partition: Vec<Marker>,
    /// Directed modularity value at this level.
    modularity: f64,
    /// Number of node moves performed at this level.
    moved_nodes: usize,
}

impl<Marker> DirectedLouvainLevel<Marker> {
    /// Returns the partition of the original nodes at this level.
    #[must_use]
    #[inline]
    pub fn partition(&self) -> &[Marker] {
        &self.partition
    }

    /// Returns the directed modularity value at this level.
    #[must_use]
    #[inline]
    pub fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Returns the number of node moves performed at this level.
    #[must_use]
    #[inline]
    pub fn moved_nodes(&self) -> usize {
        self.moved_nodes
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of the directed Louvain community detection algorithm.
pub struct DirectedLouvainResult<Marker> {
    levels: Vec<DirectedLouvainLevel<Marker>>,
}

impl<Marker> DirectedLouvainResult<Marker> {
    /// Returns the final partition of original nodes.
    #[must_use]
    #[inline]
    pub fn final_partition(&self) -> &[Marker] {
        self.levels.last().map_or(&[], DirectedLouvainLevel::partition)
    }

    /// Returns the final directed modularity value.
    #[must_use]
    #[inline]
    pub fn final_modularity(&self) -> f64 {
        self.levels.last().map_or(0.0, DirectedLouvainLevel::modularity)
    }

    /// Returns all hierarchy levels computed by directed Louvain.
    #[must_use]
    #[inline]
    pub fn levels(&self) -> &[DirectedLouvainLevel<Marker>] {
        &self.levels
    }
}

/// Trait providing the directed Louvain community detection algorithm.
///
/// The graph is a weighted, square (directed) matrix: entry `(i, j)` is the
/// weight of the arc from `i` to `j`. Weights must be finite and strictly
/// positive. Unlike [`Louvain`](super::Louvain), the matrix need not be
/// symmetric and self-loops are allowed.
pub trait DirectedLouvain<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Executes directed Louvain with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the configuration is invalid;
    /// - the matrix is not square;
    /// - at least one weight is non-finite or non-positive;
    /// - the resulting number of communities cannot fit into `Marker`.
    ///
    /// # Complexity
    ///
    /// O(L * P * (V + E)) time and O(V + E) space, where L is the number of
    /// coarsening levels, P the number of local-moving passes per level, V the
    /// number of nodes and E the number of arcs.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::DirectedLouvainConfig};
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
    ///     DirectedLouvain::<usize>::directed_louvain(&edges, &DirectedLouvainConfig::default())
    ///         .unwrap();
    /// let partition = result.final_partition();
    /// assert_eq!(partition[0], partition[1]);
    /// assert_eq!(partition[2], partition[3]);
    /// assert_ne!(partition[0], partition[2]);
    /// ```
    #[inline]
    fn directed_louvain(
        &self,
        config: &DirectedLouvainConfig,
    ) -> Result<DirectedLouvainResult<Marker>, ModularityError> {
        validate_common_config(
            config.resolution,
            config.modularity_threshold,
            config.max_levels,
            config.max_local_passes,
        )?;

        let mut graph = DirectedWorkingGraph::from_matrix(self)?;

        let original_number_of_nodes = self.number_of_rows().as_();
        let mut current_members: Vec<Vec<usize>> =
            (0..original_number_of_nodes).map(|node_id| vec![node_id]).collect();

        let mut levels: Vec<DirectedLouvainLevel<Marker>> = Vec::new();
        let mut previous_modularity: Option<f64> = None;

        for level_index in 0..config.max_levels {
            let (mut partition, moved_nodes) = graph.local_moving(
                LocalMovingConfig {
                    resolution: config.resolution,
                    max_local_passes: config.max_local_passes,
                    seed: config.seed,
                },
                level_index,
            );
            let number_of_communities = renumber_partition(&mut partition);
            let modularity = graph.modularity(&partition, config.resolution);

            let original_partition =
                project_partition(&current_members, &partition, original_number_of_nodes);
            let marker_partition = marker_partition::<Marker>(&original_partition)?;

            levels.push(DirectedLouvainLevel {
                partition: marker_partition,
                modularity,
                moved_nodes,
            });

            if let Some(previous) = previous_modularity {
                if modularity - previous < config.modularity_threshold {
                    break;
                }
            }
            previous_modularity = Some(modularity);

            if number_of_communities == graph.number_of_nodes() {
                break;
            }

            let induced = graph.induce(&partition, number_of_communities);
            current_members = regroup_members(current_members, &partition, number_of_communities);
            graph = induced;
        }

        Ok(DirectedLouvainResult { levels })
    }
}

impl<G, Marker> DirectedLouvain<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}
