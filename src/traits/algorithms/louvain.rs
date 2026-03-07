//! Submodule providing the `Louvain` trait and its blanket implementation for
//! weighted monopartite graphs.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::modularity::{
    LocalMovingConfig, ModularityError, WeightedUndirectedGraph, local_moving, marker_partition,
    modularity, project_partition, regroup_members, renumber_partition, validate_common_config,
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

#[derive(Debug, Clone, PartialEq)]
/// Configuration options for the Louvain community detection algorithm.
pub struct LouvainConfig {
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

impl Default for LouvainConfig {
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
/// Partition information captured at one Louvain level.
pub struct LouvainLevel<Marker> {
    /// Community identifier for each original node.
    partition: Vec<Marker>,
    /// Modularity value at this level.
    modularity: f64,
    /// Number of node moves performed at this level.
    moved_nodes: usize,
}

impl<Marker> LouvainLevel<Marker> {
    /// Returns the partition of the original nodes at this level.
    #[must_use]
    #[inline]
    pub fn partition(&self) -> &[Marker] {
        &self.partition
    }

    /// Returns the modularity value at this level.
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
/// Result of the Louvain community detection algorithm.
pub struct LouvainResult<Marker> {
    levels: Vec<LouvainLevel<Marker>>,
}

impl<Marker> LouvainResult<Marker> {
    /// Returns the final partition of original nodes.
    #[must_use]
    #[inline]
    pub fn final_partition(&self) -> &[Marker] {
        self.levels.last().map_or(&[], LouvainLevel::partition)
    }

    /// Returns the final modularity value.
    #[must_use]
    #[inline]
    pub fn final_modularity(&self) -> f64 {
        self.levels.last().map_or(0.0, |l| l.modularity)
    }

    /// Returns all hierarchy levels computed by Louvain.
    #[must_use]
    #[inline]
    pub fn levels(&self) -> &[LouvainLevel<Marker>] {
        &self.levels
    }
}

/// Trait providing the Louvain community detection algorithm.
///
/// The graph is expected to be represented by a weighted, square matrix with
/// symmetric entries (undirected weighted graph).
pub trait Louvain<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Executes the Louvain algorithm with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the configuration is invalid;
    /// - the matrix is not square or not symmetric;
    /// - at least one weight is non-finite or non-positive;
    /// - the resulting number of communities cannot fit into `Marker`.
    ///
    /// # Complexity
    ///
    /// O(L * P * (V + E)) time and O(V + E) space, where L is the number of
    /// coarsening levels, P the number of local-moving passes per level, V
    /// the number of nodes and E the number of edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LouvainConfig};
    ///
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///         .expected_number_of_edges(8)
    ///         .expected_shape((4, 4))
    ///         .edges(
    ///             vec![
    ///                 (0, 0, 0.1),
    ///                 (0, 1, 1.0),
    ///                 (1, 0, 1.0),
    ///                 (1, 2, 0.1),
    ///                 (2, 1, 0.1),
    ///                 (2, 3, 1.0),
    ///                 (3, 2, 1.0),
    ///                 (3, 3, 0.1),
    ///             ]
    ///             .into_iter(),
    ///         )
    ///         .build()
    ///         .unwrap();
    ///
    /// let result = Louvain::<usize>::louvain(&edges, &LouvainConfig::default()).unwrap();
    /// assert_eq!(result.final_partition().len(), 4);
    /// assert!(!result.levels().is_empty());
    /// ```
    #[inline]
    fn louvain(&self, config: &LouvainConfig) -> Result<LouvainResult<Marker>, ModularityError> {
        validate_common_config(
            config.resolution,
            config.modularity_threshold,
            config.max_levels,
            config.max_local_passes,
        )?;

        let mut graph = WeightedUndirectedGraph::from_matrix(self)?;

        let original_number_of_nodes = self.number_of_rows().as_();
        let mut current_members: Vec<Vec<usize>> =
            (0..original_number_of_nodes).map(|node_id| vec![node_id]).collect();

        let mut levels: Vec<LouvainLevel<Marker>> = Vec::new();
        let mut previous_modularity: Option<f64> = None;

        for level_index in 0..config.max_levels {
            let (mut partition, moved_nodes) = local_moving(
                &graph,
                LocalMovingConfig {
                    resolution: config.resolution,
                    max_local_passes: config.max_local_passes,
                    seed: config.seed,
                },
                level_index,
            );
            let number_of_communities = renumber_partition(&mut partition);
            let modularity = modularity(&graph, &partition, config.resolution);

            let original_partition =
                project_partition(&current_members, &partition, original_number_of_nodes);
            let marker_partition = marker_partition::<Marker>(&original_partition)?;

            levels.push(LouvainLevel { partition: marker_partition, modularity, moved_nodes });

            if let Some(previous) = previous_modularity {
                if modularity - previous < config.modularity_threshold {
                    break;
                }
            }
            previous_modularity = Some(modularity);

            if number_of_communities == graph.number_of_nodes() {
                break;
            }

            graph = graph.induced(&partition, number_of_communities);
            current_members = regroup_members(current_members, &partition, number_of_communities);
        }

        Ok(LouvainResult { levels })
    }
}

impl<G, Marker> Louvain<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}

#[cfg(test)]
mod tests {
    use super::LouvainLevel;

    #[test]
    fn test_louvain_level_moved_nodes_getter() {
        let level =
            LouvainLevel { partition: vec![0usize, 0usize], modularity: 0.5, moved_nodes: 7 };
        assert_eq!(level.moved_nodes(), 7);
    }
}
