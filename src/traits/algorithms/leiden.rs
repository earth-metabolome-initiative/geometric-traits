//! Submodule providing the `Leiden` trait and its blanket implementation for
//! weighted monopartite graphs.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};

use super::modularity::{
    LocalMovingConfig, ModularityError, WeightedUndirectedGraph, approx_eq, local_moving,
    marker_partition, mix_seed, modularity, project_partition, regroup_members, renumber_partition,
    split_disconnected_communities, validate_common_config, validate_leiden_config,
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

#[derive(Debug, Clone, PartialEq)]
/// Configuration options for the Leiden community detection algorithm.
pub struct LeidenConfig {
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
    /// Maximum refinement passes per level.
    pub max_refinement_passes: usize,
    /// Randomness parameter for refinement community selection.
    ///
    /// Lower values make refinement more greedy.
    pub theta: f64,
    /// Random seed used for node-order shuffling.
    pub seed: u64,
}

impl Default for LeidenConfig {
    #[inline]
    fn default() -> Self {
        Self {
            resolution: 1.0,
            modularity_threshold: 1.0e-7,
            max_levels: 100,
            max_local_passes: 100,
            max_refinement_passes: 100,
            theta: 0.01,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Partition information captured at one Leiden level.
pub struct LeidenLevel<Marker> {
    /// Community identifier for each original node.
    partition: Vec<Marker>,
    /// Modularity value at this level.
    modularity: f64,
    /// Number of node moves performed at this level.
    moved_nodes: usize,
    /// Number of refinement moves performed at this level.
    refinement_moves: usize,
}

impl<Marker> LeidenLevel<Marker> {
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

    /// Returns the number of refinement moves performed at this level.
    #[must_use]
    #[inline]
    pub fn refinement_moves(&self) -> usize {
        self.refinement_moves
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of the Leiden community detection algorithm.
pub struct LeidenResult<Marker> {
    levels: Vec<LeidenLevel<Marker>>,
}

impl<Marker> LeidenResult<Marker> {
    /// Returns the final partition of original nodes.
    #[must_use]
    #[inline]
    pub fn final_partition(&self) -> &[Marker] {
        self.levels.last().map_or(&[], LeidenLevel::partition)
    }

    /// Returns the final modularity value.
    #[must_use]
    #[inline]
    pub fn final_modularity(&self) -> f64 {
        self.levels.last().map_or(0.0, |l| l.modularity)
    }

    /// Returns all hierarchy levels computed by Leiden.
    #[must_use]
    #[inline]
    pub fn levels(&self) -> &[LeidenLevel<Marker>] {
        &self.levels
    }
}

/// Trait providing the Leiden community detection algorithm.
///
/// The graph is expected to be represented by a weighted, square matrix with
/// symmetric entries (undirected weighted graph).
pub trait Leiden<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Executes the Leiden algorithm with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the configuration is invalid;
    /// - the matrix is not square or not symmetric;
    /// - at least one weight is non-finite or non-positive;
    /// - the resulting number of communities cannot fit into `Marker`.
    #[inline]
    fn leiden(&self, config: &LeidenConfig) -> Result<LeidenResult<Marker>, ModularityError> {
        validate_common_config(
            config.resolution,
            config.modularity_threshold,
            config.max_levels,
            config.max_local_passes,
        )?;
        validate_leiden_config(config.max_refinement_passes, config.theta)?;

        let mut graph = WeightedUndirectedGraph::from_matrix(self)?;

        let original_number_of_nodes = self.number_of_rows().as_();
        let mut current_members: Vec<Vec<usize>> =
            (0..original_number_of_nodes).map(|node_id| vec![node_id]).collect();

        let mut levels: Vec<LeidenLevel<Marker>> = Vec::new();
        let mut previous_modularity: Option<f64> = None;

        for level_index in 0..config.max_levels {
            let (mut local_partition, moved_nodes) = local_moving(
                &graph,
                LocalMovingConfig {
                    resolution: config.resolution,
                    max_local_passes: config.max_local_passes,
                    seed: config.seed,
                },
                level_index,
            );
            renumber_partition(&mut local_partition);

            let (mut refined_partition, refinement_moves) =
                refine_partition(&graph, &local_partition, config, level_index);
            split_disconnected_communities(&graph, &mut refined_partition);
            let number_of_communities = renumber_partition(&mut refined_partition);

            let level_modularity = modularity(&graph, &refined_partition, config.resolution);
            let original_partition =
                project_partition(&current_members, &refined_partition, original_number_of_nodes);
            let marker_level_partition = marker_partition::<Marker>(&original_partition)?;

            levels.push(LeidenLevel {
                partition: marker_level_partition,
                modularity: level_modularity,
                moved_nodes: moved_nodes + refinement_moves,
                refinement_moves,
            });

            if let Some(previous) = previous_modularity {
                if level_modularity - previous < config.modularity_threshold {
                    break;
                }
            }
            previous_modularity = Some(level_modularity);

            if number_of_communities == graph.number_of_nodes() {
                break;
            }

            graph = graph.induced(&refined_partition, number_of_communities);
            current_members =
                regroup_members(current_members, &refined_partition, number_of_communities);
        }

        Ok(LeidenResult { levels })
    }
}

impl<G, Marker> Leiden<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}

fn refine_partition(
    graph: &WeightedUndirectedGraph,
    parent_partition: &[usize],
    config: &LeidenConfig,
    level_index: usize,
) -> (Vec<usize>, usize) {
    let number_of_nodes = graph.number_of_nodes();
    let mut refined_partition: Vec<usize> = (0..number_of_nodes).collect();

    if number_of_nodes == 0 || graph.total_weight <= 0.0 || !graph.total_weight.is_normal() {
        return (refined_partition, 0);
    }

    let number_of_parent_communities =
        parent_partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    if number_of_parent_communities == 0 {
        return (refined_partition, 0);
    }

    let mut nodes_per_parent: Vec<Vec<usize>> = vec![Vec::new(); number_of_parent_communities];
    for (node, parent_community) in parent_partition.iter().copied().enumerate() {
        nodes_per_parent[parent_community].push(node);
    }

    let mut community_totals = graph.degree.clone();
    let mut weights_to_communities = vec![0.0; number_of_nodes];
    let mut touched_communities: Vec<usize> = Vec::new();
    let mut candidate_moves: Vec<(usize, f64)> = Vec::new();
    let mut moved_nodes = 0usize;

    for pass_index in 0..config.max_refinement_passes {
        let mut moved_in_pass = 0usize;

        for (parent_community, nodes) in nodes_per_parent.iter().enumerate() {
            if nodes.len() <= 1 {
                continue;
            }

            let mut order = nodes.clone();
            let mut rng = SmallRng::seed_from_u64(mix_seed(
                config.seed,
                level_index.wrapping_add(parent_community),
                pass_index,
            ));
            order.shuffle(&mut rng);

            for node in &order {
                let node = *node;
                let node_degree = graph.degree[node];
                if node_degree <= 0.0 {
                    continue;
                }

                let source_community = refined_partition[node];
                touched_communities.clear();

                for (neighbor, weight) in &graph.adjacency[node] {
                    if parent_partition[*neighbor] != parent_community {
                        continue;
                    }
                    let neighbor_community = refined_partition[*neighbor];
                    if weights_to_communities[neighbor_community] == 0.0 {
                        touched_communities.push(neighbor_community);
                    }
                    weights_to_communities[neighbor_community] += *weight;
                }

                community_totals[source_community] -= node_degree;

                let current_gain = weights_to_communities[source_community]
                    - config.resolution * node_degree * community_totals[source_community]
                        / graph.total_weight;

                candidate_moves.clear();
                for community in &touched_communities {
                    let community = *community;
                    if community == source_community {
                        continue;
                    }
                    let gain = weights_to_communities[community]
                        - config.resolution * node_degree * community_totals[community]
                            / graph.total_weight;
                    let delta = gain - current_gain;
                    if delta > f64::EPSILON {
                        candidate_moves.push((community, delta));
                    }
                }

                candidate_moves.sort_unstable_by_key(|(community, _)| *community);

                let destination = if candidate_moves.is_empty() {
                    source_community
                } else {
                    sample_softmax_destination(&candidate_moves, config.theta, &mut rng)
                };

                refined_partition[node] = destination;
                community_totals[destination] += node_degree;

                if destination != source_community {
                    moved_in_pass += 1;
                    moved_nodes += 1;
                }

                for community in &touched_communities {
                    weights_to_communities[*community] = 0.0;
                }
                weights_to_communities[source_community] = 0.0;
            }
        }

        if moved_in_pass == 0 {
            break;
        }
    }

    (refined_partition, moved_nodes)
}

fn sample_softmax_destination(
    candidate_moves: &[(usize, f64)],
    theta: f64,
    rng: &mut SmallRng,
) -> usize {
    let max_score =
        candidate_moves.iter().map(|(_, score)| *score).fold(f64::NEG_INFINITY, f64::max);

    let mut cumulative_weights: Vec<f64> = Vec::with_capacity(candidate_moves.len());
    let mut total_weight = 0.0;
    for (_, score) in candidate_moves {
        let scaled = ((*score - max_score) / theta).exp();
        if !scaled.is_finite() || scaled <= 0.0 {
            cumulative_weights.push(total_weight);
            continue;
        }
        total_weight += scaled;
        cumulative_weights.push(total_weight);
    }

    if !total_weight.is_finite() || total_weight <= 0.0 {
        return best_candidate(candidate_moves);
    }

    let draw = rng.gen_range(0.0..total_weight);
    for idx in 0..candidate_moves.len().saturating_sub(1) {
        if draw <= cumulative_weights[idx] {
            return candidate_moves[idx].0;
        }
    }

    candidate_moves.last().map_or(0, |(community, _)| *community)
}

fn best_candidate(candidate_moves: &[(usize, f64)]) -> usize {
    let mut best = candidate_moves[0];
    for candidate in candidate_moves.iter().copied().skip(1) {
        if candidate.1 > best.1 + f64::EPSILON
            || (approx_eq(candidate.1, best.1) && candidate.0 < best.0)
        {
            best = candidate;
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::{
        LeidenConfig, LeidenLevel, WeightedUndirectedGraph, best_candidate, refine_partition,
        sample_softmax_destination,
    };

    #[test]
    fn test_leiden_level_move_getters() {
        let level = LeidenLevel {
            partition: vec![0usize, 1usize],
            modularity: 0.25,
            moved_nodes: 11,
            refinement_moves: 3,
        };
        assert_eq!(level.moved_nodes(), 11);
        assert_eq!(level.refinement_moves(), 3);
    }

    #[test]
    fn test_refine_partition_returns_when_parent_partition_is_empty() {
        let graph = WeightedUndirectedGraph {
            adjacency: vec![vec![(0, 1.0)]],
            degree: vec![1.0],
            total_weight: 1.0,
        };
        let config = LeidenConfig::default();

        let (partition, moved_nodes) = refine_partition(&graph, &[], &config, 0);

        assert_eq!(partition, vec![0]);
        assert_eq!(moved_nodes, 0);
    }

    #[test]
    fn test_refine_partition_skips_zero_degree_nodes() {
        let graph = WeightedUndirectedGraph {
            adjacency: vec![vec![], vec![(1, 1.0)]],
            degree: vec![0.0, 1.0],
            total_weight: 1.0,
        };
        let config = LeidenConfig {
            max_refinement_passes: 1,
            max_local_passes: 1,
            max_levels: 1,
            theta: 0.5,
            ..LeidenConfig::default()
        };

        let (partition, moved_nodes) = refine_partition(&graph, &[0, 0], &config, 0);

        assert_eq!(partition.len(), 2);
        assert_eq!(moved_nodes, 0);
    }

    #[test]
    fn test_sample_softmax_destination_falls_back_to_best_candidate_on_non_finite_scores() {
        let candidates = vec![(5usize, f64::NAN), (3usize, f64::NAN)];
        let mut rng = SmallRng::seed_from_u64(42);

        let chosen = sample_softmax_destination(&candidates, 0.1, &mut rng);

        assert_eq!(chosen, 5);
    }

    #[test]
    fn test_best_candidate_prefers_larger_gain_and_smaller_id_on_tie() {
        let candidates = vec![(5usize, 1.0), (4usize, 2.0), (3usize, 2.0)];
        assert_eq!(best_candidate(&candidates), 3);
    }
}
