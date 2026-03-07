//! Shared internals for modularity-based community detection algorithms.

use alloc::{
    collections::{BTreeMap, VecDeque},
    vec::Vec,
};

use num_traits::{AsPrimitive, ToPrimitive};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::traits::{Finite, MonopartiteGraph, PositiveInteger, SparseValuedMatrix2D};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error enumeration for modularity-based community detection algorithms.
pub enum ModularityError {
    /// The resolution parameter must be finite and strictly positive.
    #[error("The modularity resolution must be finite and strictly positive.")]
    InvalidResolution,
    /// The modularity threshold must be finite and non-negative.
    #[error("The modularity threshold must be finite and non-negative.")]
    InvalidModularityThreshold,
    /// The maximum number of levels must be strictly positive.
    #[error("The maximum number of levels must be strictly positive.")]
    InvalidMaxLevels,
    /// The maximum number of local passes must be strictly positive.
    #[error("The maximum number of local passes must be strictly positive.")]
    InvalidMaxLocalPasses,
    /// The maximum number of refinement passes must be strictly positive.
    #[error("The maximum number of refinement passes must be strictly positive.")]
    InvalidMaxRefinementPasses,
    /// The Leiden theta parameter must be finite and strictly positive.
    #[error("The Leiden theta parameter must be finite and strictly positive.")]
    InvalidTheta,
    /// The matrix is not square.
    #[error("The modularity matrix must be square, but received shape ({rows}, {columns}).")]
    NonSquareMatrix {
        /// Number of rows in the matrix.
        rows: usize,
        /// Number of columns in the matrix.
        columns: usize,
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
    /// The edge weight must be strictly positive.
    #[error("Found a non-positive edge weight on ({source_id}, {destination_id}).")]
    NonPositiveWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
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
    /// The selected community marker type is too small.
    #[error("The selected community marker type is too small for this partition.")]
    TooManyCommunities,
}

impl From<ModularityError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: ModularityError) -> Self {
        Self::ModularityError(error)
    }
}

impl<G: MonopartiteGraph> From<ModularityError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: ModularityError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalMovingConfig {
    pub(crate) resolution: f64,
    pub(crate) max_local_passes: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct WeightedUndirectedGraph {
    pub(crate) adjacency: Vec<Vec<(usize, f64)>>,
    pub(crate) degree: Vec<f64>,
    pub(crate) total_weight: f64,
}

impl WeightedUndirectedGraph {
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub(crate) fn from_matrix<M>(matrix: &M) -> Result<Self, ModularityError>
    where
        M: SparseValuedMatrix2D,
        M::RowIndex: AsPrimitive<usize>,
        M::ColumnIndex: AsPrimitive<usize>,
        M::Value: ToPrimitive + Finite,
    {
        let rows = matrix.number_of_rows().as_();
        let columns = matrix.number_of_columns().as_();
        if rows != columns {
            return Err(ModularityError::NonSquareMatrix { rows, columns });
        }

        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows];
        let mut degree = vec![0.0; rows];

        for row_id in matrix.row_indices() {
            let source = row_id.as_();
            for (column_id, weight) in
                matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id))
            {
                let destination = column_id.as_();
                if !weight.is_finite() {
                    return Err(ModularityError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                let weight = weight.to_f64().ok_or(ModularityError::UnrepresentableWeight {
                    source_id: source,
                    destination_id: destination,
                })?;
                if !weight.is_finite() {
                    return Err(ModularityError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                if weight <= 0.0 {
                    return Err(ModularityError::NonPositiveWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                adjacency[source].push((destination, weight));
                degree[source] += weight;
            }
        }

        for row in &mut adjacency {
            row.sort_unstable_by_key(|(destination, _)| *destination);
        }

        for (source, neighbors) in adjacency.iter().enumerate() {
            for (destination, weight) in neighbors {
                if !has_matching_edge(&adjacency[*destination], source, *weight) {
                    return Err(ModularityError::NonSymmetricEdge {
                        source_id: source,
                        destination_id: *destination,
                    });
                }
            }
        }

        let total_weight = degree.iter().sum();
        Ok(Self { adjacency, degree, total_weight })
    }

    pub(crate) fn induced(&self, partition: &[usize], number_of_communities: usize) -> Self {
        let mut compact_edges: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (source, neighbors) in self.adjacency.iter().enumerate() {
            let source_community = partition[source];
            for (destination, weight) in neighbors {
                let destination_community = partition[*destination];
                *compact_edges.entry((source_community, destination_community)).or_insert(0.0) +=
                    *weight;
            }
        }

        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); number_of_communities];
        let mut degree: Vec<f64> = vec![0.0; number_of_communities];

        for ((source, destination), weight) in compact_edges {
            adjacency[source].push((destination, weight));
            degree[source] += weight;
        }

        let total_weight = degree.iter().sum();
        Self { adjacency, degree, total_weight }
    }
}

pub(crate) fn validate_common_config(
    resolution: f64,
    modularity_threshold: f64,
    max_levels: usize,
    max_local_passes: usize,
) -> Result<(), ModularityError> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(ModularityError::InvalidResolution);
    }
    if !modularity_threshold.is_finite() || modularity_threshold < 0.0 {
        return Err(ModularityError::InvalidModularityThreshold);
    }
    if max_levels == 0 {
        return Err(ModularityError::InvalidMaxLevels);
    }
    if max_local_passes == 0 {
        return Err(ModularityError::InvalidMaxLocalPasses);
    }
    Ok(())
}

pub(crate) fn validate_leiden_config(
    max_refinement_passes: usize,
    theta: f64,
) -> Result<(), ModularityError> {
    if max_refinement_passes == 0 {
        return Err(ModularityError::InvalidMaxRefinementPasses);
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(ModularityError::InvalidTheta);
    }
    Ok(())
}

pub(crate) fn local_moving(
    graph: &WeightedUndirectedGraph,
    config: LocalMovingConfig,
    level_index: usize,
) -> (Vec<usize>, usize) {
    let number_of_nodes = graph.number_of_nodes();
    let mut partition: Vec<usize> = (0..number_of_nodes).collect();

    if number_of_nodes == 0 || graph.total_weight <= 0.0 || !graph.total_weight.is_normal() {
        return (partition, 0);
    }

    let mut community_totals = graph.degree.clone();
    let mut order: Vec<usize> = (0..number_of_nodes).collect();
    let mut touched_communities = Vec::new();
    let mut weights_to_communities = vec![0.0; number_of_nodes];
    let mut moved_nodes = 0usize;

    for pass_index in 0..config.max_local_passes {
        let mut rng = SmallRng::seed_from_u64(mix_seed(config.seed, level_index, pass_index));
        order.shuffle(&mut rng);

        let mut moved_in_pass = 0usize;

        for node in &order {
            let node = *node;
            let node_degree = graph.degree[node];
            if node_degree <= 0.0 {
                continue;
            }

            let source_community = partition[node];
            touched_communities.clear();

            for (neighbor, weight) in &graph.adjacency[node] {
                let neighbor_community = partition[*neighbor];
                if weights_to_communities[neighbor_community] == 0.0 {
                    touched_communities.push(neighbor_community);
                }
                weights_to_communities[neighbor_community] += *weight;
            }

            community_totals[source_community] -= node_degree;

            let mut best_community = source_community;
            let mut best_gain = weights_to_communities[source_community]
                - config.resolution * node_degree * community_totals[source_community]
                    / graph.total_weight;

            for community in &touched_communities {
                let community = *community;
                let gain = weights_to_communities[community]
                    - config.resolution * node_degree * community_totals[community]
                        / graph.total_weight;
                if gain > best_gain + f64::EPSILON
                    || (approx_eq(gain, best_gain) && community < best_community)
                {
                    best_gain = gain;
                    best_community = community;
                }
            }

            partition[node] = best_community;
            community_totals[best_community] += node_degree;

            if best_community != source_community {
                moved_in_pass += 1;
                moved_nodes += 1;
            }

            for community in &touched_communities {
                weights_to_communities[*community] = 0.0;
            }
            weights_to_communities[source_community] = 0.0;
        }

        if moved_in_pass == 0 {
            break;
        }
    }

    (partition, moved_nodes)
}

pub(crate) fn split_disconnected_communities(
    graph: &WeightedUndirectedGraph,
    partition: &mut [usize],
) {
    let number_of_nodes = partition.len();
    if number_of_nodes <= 1 {
        return;
    }

    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    if number_of_communities <= 1 {
        return;
    }

    let mut nodes_per_community: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    for (node, community) in partition.iter().copied().enumerate() {
        nodes_per_community[community].push(node);
    }

    let mut community_mask = vec![false; number_of_nodes];
    let mut visited = vec![false; number_of_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut next_community_id = number_of_communities;

    for (community_id, nodes) in nodes_per_community.into_iter().enumerate() {
        if nodes.len() <= 1 {
            continue;
        }

        for &node in &nodes {
            community_mask[node] = true;
        }

        let mut first_component = true;
        for &start in &nodes {
            if visited[start] {
                continue;
            }

            visited[start] = true;
            queue.push_back(start);
            let target_community = if first_component {
                community_id
            } else {
                let id = next_community_id;
                next_community_id += 1;
                id
            };
            first_component = false;

            while let Some(node) = queue.pop_front() {
                partition[node] = target_community;
                for (neighbor, _) in &graph.adjacency[node] {
                    let neighbor = *neighbor;
                    if community_mask[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        for &node in &nodes {
            community_mask[node] = false;
            visited[node] = false;
        }
    }
}

pub(crate) fn modularity(
    graph: &WeightedUndirectedGraph,
    partition: &[usize],
    resolution: f64,
) -> f64 {
    if graph.total_weight <= 0.0 || !graph.total_weight.is_normal() {
        return 0.0;
    }

    let number_of_communities = partition.iter().copied().max().map_or(0, |max| max + 1);
    let mut total_weight_per_community = vec![0.0; number_of_communities];
    let mut internal_weight_per_community = vec![0.0; number_of_communities];

    for (source, source_community) in partition.iter().copied().enumerate() {
        total_weight_per_community[source_community] += graph.degree[source];
        for (destination, weight) in &graph.adjacency[source] {
            if partition[*destination] == source_community {
                internal_weight_per_community[source_community] += *weight;
            }
        }
    }

    let inverse_total_weight = 1.0 / graph.total_weight;
    total_weight_per_community.iter().zip(internal_weight_per_community.iter()).fold(
        0.0,
        |modularity, (total_weight, internal_weight)| {
            if *total_weight <= 0.0 {
                return modularity;
            }
            let total_fraction = *total_weight * inverse_total_weight;
            modularity + (*internal_weight * inverse_total_weight)
                - resolution * total_fraction * total_fraction
        },
    )
}

pub(crate) fn renumber_partition(partition: &mut [usize]) -> usize {
    let mut mapping = vec![usize::MAX; partition.len()];
    let mut next_community_id = 0usize;

    for community in partition {
        if mapping[*community] == usize::MAX {
            mapping[*community] = next_community_id;
            next_community_id += 1;
        }
        *community = mapping[*community];
    }

    next_community_id
}

pub(crate) fn project_partition(
    current_members: &[Vec<usize>],
    partition: &[usize],
    number_of_original_nodes: usize,
) -> Vec<usize> {
    let mut projected_partition = vec![0usize; number_of_original_nodes];
    for (current_node, member_nodes) in current_members.iter().enumerate() {
        let community = partition[current_node];
        for original_node in member_nodes {
            projected_partition[*original_node] = community;
        }
    }
    projected_partition
}

pub(crate) fn regroup_members(
    current_members: Vec<Vec<usize>>,
    partition: &[usize],
    number_of_communities: usize,
) -> Vec<Vec<usize>> {
    let mut next_members = vec![Vec::new(); number_of_communities];
    for (current_node, member_nodes) in current_members.into_iter().enumerate() {
        next_members[partition[current_node]].extend(member_nodes);
    }
    next_members
}

pub(crate) fn marker_partition<Marker: PositiveInteger>(
    partition: &[usize],
) -> Result<Vec<Marker>, ModularityError> {
    partition
        .iter()
        .copied()
        .map(|community| {
            Marker::try_from_usize(community).map_err(|_| ModularityError::TooManyCommunities)
        })
        .collect()
}

pub(crate) fn mix_seed(seed: u64, level_index: usize, pass_index: usize) -> u64 {
    let level = level_index as u64;
    let pass = pass_index as u64;
    seed ^ level.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ pass.wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

pub(crate) fn approx_eq(left: f64, right: f64) -> bool {
    let tolerance = (left.abs().max(right.abs()).max(1.0)) * 16.0 * f64::EPSILON;
    (left - right).abs() <= tolerance
}

fn has_matching_edge(row: &[(usize, f64)], destination: usize, weight: f64) -> bool {
    row.binary_search_by_key(&destination, |(col, _)| *col)
        .is_ok_and(|idx| approx_eq(weight, row[idx].1))
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::{ModularityError, WeightedUndirectedGraph, split_disconnected_communities};
    use crate::{
        impls::{CSR2D, GenericImplicitValuedMatrix2D},
        naive_structs::{GenericEdgesBuilder, GenericGraph},
        traits::{EdgesBuilder, Finite},
    };

    #[derive(Clone, Copy, Debug)]
    struct NonFiniteAfterConversion;

    impl ToPrimitive for NonFiniteAfterConversion {
        fn to_i64(&self) -> Option<i64> {
            None
        }

        fn to_u64(&self) -> Option<u64> {
            None
        }

        fn to_f64(&self) -> Option<f64> {
            Some(f64::INFINITY)
        }
    }

    impl Finite for NonFiniteAfterConversion {
        fn is_finite(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_modularity_error_converts_to_algorithm_error() {
        let converted: crate::errors::monopartite_graph_error::MonopartiteAlgorithmError =
            ModularityError::InvalidResolution.into();

        assert!(matches!(
            converted,
            crate::errors::monopartite_graph_error::MonopartiteAlgorithmError::ModularityError(
                ModularityError::InvalidResolution
            )
        ));
    }

    #[test]
    fn test_modularity_error_converts_to_monopartite_error() {
        type Graph = GenericGraph<usize, CSR2D<usize, usize, usize>>;
        let converted: crate::errors::MonopartiteError<Graph> =
            ModularityError::InvalidTheta.into();

        assert!(matches!(
            converted,
            crate::errors::MonopartiteError::AlgorithmError(
                crate::errors::monopartite_graph_error::MonopartiteAlgorithmError::ModularityError(
                    ModularityError::InvalidTheta
                )
            )
        ));
    }

    #[test]
    fn test_from_matrix_rejects_non_finite_after_to_f64_conversion() {
        let structure: CSR2D<usize, usize, usize> =
            GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
                .expected_number_of_edges(2)
                .expected_shape((2, 2))
                .edges(vec![(0, 1), (1, 0)].into_iter())
                .build()
                .unwrap();

        let matrix = GenericImplicitValuedMatrix2D::new(structure, |_| NonFiniteAfterConversion);
        let error = WeightedUndirectedGraph::from_matrix(&matrix).unwrap_err();

        assert!(matches!(error, ModularityError::NonFiniteWeight { .. }));
    }

    #[test]
    fn test_split_disconnected_communities_returns_for_single_community_partition() {
        let graph = WeightedUndirectedGraph {
            adjacency: vec![vec![(1, 1.0)], vec![(0, 1.0)]],
            degree: vec![1.0, 1.0],
            total_weight: 2.0,
        };
        let mut partition = vec![0usize, 0usize];

        split_disconnected_communities(&graph, &mut partition);

        assert_eq!(partition, vec![0, 0]);
    }

    #[test]
    fn test_split_disconnected_communities_splits_multi_component_community() {
        let graph = WeightedUndirectedGraph {
            adjacency: vec![
                vec![(1, 1.0)],
                vec![(0, 1.0)],
                vec![(3, 1.0)],
                vec![(2, 1.0)],
                vec![(4, 1.0)],
            ],
            degree: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            total_weight: 5.0,
        };
        let mut partition = vec![0usize, 0usize, 0usize, 0usize, 1usize];

        split_disconnected_communities(&graph, &mut partition);

        assert_eq!(partition[0], partition[1]);
        assert_eq!(partition[2], partition[3]);
        assert_ne!(partition[0], partition[2]);
    }
}
