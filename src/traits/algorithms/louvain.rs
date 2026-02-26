//! Submodule providing the `Louvain` trait and its blanket implementation for
//! weighted monopartite graphs.
use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt::{Display, Formatter};

use num_traits::ToPrimitive;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::traits::{
    Finite, IntoUsize, MonopartiteGraph, Number, PositiveInteger, SparseValuedMatrix2D,
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error enumeration for Louvain community detection.
pub enum LouvainError {
    /// The resolution parameter must be finite and strictly positive.
    InvalidResolution,
    /// The modularity threshold must be finite and non-negative.
    InvalidModularityThreshold,
    /// The maximum number of levels must be strictly positive.
    InvalidMaxLevels,
    /// The maximum number of local passes must be strictly positive.
    InvalidMaxLocalPasses,
    /// The matrix is not square.
    NonSquareMatrix {
        /// Number of rows in the matrix.
        rows: usize,
        /// Number of columns in the matrix.
        columns: usize,
    },
    /// The edge weight cannot be represented as `f64`.
    UnrepresentableWeight {
        /// Source node identifier.
        source: usize,
        /// Destination node identifier.
        destination: usize,
    },
    /// The edge weight is not finite.
    NonFiniteWeight {
        /// Source node identifier.
        source: usize,
        /// Destination node identifier.
        destination: usize,
    },
    /// The edge weight must be strictly positive.
    NonPositiveWeight {
        /// Source node identifier.
        source: usize,
        /// Destination node identifier.
        destination: usize,
    },
    /// The matrix does not represent an undirected graph.
    NonSymmetricEdge {
        /// Source node identifier.
        source: usize,
        /// Destination node identifier.
        destination: usize,
    },
    /// The selected community marker type is too small.
    TooManyCommunities,
}

impl Display for LouvainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidResolution => {
                write!(f, "The Louvain resolution must be finite and strictly positive.")
            }
            Self::InvalidModularityThreshold => {
                write!(f, "The Louvain modularity threshold must be finite and non-negative.")
            }
            Self::InvalidMaxLevels => {
                write!(f, "The Louvain maximum number of levels must be strictly positive.")
            }
            Self::InvalidMaxLocalPasses => {
                write!(f, "The Louvain maximum number of local passes must be strictly positive.")
            }
            Self::NonSquareMatrix { rows, columns } => {
                write!(
                    f,
                    "The Louvain matrix must be square, but received shape ({rows}, {columns})."
                )
            }
            Self::UnrepresentableWeight { source, destination } => {
                write!(
                    f,
                    "Found an edge weight on ({source}, {destination}) that cannot be represented as f64."
                )
            }
            Self::NonFiniteWeight { source, destination } => {
                write!(f, "Found a non-finite edge weight on ({source}, {destination}).")
            }
            Self::NonPositiveWeight { source, destination } => {
                write!(f, "Found a non-positive edge weight on ({source}, {destination}).")
            }
            Self::NonSymmetricEdge { source, destination } => {
                write!(
                    f,
                    "The matrix is not symmetric: edge ({source}, {destination}) has no matching reverse edge."
                )
            }
            Self::TooManyCommunities => {
                write!(f, "The selected community marker type is too small for this partition.")
            }
        }
    }
}

impl core::error::Error for LouvainError {}

impl From<LouvainError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    fn from(error: LouvainError) -> Self {
        Self::LouvainError(error)
    }
}

impl<G: MonopartiteGraph> From<LouvainError> for crate::errors::MonopartiteError<G> {
    fn from(error: LouvainError) -> Self {
        Self::AlgorithmError(error.into())
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
    pub fn partition(&self) -> &[Marker] {
        &self.partition
    }

    /// Returns the modularity value at this level.
    pub fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Returns the number of node moves performed at this level.
    pub fn moved_nodes(&self) -> usize {
        self.moved_nodes
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of the Louvain community detection algorithm.
pub struct LouvainResult<Marker> {
    final_partition: Vec<Marker>,
    final_modularity: f64,
    levels: Vec<LouvainLevel<Marker>>,
}

impl<Marker> LouvainResult<Marker> {
    /// Returns the final partition of original nodes.
    pub fn final_partition(&self) -> &[Marker] {
        &self.final_partition
    }

    /// Returns the final modularity value.
    pub fn final_modularity(&self) -> f64 {
        self.final_modularity
    }

    /// Returns all hierarchy levels computed by Louvain.
    pub fn levels(&self) -> &[LouvainLevel<Marker>] {
        &self.levels
    }
}

/// Trait providing the Louvain community detection algorithm.
///
/// The graph is expected to be represented by a weighted, square matrix with
/// symmetric entries (undirected weighted graph).
pub trait Louvain<Marker: IntoUsize + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: IntoUsize,
    Self::ColumnIndex: IntoUsize,
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
    /// let result = Louvain::<usize>::louvain(&edges, LouvainConfig::default()).unwrap();
    /// assert_eq!(result.final_partition().len(), 4);
    /// assert!(!result.levels().is_empty());
    /// ```
    fn louvain(&self, config: LouvainConfig) -> Result<LouvainResult<Marker>, LouvainError> {
        validate_config(&config)?;

        let mut graph = WeightedUndirectedGraph::from_matrix(self)?;

        let original_number_of_nodes = self.number_of_rows().into_usize();
        let mut current_members: Vec<Vec<usize>> =
            (0..original_number_of_nodes).map(|node_id| vec![node_id]).collect();

        let mut levels: Vec<LouvainLevel<Marker>> = Vec::new();
        let mut previous_modularity: Option<f64> = None;

        for level_index in 0..config.max_levels {
            let (mut partition, moved_nodes) = local_moving(&graph, &config, level_index);
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

        let final_level = levels.last().cloned().unwrap_or(LouvainLevel {
            partition: Vec::new(),
            modularity: 0.0,
            moved_nodes: 0,
        });

        Ok(LouvainResult {
            final_partition: final_level.partition,
            final_modularity: final_level.modularity,
            levels,
        })
    }
}

impl<G, Marker> Louvain<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: IntoUsize + PositiveInteger,
    G::RowIndex: IntoUsize,
    G::ColumnIndex: IntoUsize,
    G::Value: Number + ToPrimitive + Finite,
{
}

#[derive(Debug, Clone)]
struct WeightedUndirectedGraph {
    adjacency: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
    total_weight: f64,
}

impl WeightedUndirectedGraph {
    fn number_of_nodes(&self) -> usize {
        self.adjacency.len()
    }

    fn from_matrix<M>(matrix: &M) -> Result<Self, LouvainError>
    where
        M: SparseValuedMatrix2D,
        M::RowIndex: IntoUsize,
        M::ColumnIndex: IntoUsize,
        M::Value: ToPrimitive + Finite,
    {
        let rows = matrix.number_of_rows().into_usize();
        let columns = matrix.number_of_columns().into_usize();
        if rows != columns {
            return Err(LouvainError::NonSquareMatrix { rows, columns });
        }

        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows];
        let mut degree = vec![0.0; rows];

        for row_id in matrix.row_indices() {
            let source = row_id.into_usize();
            for (column_id, weight) in
                matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id))
            {
                let destination = column_id.into_usize();
                if !weight.is_finite() {
                    return Err(LouvainError::NonFiniteWeight { source, destination });
                }
                let weight = weight
                    .to_f64()
                    .ok_or(LouvainError::UnrepresentableWeight { source, destination })?;
                if !weight.is_finite() {
                    return Err(LouvainError::NonFiniteWeight { source, destination });
                }
                if weight <= 0.0 {
                    return Err(LouvainError::NonPositiveWeight { source, destination });
                }
                adjacency[source].push((destination, weight));
                degree[source] += weight;
            }
        }

        for (source, neighbors) in adjacency.iter().enumerate() {
            for (destination, weight) in neighbors {
                if !has_matching_edge(&adjacency[*destination], source, *weight) {
                    return Err(LouvainError::NonSymmetricEdge {
                        source,
                        destination: *destination,
                    });
                }
            }
        }

        let total_weight = degree.iter().sum();
        Ok(Self { adjacency, degree, total_weight })
    }

    fn induced(&self, partition: &[usize], number_of_communities: usize) -> Self {
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

fn validate_config(config: &LouvainConfig) -> Result<(), LouvainError> {
    if !config.resolution.is_finite() || config.resolution <= 0.0 {
        return Err(LouvainError::InvalidResolution);
    }
    if !config.modularity_threshold.is_finite() || config.modularity_threshold < 0.0 {
        return Err(LouvainError::InvalidModularityThreshold);
    }
    if config.max_levels == 0 {
        return Err(LouvainError::InvalidMaxLevels);
    }
    if config.max_local_passes == 0 {
        return Err(LouvainError::InvalidMaxLocalPasses);
    }
    Ok(())
}

fn local_moving(
    graph: &WeightedUndirectedGraph,
    config: &LouvainConfig,
    level_index: usize,
) -> (Vec<usize>, usize) {
    let number_of_nodes = graph.number_of_nodes();
    let mut partition: Vec<usize> = (0..number_of_nodes).collect();

    if number_of_nodes == 0 || graph.total_weight <= f64::EPSILON {
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
            if node_degree <= f64::EPSILON {
                continue;
            }

            let source_community = partition[node];
            touched_communities.clear();

            for (neighbor, weight) in &graph.adjacency[node] {
                let neighbor_community = partition[*neighbor];
                if weights_to_communities[neighbor_community] <= f64::EPSILON {
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

fn modularity(graph: &WeightedUndirectedGraph, partition: &[usize], resolution: f64) -> f64 {
    if graph.total_weight <= f64::EPSILON {
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
            if *total_weight <= f64::EPSILON {
                return modularity;
            }
            let total_fraction = *total_weight * inverse_total_weight;
            modularity + (*internal_weight * inverse_total_weight)
                - resolution * total_fraction * total_fraction
        },
    )
}

fn renumber_partition(partition: &mut [usize]) -> usize {
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

fn project_partition(
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

fn regroup_members(
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

fn marker_partition<Marker: PositiveInteger>(
    partition: &[usize],
) -> Result<Vec<Marker>, LouvainError> {
    partition
        .iter()
        .copied()
        .map(|community| {
            Marker::try_from_usize(community).map_err(|_| LouvainError::TooManyCommunities)
        })
        .collect()
}

fn mix_seed(seed: u64, level_index: usize, pass_index: usize) -> u64 {
    let level = level_index as u64;
    let pass = pass_index as u64;
    seed ^ level.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ pass.wrapping_mul(0xD1B5_4A32_D192_ED03)
}

fn has_matching_edge(row: &[(usize, f64)], destination: usize, weight: f64) -> bool {
    row.iter().any(|(column, candidate_weight)| {
        *column == destination && approx_eq(weight, *candidate_weight)
    })
}

fn approx_eq(left: f64, right: f64) -> bool {
    let tolerance = (left.abs().max(right.abs()).max(1.0)) * 16.0 * f64::EPSILON;
    (left - right).abs() <= tolerance
}
