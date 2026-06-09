//! Shared internals for modularity-based community detection algorithms.
//!
//! Each level's working graph is a [`ValuedCSR2D`] (the crate's standard sparse
//! matrix); [`UndirectedView`] borrows one and caches the per-node degree and
//! total weight, reading edges through [`SparseValuedMatrix2D`].

use alloc::{
    collections::{BTreeMap, VecDeque},
    vec::Vec,
};

use num_traits::{AsPrimitive, ToPrimitive};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::{
    impls::ValuedCSR2D,
    naive_structs::GenericEdgesBuilder,
    traits::{
        EdgesBuilder, Finite, Matrix2D, MonopartiteGraph, PositiveInteger, SparseMatrix2D,
        SparseValuedMatrix2D,
    },
};

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
    /// The maximum number of power iterations must be strictly positive.
    #[error("The maximum number of power iterations must be strictly positive.")]
    InvalidMaxPowerIterations,
    /// The power-iteration tolerance must be finite and non-negative.
    #[error("The power-iteration tolerance must be finite and non-negative.")]
    InvalidPowerTolerance,
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
    /// The supplied partition length does not match the number of nodes.
    #[error("The partition has {found} entries but the graph has {expected} nodes.")]
    PartitionLengthMismatch {
        /// Number of nodes in the graph.
        expected: usize,
        /// Number of entries in the supplied partition.
        found: usize,
    },
    /// The internal working graph could not be constructed from the validated
    /// edges, for example because an entry referenced an out-of-bounds node.
    #[error("Failed to construct the internal working graph.")]
    GraphConstructionFailed,
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

/// The working graph used internally by Louvain and Leiden: a weighted square
/// sparse matrix with `f64` weights. Level zero is built from the input matrix,
/// coarser levels from [`UndirectedView::induce`].
pub(crate) type WorkingGraph = ValuedCSR2D<usize, usize, usize, f64>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalMovingConfig {
    pub(crate) resolution: f64,
    pub(crate) max_local_passes: usize,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RefineConfig {
    pub(crate) resolution: f64,
    pub(crate) theta: f64,
    pub(crate) max_refinement_passes: usize,
    pub(crate) seed: u64,
}

/// The operations a graph must provide for the shared Louvain and Leiden
/// multi-level drivers ([`louvain_levels`](super::louvain::louvain_levels) and
/// [`leiden_levels`](super::leiden::leiden_levels)).
///
/// Implemented by both [`WorkingGraph`] (undirected, through a transient
/// [`UndirectedView`]) and the directed working graph, so the two detector
/// families share one driver per algorithm and differ only in the null model
/// baked into these methods.
pub(crate) trait CoarsenableGraph: Sized {
    /// Number of nodes in the graph.
    fn number_of_nodes(&self) -> usize;
    /// Modularity of `partition` under this graph's null model.
    fn modularity(&self, partition: &[usize], resolution: f64) -> f64;
    /// Greedy local-moving phase, returning the partition and the move count.
    fn local_moving(&self, config: LocalMovingConfig, level_index: usize) -> (Vec<usize>, usize);
    /// Coarsened graph obtained by contracting each community to a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the contracted graph cannot be assembled.
    fn induce(
        &self,
        partition: &[usize],
        number_of_communities: usize,
    ) -> Result<Self, ModularityError>;
    /// Leiden refinement phase restricted to within each parent community.
    fn refine(
        &self,
        parent_partition: &[usize],
        config: RefineConfig,
        level_index: usize,
    ) -> (Vec<usize>, usize);
    /// Splits any (weakly) disconnected community into its components.
    fn split_disconnected_communities(&self, partition: &mut [usize]);
}

/// Validates an input matrix and materializes it as a [`WorkingGraph`].
///
/// The matrix must be square, every weight finite, representable as `f64`, and
/// strictly positive, and the entries symmetric (an undirected graph).
pub(crate) fn build_working_graph<M>(matrix: &M) -> Result<WorkingGraph, ModularityError>
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

    let mut edge_weight: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for row_id in matrix.row_indices() {
        let source = row_id.as_();
        for (column_id, weight) in matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id)) {
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
            edge_weight.insert((source, destination), weight);
        }
    }

    for (&(source, destination), &weight) in &edge_weight {
        match edge_weight.get(&(destination, source)) {
            Some(&reverse) if approx_eq(weight, reverse) => {}
            _ => {
                return Err(ModularityError::NonSymmetricEdge {
                    source_id: source,
                    destination_id: destination,
                });
            }
        }
    }

    let edges: Vec<(usize, usize, f64)> = edge_weight
        .into_iter()
        .map(|((source, destination), weight)| (source, destination, weight))
        .collect();
    GenericEdgesBuilder::<_, WorkingGraph>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, rows))
        .edges(edges.into_iter())
        .build()
        .map_err(|_| ModularityError::GraphConstructionFailed)
}

/// Borrowed weighted undirected graph plus cached per-node degree and total
/// weight. Edges are read from the matrix through [`SparseValuedMatrix2D`].
pub(crate) struct UndirectedView<'graph> {
    graph: &'graph WorkingGraph,
    degree: Vec<f64>,
    total_weight: f64,
}

impl<'graph> UndirectedView<'graph> {
    /// Builds a view over a [`WorkingGraph`], caching the per-node degree and
    /// the total weight.
    ///
    /// The graph is trusted to be valid: input validation happens once in
    /// [`build_working_graph`], and coarser levels come from [`Self::induce`],
    /// which preserves validity. This constructor performs no checks.
    pub(crate) fn from_working_graph(graph: &'graph WorkingGraph) -> Self {
        let number_of_nodes = graph.number_of_rows();
        let mut degree = vec![0.0; number_of_nodes];
        for (node, value) in degree.iter_mut().enumerate() {
            *value = graph.sparse_row_values(node).sum();
        }
        let total_weight = degree.iter().sum();
        Self { graph, degree, total_weight }
    }

    pub(crate) fn number_of_nodes(&self) -> usize {
        self.degree.len()
    }

    /// Visits every neighbor of `node` as a `(neighbor, weight)` pair, reading
    /// the row straight from the matrix.
    pub(crate) fn for_each_neighbor<F: FnMut(usize, f64)>(&self, node: usize, mut visit: F) {
        for (neighbor, weight) in
            self.graph.sparse_row(node).zip(self.graph.sparse_row_values(node))
        {
            visit(neighbor, weight);
        }
    }

    /// Greedy local-moving phase: repeatedly assigns each node to the
    /// neighboring community that most increases modularity, breaking ties
    /// toward the smaller community id.
    pub(crate) fn local_moving(
        &self,
        config: LocalMovingConfig,
        level_index: usize,
    ) -> (Vec<usize>, usize) {
        let number_of_nodes = self.number_of_nodes();
        let mut partition: Vec<usize> = (0..number_of_nodes).collect();

        if number_of_nodes == 0 || self.total_weight <= 0.0 || !self.total_weight.is_normal() {
            return (partition, 0);
        }

        let mut community_totals = self.degree.clone();
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
                let node_degree = self.degree[node];
                if node_degree <= 0.0 {
                    continue;
                }

                let source_community = partition[node];
                touched_communities.clear();

                self.for_each_neighbor(node, |neighbor, weight| {
                    let neighbor_community = partition[neighbor];
                    if weights_to_communities[neighbor_community] == 0.0 {
                        touched_communities.push(neighbor_community);
                    }
                    weights_to_communities[neighbor_community] += weight;
                });

                community_totals[source_community] -= node_degree;

                let mut best_community = source_community;
                let mut best_gain = weights_to_communities[source_community]
                    - config.resolution * node_degree * community_totals[source_community]
                        / self.total_weight;

                for community in &touched_communities {
                    let community = *community;
                    let gain = weights_to_communities[community]
                        - config.resolution * node_degree * community_totals[community]
                            / self.total_weight;
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

    /// Splits any community that induces a disconnected subgraph into its
    /// connected components, assigning fresh community ids to the extra pieces.
    pub(crate) fn split_disconnected_communities(&self, partition: &mut [usize]) {
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
                    self.for_each_neighbor(node, |neighbor, _| {
                        if community_mask[neighbor] && !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    });
                }
            }

            for &node in &nodes {
                community_mask[node] = false;
                visited[node] = false;
            }
        }
    }

    /// Modularity of `partition` on this undirected graph.
    pub(crate) fn modularity(&self, partition: &[usize], resolution: f64) -> f64 {
        if self.total_weight <= 0.0 || !self.total_weight.is_normal() {
            return 0.0;
        }

        let number_of_communities = partition.iter().copied().max().map_or(0, |max| max + 1);
        let mut total_weight_per_community = vec![0.0; number_of_communities];
        let mut internal_weight_per_community = vec![0.0; number_of_communities];

        for (source, source_community) in partition.iter().copied().enumerate() {
            total_weight_per_community[source_community] += self.degree[source];
            self.for_each_neighbor(source, |destination, weight| {
                if partition[destination] == source_community {
                    internal_weight_per_community[source_community] += weight;
                }
            });
        }

        let inverse_total_weight = 1.0 / self.total_weight;
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

    /// Builds the coarsened graph obtained by contracting each community of
    /// `partition` into a single node, summing edge weights.
    ///
    /// # Errors
    ///
    /// Returns [`ModularityError::GraphConstructionFailed`] if the contracted
    /// edges cannot be assembled into a [`WorkingGraph`].
    pub(crate) fn induce(
        &self,
        partition: &[usize],
        number_of_communities: usize,
    ) -> Result<WorkingGraph, ModularityError> {
        let mut compact_edges: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for source in 0..self.number_of_nodes() {
            let source_community = partition[source];
            self.for_each_neighbor(source, |destination, weight| {
                let destination_community = partition[destination];
                *compact_edges.entry((source_community, destination_community)).or_insert(0.0) +=
                    weight;
            });
        }

        let edges: Vec<(usize, usize, f64)> = compact_edges
            .into_iter()
            .map(|((source, destination), weight)| (source, destination, weight))
            .collect();
        GenericEdgesBuilder::<_, WorkingGraph>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((number_of_communities, number_of_communities))
            .edges(edges.into_iter())
            .build()
            .map_err(|_| ModularityError::GraphConstructionFailed)
    }

    /// Borrows the per-node degree vector.
    pub(crate) fn degree(&self) -> &[f64] {
        &self.degree
    }

    /// Returns the total edge weight `2m`.
    pub(crate) fn total_weight(&self) -> f64 {
        self.total_weight
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

/// Returns whether two values are equal within a tolerance scaled by the
/// larger magnitude and floored at 1.0.
///
/// The scale is multiplied by the combined epsilon BEFORE being applied,
/// so the tolerance cannot overflow to infinity for values near
/// `f64::MAX` (a fuzzer-found pitfall: `huge * 16.0` overflows and the
/// resulting infinite tolerance would accept any pair as equal).
pub(crate) fn approx_eq(left: f64, right: f64) -> bool {
    let tolerance = left.abs().max(right.abs()).max(1.0) * (16.0 * f64::EPSILON);
    (left - right).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use num_traits::ToPrimitive;

    use super::{ModularityError, UndirectedView, WorkingGraph, build_working_graph};
    use crate::{
        impls::{CSR2D, GenericImplicitValuedMatrix2D},
        naive_structs::{GenericEdgesBuilder, GenericGraph},
        traits::{EdgesBuilder, Finite},
    };

    fn working(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WorkingGraph {
        let mut edges = edges;
        edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
        GenericEdgesBuilder::<_, WorkingGraph>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((node_count, node_count))
            .edges(edges.into_iter())
            .build()
            .unwrap()
    }

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
    fn test_build_working_graph_rejects_non_finite_after_to_f64_conversion() {
        let structure: CSR2D<usize, usize, usize> =
            GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
                .expected_number_of_edges(2)
                .expected_shape((2, 2))
                .edges(vec![(0, 1), (1, 0)].into_iter())
                .build()
                .unwrap();

        let matrix = GenericImplicitValuedMatrix2D::new(structure, |_| NonFiniteAfterConversion);

        assert!(matches!(
            build_working_graph(&matrix),
            Err(ModularityError::NonFiniteWeight { .. })
        ));
    }

    #[test]
    fn test_approx_eq_does_not_overflow_on_huge_values() {
        assert!(!super::approx_eq(2.527457358493282e307, 1.656712100688973e-24));
        assert!(super::approx_eq(2.527457358493282e307, 2.527457358493282e307));
        assert!(!super::approx_eq(2.5e307, 2.6e307));
        assert!(super::approx_eq(1e-300, 2e-300));
    }

    #[test]
    fn test_build_working_graph_rejects_huge_weight_asymmetry() {
        let matrix: WorkingGraph = GenericEdgesBuilder::<_, WorkingGraph>::default()
            .expected_number_of_edges(2)
            .expected_shape((2, 2))
            .edges(vec![(0, 1, 2.527457358493282e307), (1, 0, 1.656712100688973e-24)].into_iter())
            .build()
            .unwrap();

        assert!(matches!(
            build_working_graph(&matrix),
            Err(ModularityError::NonSymmetricEdge { .. })
        ));
    }

    #[test]
    fn test_split_disconnected_communities_returns_for_single_community_partition() {
        let graph = working(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let view = UndirectedView::from_working_graph(&graph);
        let mut partition = vec![0usize, 0usize];

        view.split_disconnected_communities(&mut partition);

        assert_eq!(partition, vec![0, 0]);
    }

    #[test]
    fn test_split_disconnected_communities_splits_multi_component_community() {
        let graph =
            working(5, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0), (4, 4, 1.0)]);
        let view = UndirectedView::from_working_graph(&graph);
        let mut partition = vec![0usize, 0usize, 0usize, 0usize, 1usize];

        view.split_disconnected_communities(&mut partition);

        assert_eq!(partition[0], partition[1]);
        assert_eq!(partition[2], partition[3]);
        assert_ne!(partition[0], partition[2]);
    }
}
