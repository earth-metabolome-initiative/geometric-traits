//! Shared internals for the directed Louvain and directed Leiden community
//! detectors.
//!
//! These detectors run the same greedy skeleton as the undirected
//! [`Louvain`](super::Louvain) and [`Leiden`](super::Leiden), but optimize the
//! directed (Leicht-Newman / Dugue-Perez) modularity, so the null model keeps
//! each node's in-strength and out-strength separate:
//!
//! ```text
//! Q = sum_C [ internal_C / m - gamma * (Sigma_out^C * Sigma_in^C) / m^2 ]
//! ```
//!
//! Local moving needs both outgoing and incoming arcs of every node, and the
//! coarsened levels must keep arc direction. The generic matrix bound only
//! exposes rows (outgoing arcs), so [`DirectedWorkingGraph`] caches both a CSR
//! out-adjacency and a CSC in-adjacency, built once from the input matrix and
//! preserved across [`induce`](DirectedWorkingGraph::induce).

use alloc::{collections::BTreeMap, vec::Vec};

use num_traits::{AsPrimitive, ToPrimitive};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use super::{
    leiden::sample_softmax_destination,
    modularity::{
        CoarsenableGraph, LocalMovingConfig, ModularityError, RefineConfig, approx_eq, mix_seed,
    },
};
use crate::traits::{Finite, SparseValuedMatrix2D};

/// A weighted directed graph stored as both a CSR out-adjacency and a CSC
/// in-adjacency, with cached per-node strengths and total arc weight.
///
/// Level zero is built from the input matrix by [`Self::from_matrix`], coarser
/// levels by [`Self::induce`]. Both constructors take arcs grouped by ascending
/// source, which keeps the adjacency order deterministic.
pub(crate) struct DirectedWorkingGraph {
    /// CSR row offsets into `out_targets`/`out_weights`, length `n + 1`.
    out_offsets: Vec<usize>,
    /// Destination node of each outgoing arc.
    out_targets: Vec<usize>,
    /// Weight of each outgoing arc.
    out_weights: Vec<f64>,
    /// CSC column offsets into `in_sources`/`in_weights`, length `n + 1`.
    in_offsets: Vec<usize>,
    /// Source node of each incoming arc.
    in_sources: Vec<usize>,
    /// Weight of each incoming arc.
    in_weights: Vec<f64>,
    /// Out-strength (sum of outgoing arc weights) of each node.
    out_degree: Vec<f64>,
    /// In-strength (sum of incoming arc weights) of each node.
    in_degree: Vec<f64>,
    /// Total arc weight `m = sum_ij A_ij`.
    total_weight: f64,
}

/// Directed gain of moving an isolated node into community `C`.
///
/// `out_to_c`/`in_to_c` are the arc weights from the node into `C` and from `C`
/// into the node; `k_out`/`k_in` the node strengths; `sigma_out_c`/`sigma_in_c`
/// the community strengths excluding the node; `inverse_total` is `1 / m`.
#[allow(clippy::too_many_arguments)]
fn directed_gain(
    out_to_c: f64,
    in_to_c: f64,
    k_out: f64,
    k_in: f64,
    sigma_out_c: f64,
    sigma_in_c: f64,
    resolution: f64,
    inverse_total: f64,
) -> f64 {
    (out_to_c + in_to_c) * inverse_total
        - resolution * (k_out * sigma_in_c + k_in * sigma_out_c) * inverse_total * inverse_total
}

/// Resets the scratch accumulators touched while scanning a node's arcs.
fn clear_arcs(
    source_community: usize,
    touched_communities: &[usize],
    out_to_communities: &mut [f64],
    in_to_communities: &mut [f64],
) {
    for community in touched_communities {
        out_to_communities[*community] = 0.0;
        in_to_communities[*community] = 0.0;
    }
    out_to_communities[source_community] = 0.0;
    in_to_communities[source_community] = 0.0;
}

/// Prefix-sums `counts` into offsets of length `counts.len() + 1`.
fn prefix_sum(counts: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut running = 0usize;
    offsets.push(0);
    for &count in counts {
        running += count;
        offsets.push(running);
    }
    offsets
}

impl DirectedWorkingGraph {
    /// Builds a view over `matrix`, validating that it is square and that every
    /// weight is finite, representable as `f64`, and strictly positive.
    ///
    /// Unlike the undirected `build_working_graph`, no symmetry is required and
    /// self-loops are allowed.
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

        // `row_indices()` yields the contiguous range `0..rows` in order, so the
        // collected arcs are grouped by ascending source with sorted targets.
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
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
                edges.push((source, destination, weight));
            }
        }

        Ok(Self::from_sorted_edges(rows, &edges))
    }

    /// Builds the graph from arcs already grouped by ascending source.
    ///
    /// The outgoing adjacency is just the edge list (already in CSR order); the
    /// incoming adjacency is scattered into per-destination buckets, where the
    /// ascending source grouping leaves each bucket sorted by source.
    fn from_sorted_edges(number_of_nodes: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut out_counts = vec![0usize; number_of_nodes];
        let mut in_counts = vec![0usize; number_of_nodes];
        let mut out_degree = vec![0.0; number_of_nodes];
        let mut in_degree = vec![0.0; number_of_nodes];
        for &(source, destination, weight) in edges {
            out_counts[source] += 1;
            in_counts[destination] += 1;
            out_degree[source] += weight;
            in_degree[destination] += weight;
        }

        let out_offsets = prefix_sum(&out_counts);
        let out_targets: Vec<usize> =
            edges.iter().map(|&(_, destination, _)| destination).collect();
        let out_weights: Vec<f64> = edges.iter().map(|&(_, _, weight)| weight).collect();

        let in_offsets = prefix_sum(&in_counts);
        let mut in_sources = vec![0usize; edges.len()];
        let mut in_weights = vec![0.0; edges.len()];
        let mut cursor = in_offsets[..number_of_nodes].to_vec();
        for &(source, destination, weight) in edges {
            let position = cursor[destination];
            in_sources[position] = source;
            in_weights[position] = weight;
            cursor[destination] += 1;
        }

        let total_weight = out_weights.iter().sum();
        Self {
            out_offsets,
            out_targets,
            out_weights,
            in_offsets,
            in_sources,
            in_weights,
            out_degree,
            in_degree,
            total_weight,
        }
    }

    /// Number of nodes in the graph.
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.out_degree.len()
    }

    /// Total arc weight `m`.
    pub(crate) fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Visits every outgoing arc of `node` as a `(destination, weight)` pair.
    pub(crate) fn for_each_out_edge<F: FnMut(usize, f64)>(&self, node: usize, mut visit: F) {
        for index in self.out_offsets[node]..self.out_offsets[node + 1] {
            visit(self.out_targets[index], self.out_weights[index]);
        }
    }

    /// Visits every incoming arc of `node` as a `(source, weight)` pair.
    pub(crate) fn for_each_in_edge<F: FnMut(usize, f64)>(&self, node: usize, mut visit: F) {
        for index in self.in_offsets[node]..self.in_offsets[node + 1] {
            visit(self.in_sources[index], self.in_weights[index]);
        }
    }

    /// Directed modularity of `partition`, whose entries are community ids in
    /// `0..number_of_communities`.
    pub(crate) fn modularity(&self, partition: &[usize], resolution: f64) -> f64 {
        if self.total_weight <= 0.0 || !self.total_weight.is_normal() {
            return 0.0;
        }

        let number_of_communities = partition.iter().copied().max().map_or(0, |max| max + 1);
        let mut internal_weight = vec![0.0; number_of_communities];
        let mut out_strength = vec![0.0; number_of_communities];
        let mut in_strength = vec![0.0; number_of_communities];

        for (node, community) in partition.iter().copied().enumerate() {
            out_strength[community] += self.out_degree[node];
            in_strength[community] += self.in_degree[node];
            self.for_each_out_edge(node, |destination, weight| {
                if partition[destination] == community {
                    internal_weight[community] += weight;
                }
            });
        }

        let inverse_total = 1.0 / self.total_weight;
        (0..number_of_communities).fold(0.0, |modularity, community| {
            modularity + internal_weight[community] * inverse_total
                - resolution
                    * (out_strength[community] * inverse_total)
                    * (in_strength[community] * inverse_total)
        })
    }

    /// Greedy local-moving phase: repeatedly assigns each node to the
    /// neighboring community that most increases the directed modularity,
    /// breaking ties toward the smaller community id.
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

        let inverse_total = 1.0 / self.total_weight;
        let mut community_out_totals = self.out_degree.clone();
        let mut community_in_totals = self.in_degree.clone();
        let mut order: Vec<usize> = (0..number_of_nodes).collect();
        let mut touched_communities: Vec<usize> = Vec::new();
        let mut out_to_communities = vec![0.0; number_of_nodes];
        let mut in_to_communities = vec![0.0; number_of_nodes];
        let mut moved_nodes = 0usize;

        for pass_index in 0..config.max_local_passes {
            let mut rng = SmallRng::seed_from_u64(mix_seed(config.seed, level_index, pass_index));
            order.shuffle(&mut rng);

            let mut moved_in_pass = 0usize;

            for node in &order {
                let node = *node;
                let node_out = self.out_degree[node];
                let node_in = self.in_degree[node];
                if node_out <= 0.0 && node_in <= 0.0 {
                    continue;
                }

                let source_community = partition[node];
                touched_communities.clear();
                self.accumulate_arcs(
                    node,
                    &partition,
                    &mut touched_communities,
                    &mut out_to_communities,
                    &mut in_to_communities,
                );

                community_out_totals[source_community] -= node_out;
                community_in_totals[source_community] -= node_in;

                let mut best_community = source_community;
                let mut best_gain = directed_gain(
                    out_to_communities[source_community],
                    in_to_communities[source_community],
                    node_out,
                    node_in,
                    community_out_totals[source_community],
                    community_in_totals[source_community],
                    config.resolution,
                    inverse_total,
                );

                for community in &touched_communities {
                    let community = *community;
                    let gain = directed_gain(
                        out_to_communities[community],
                        in_to_communities[community],
                        node_out,
                        node_in,
                        community_out_totals[community],
                        community_in_totals[community],
                        config.resolution,
                        inverse_total,
                    );
                    if gain > best_gain + f64::EPSILON
                        || (approx_eq(gain, best_gain) && community < best_community)
                    {
                        best_gain = gain;
                        best_community = community;
                    }
                }

                partition[node] = best_community;
                community_out_totals[best_community] += node_out;
                community_in_totals[best_community] += node_in;

                if best_community != source_community {
                    moved_in_pass += 1;
                    moved_nodes += 1;
                }

                clear_arcs(
                    source_community,
                    &touched_communities,
                    &mut out_to_communities,
                    &mut in_to_communities,
                );
            }

            if moved_in_pass == 0 {
                break;
            }
        }

        (partition, moved_nodes)
    }

    /// Accumulates, for `node`, the arc weight to and from every neighboring
    /// community, recording each touched community exactly once.
    ///
    /// Self-loops are skipped: a self-loop is internal to whichever community
    /// the node joins, so it adds the same constant to every candidate's gain
    /// and cancels in the comparison, while its weight already enters the null
    /// model through the node's in- and out-strength.
    fn accumulate_arcs(
        &self,
        node: usize,
        partition: &[usize],
        touched_communities: &mut Vec<usize>,
        out_to_communities: &mut [f64],
        in_to_communities: &mut [f64],
    ) {
        self.for_each_out_edge(node, |destination, weight| {
            if destination == node {
                return;
            }
            let community = partition[destination];
            if out_to_communities[community] == 0.0 && in_to_communities[community] == 0.0 {
                touched_communities.push(community);
            }
            out_to_communities[community] += weight;
        });
        self.for_each_in_edge(node, |source, weight| {
            if source == node {
                return;
            }
            let community = partition[source];
            if out_to_communities[community] == 0.0 && in_to_communities[community] == 0.0 {
                touched_communities.push(community);
            }
            in_to_communities[community] += weight;
        });
    }

    /// Builds the coarsened graph obtained by contracting each community of
    /// `partition` into a single node, summing arc weights and keeping arc
    /// direction (so `i -> j` stays distinct from `j -> i`, and arcs internal
    /// to a community become its diagonal entry).
    pub(crate) fn induce(&self, partition: &[usize], number_of_communities: usize) -> Self {
        let mut compact_edges: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for source in 0..self.number_of_nodes() {
            let source_community = partition[source];
            self.for_each_out_edge(source, |destination, weight| {
                let destination_community = partition[destination];
                *compact_edges.entry((source_community, destination_community)).or_insert(0.0) +=
                    weight;
            });
        }

        let edges: Vec<(usize, usize, f64)> = compact_edges
            .into_iter()
            .map(|((source, destination), weight)| (source, destination, weight))
            .collect();
        Self::from_sorted_edges(number_of_communities, &edges)
    }

    /// Splits any community that is weakly disconnected into its connected
    /// components (ignoring arc direction), assigning fresh community ids to
    /// the extra pieces. This preserves the Leiden well-connectedness
    /// guarantee on directed graphs.
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
        let mut stack: Vec<usize> = Vec::new();
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

                let target_community = if first_component {
                    community_id
                } else {
                    let id = next_community_id;
                    next_community_id += 1;
                    id
                };
                first_component = false;

                visited[start] = true;
                stack.push(start);
                while let Some(node) = stack.pop() {
                    partition[node] = target_community;
                    let mut visit_neighbor = |neighbor: usize| {
                        if community_mask[neighbor] && !visited[neighbor] {
                            visited[neighbor] = true;
                            stack.push(neighbor);
                        }
                    };
                    self.for_each_out_edge(node, |destination, _| visit_neighbor(destination));
                    self.for_each_in_edge(node, |source, _| visit_neighbor(source));
                }
            }

            for &node in &nodes {
                community_mask[node] = false;
                visited[node] = false;
            }
        }
    }
}

/// The directed working graph satisfies [`CoarsenableGraph`] directly: it is
/// self-contained (it owns both adjacencies), so the shared Leiden and Louvain
/// drivers run over it with the directed null model baked into these methods.
impl CoarsenableGraph for DirectedWorkingGraph {
    #[inline]
    fn number_of_nodes(&self) -> usize {
        self.number_of_nodes()
    }

    #[inline]
    fn modularity(&self, partition: &[usize], resolution: f64) -> f64 {
        self.modularity(partition, resolution)
    }

    #[inline]
    fn local_moving(&self, config: LocalMovingConfig, level_index: usize) -> (Vec<usize>, usize) {
        self.local_moving(config, level_index)
    }

    #[inline]
    fn induce(
        &self,
        partition: &[usize],
        number_of_communities: usize,
    ) -> Result<Self, ModularityError> {
        Ok(self.induce(partition, number_of_communities))
    }

    #[inline]
    fn refine(
        &self,
        parent_partition: &[usize],
        config: RefineConfig,
        level_index: usize,
    ) -> (Vec<usize>, usize) {
        directed_refine_partition(self, parent_partition, config, level_index)
    }

    #[inline]
    fn split_disconnected_communities(&self, partition: &mut [usize]) {
        self.split_disconnected_communities(partition);
    }
}

/// Leiden refinement on a directed graph: within each parent community, moves
/// nodes between sub-communities by the directed gain, sampling the destination
/// with the softmax rule shared with the undirected Leiden.
#[allow(clippy::too_many_lines)]
pub(crate) fn directed_refine_partition(
    graph: &DirectedWorkingGraph,
    parent_partition: &[usize],
    config: RefineConfig,
    level_index: usize,
) -> (Vec<usize>, usize) {
    let number_of_nodes = graph.number_of_nodes();
    let mut refined_partition: Vec<usize> = (0..number_of_nodes).collect();

    let total_weight = graph.total_weight();
    if number_of_nodes == 0 || total_weight <= 0.0 || !total_weight.is_normal() {
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

    let inverse_total = 1.0 / total_weight;
    let mut community_out_totals = graph.out_degree.clone();
    let mut community_in_totals = graph.in_degree.clone();
    let mut out_to_communities = vec![0.0; number_of_nodes];
    let mut in_to_communities = vec![0.0; number_of_nodes];
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
                let node_out = graph.out_degree[node];
                let node_in = graph.in_degree[node];
                if node_out <= 0.0 && node_in <= 0.0 {
                    continue;
                }

                let source_community = refined_partition[node];
                touched_communities.clear();
                accumulate_refine_arcs(
                    graph,
                    node,
                    parent_community,
                    parent_partition,
                    &refined_partition,
                    &mut touched_communities,
                    &mut out_to_communities,
                    &mut in_to_communities,
                );

                community_out_totals[source_community] -= node_out;
                community_in_totals[source_community] -= node_in;

                let current_gain = directed_gain(
                    out_to_communities[source_community],
                    in_to_communities[source_community],
                    node_out,
                    node_in,
                    community_out_totals[source_community],
                    community_in_totals[source_community],
                    config.resolution,
                    inverse_total,
                );

                candidate_moves.clear();
                for community in &touched_communities {
                    let community = *community;
                    if community == source_community {
                        continue;
                    }
                    let gain = directed_gain(
                        out_to_communities[community],
                        in_to_communities[community],
                        node_out,
                        node_in,
                        community_out_totals[community],
                        community_in_totals[community],
                        config.resolution,
                        inverse_total,
                    );
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
                community_out_totals[destination] += node_out;
                community_in_totals[destination] += node_in;

                if destination != source_community {
                    moved_in_pass += 1;
                    moved_nodes += 1;
                }

                for community in &touched_communities {
                    out_to_communities[*community] = 0.0;
                    in_to_communities[*community] = 0.0;
                }
                out_to_communities[source_community] = 0.0;
                in_to_communities[source_community] = 0.0;
            }
        }

        if moved_in_pass == 0 {
            break;
        }
    }

    (refined_partition, moved_nodes)
}

/// Accumulates arc weights to and from neighboring sub-communities, restricted
/// to neighbors inside the node's parent community.
#[allow(clippy::too_many_arguments)]
fn accumulate_refine_arcs(
    graph: &DirectedWorkingGraph,
    node: usize,
    parent_community: usize,
    parent_partition: &[usize],
    refined_partition: &[usize],
    touched_communities: &mut Vec<usize>,
    out_to_communities: &mut [f64],
    in_to_communities: &mut [f64],
) {
    graph.for_each_out_edge(node, |destination, weight| {
        if destination == node || parent_partition[destination] != parent_community {
            return;
        }
        let community = refined_partition[destination];
        if out_to_communities[community] == 0.0 && in_to_communities[community] == 0.0 {
            touched_communities.push(community);
        }
        out_to_communities[community] += weight;
    });
    graph.for_each_in_edge(node, |source, weight| {
        if source == node || parent_partition[source] != parent_community {
            return;
        }
        let community = refined_partition[source];
        if out_to_communities[community] == 0.0 && in_to_communities[community] == 0.0 {
            touched_communities.push(community);
        }
        in_to_communities[community] += weight;
    });
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        DirectedWorkingGraph, LocalMovingConfig, ModularityError, RefineConfig, directed_gain,
        directed_refine_partition,
    };
    use crate::{impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder};

    type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

    const TOLERANCE: f64 = 1e-12;

    fn build(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WeightedMatrix {
        let mut edges = edges;
        edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
        GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((node_count, node_count))
            .edges(edges.into_iter())
            .build()
            .unwrap()
    }

    fn graph(node_count: usize, edges: Vec<(usize, usize, f64)>) -> DirectedWorkingGraph {
        DirectedWorkingGraph::from_matrix(&build(node_count, edges)).unwrap()
    }

    fn collect_out(graph: &DirectedWorkingGraph, node: usize) -> Vec<(usize, f64)> {
        let mut edges = Vec::new();
        graph.for_each_out_edge(node, |destination, weight| edges.push((destination, weight)));
        edges
    }

    fn collect_in(graph: &DirectedWorkingGraph, node: usize) -> Vec<(usize, f64)> {
        let mut edges = Vec::new();
        graph.for_each_in_edge(node, |source, weight| edges.push((source, weight)));
        edges
    }

    #[test]
    fn test_from_matrix_builds_both_adjacencies() {
        // 0->1 (w=2), 0->2 (w=1), 2->0 (w=3).
        let graph = graph(3, vec![(0, 1, 2.0), (0, 2, 1.0), (2, 0, 3.0)]);
        assert_eq!(graph.number_of_nodes(), 3);
        assert_eq!(collect_out(&graph, 0), vec![(1, 2.0), (2, 1.0)]);
        assert_eq!(collect_out(&graph, 1), vec![]);
        assert_eq!(collect_out(&graph, 2), vec![(0, 3.0)]);
        // Incoming arcs: node 0 receives from 2, node 1 from 0, node 2 from 0.
        assert_eq!(collect_in(&graph, 0), vec![(2, 3.0)]);
        assert_eq!(collect_in(&graph, 1), vec![(0, 2.0)]);
        assert_eq!(collect_in(&graph, 2), vec![(0, 1.0)]);
        assert!((graph.total_weight() - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_from_matrix_handles_self_loops_and_sinks() {
        // Self-loop 0->0 (w=2), 0->1 (w=1); node 1 is a sink (out-strength 0),
        // node 2 is a source-less, sink-less isolate.
        let graph = graph(3, vec![(0, 0, 2.0), (0, 1, 1.0)]);
        assert_eq!(collect_out(&graph, 0), vec![(0, 2.0), (1, 1.0)]);
        assert_eq!(collect_in(&graph, 0), vec![(0, 2.0)]);
        assert_eq!(collect_in(&graph, 1), vec![(0, 1.0)]);
        assert_eq!(collect_out(&graph, 1), vec![]);
        assert_eq!(collect_out(&graph, 2), vec![]);
        assert_eq!(collect_in(&graph, 2), vec![]);
    }

    #[test]
    fn test_modularity_matches_hand_values() {
        // Two disjoint directed 2-cycles: correct split is 0.5.
        let graph = graph(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
        assert!((graph.modularity(&[0, 0, 1, 1], 1.0) - 0.5).abs() < TOLERANCE);
        assert!(graph.modularity(&[0, 0, 0, 0], 1.0).abs() < TOLERANCE);
        // With gamma = 2 each community nets to zero.
        assert!(graph.modularity(&[0, 0, 1, 1], 2.0).abs() < TOLERANCE);
    }

    /// Moving a node and recomputing the modularity must equal the incremental
    /// directed gain difference (gain at destination minus gain at source).
    /// This is the decisive self-validation of the gain formula.
    #[test]
    fn test_incremental_gain_equals_modularity_delta() {
        let graph = graph(
            5,
            vec![
                (0, 1, 1.0),
                (1, 2, 2.0),
                (2, 0, 1.0),
                (3, 4, 1.0),
                (4, 3, 1.0),
                (1, 3, 1.0),
                (0, 0, 1.0),
            ],
        );
        let resolution = 1.3;
        let inverse_total = 1.0 / graph.total_weight();

        // Start from a partition, move node 1 from community A into community B,
        // measuring the gain by the same accumulation the local-moving uses.
        // Move node 0, which carries a self-loop, to exercise the self-loop
        // handling: a self-loop must not enter the arc-to-community terms.
        let mut partition = vec![0usize, 0, 0, 1, 1];
        let node = 0;
        let source = partition[node];
        let destination = 1;

        let modularity_before = graph.modularity(&partition, resolution);

        // Community strengths excluding `node`, and the node's arc weights to
        // each community.
        let node_out = graph.out_degree[node];
        let node_in = graph.in_degree[node];
        let community_total = |partition: &[usize], community: usize| {
            let mut sigma_out = 0.0;
            let mut sigma_in = 0.0;
            for (other, other_community) in partition.iter().copied().enumerate() {
                if other != node && other_community == community {
                    sigma_out += graph.out_degree[other];
                    sigma_in += graph.in_degree[other];
                }
            }
            (sigma_out, sigma_in)
        };
        let arcs_to = |partition: &[usize], community: usize| {
            let mut out_to = 0.0;
            let mut in_to = 0.0;
            graph.for_each_out_edge(node, |dst, w| {
                if dst != node && partition[dst] == community {
                    out_to += w;
                }
            });
            graph.for_each_in_edge(node, |src, w| {
                if src != node && partition[src] == community {
                    in_to += w;
                }
            });
            (out_to, in_to)
        };

        let (source_out, source_in) = community_total(&partition, source);
        let (source_arc_out, source_arc_in) = arcs_to(&partition, source);
        let gain_source = directed_gain(
            source_arc_out,
            source_arc_in,
            node_out,
            node_in,
            source_out,
            source_in,
            resolution,
            inverse_total,
        );

        partition[node] = destination;
        let modularity_after = graph.modularity(&partition, resolution);
        let (dest_out, dest_in) = community_total(&partition, destination);
        let (dest_arc_out, dest_arc_in) = arcs_to(&partition, destination);
        let gain_destination = directed_gain(
            dest_arc_out,
            dest_arc_in,
            node_out,
            node_in,
            dest_out,
            dest_in,
            resolution,
            inverse_total,
        );

        let measured = modularity_after - modularity_before;
        let predicted = gain_destination - gain_source;
        assert!(
            (measured - predicted).abs() < 1e-12,
            "measured delta {measured:.15} != predicted {predicted:.15}",
        );
    }

    #[test]
    fn test_induce_preserves_direction_and_diagonal() {
        // 0->1, 1->0 inside community 0; 2->3 inside community 1; 1->2 across.
        let graph = graph(4, vec![(0, 1, 1.0), (1, 0, 2.0), (2, 3, 4.0), (1, 2, 5.0)]);
        let induced = graph.induce(&[0, 0, 1, 1], 2);
        // Community 0 internal arcs (0->1 + 1->0 = 3) collapse to the diagonal
        // (0,0); the cross arc 1->2 becomes 0->1; community 1 internal 2->3 = 4
        // becomes the diagonal (1,1).
        assert_eq!(collect_out(&induced, 0), vec![(0, 3.0), (1, 5.0)]);
        assert_eq!(collect_out(&induced, 1), vec![(1, 4.0)]);
        // The induced modularity of the trivial single-community-per-node
        // partition equals the original two-community modularity.
        let original = graph.modularity(&[0, 0, 1, 1], 1.0);
        let coarse = induced.modularity(&[0, 1], 1.0);
        assert!((original - coarse).abs() < TOLERANCE);
    }

    #[test]
    fn test_split_separates_weakly_disconnected_community() {
        // Two directed 2-cycles wrongly merged into community 0, plus a separate
        // community 1, so the single-community early return does not apply. The
        // disconnected community 0 must split into two weakly connected pieces.
        let graph = graph(5, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0), (4, 4, 1.0)]);
        let mut partition = vec![0usize, 0, 0, 0, 1];
        graph.split_disconnected_communities(&mut partition);
        assert_eq!(partition[0], partition[1]);
        assert_eq!(partition[2], partition[3]);
        assert_ne!(partition[0], partition[2]);
        assert_eq!(partition[4], 1);
    }

    #[test]
    fn test_degenerate_graphs_do_not_panic() {
        let config = super::LocalMovingConfig { resolution: 1.0, max_local_passes: 10, seed: 1 };

        // Empty graph.
        let empty = graph(0, vec![]);
        assert!(empty.modularity(&[], 1.0).abs() < TOLERANCE);
        assert_eq!(empty.local_moving(config, 0).0, Vec::<usize>::new());

        // Single node, no arcs.
        let single = graph(1, vec![]);
        assert_eq!(single.local_moving(config, 0).0, vec![0]);

        // All self-loops: every node is its own source and sink.
        let loops = graph(3, vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let (partition, _) = loops.local_moving(config, 0);
        assert_eq!(partition.len(), 3);
        assert!(loops.modularity(&partition, 1.0).is_finite());
    }

    #[test]
    fn test_from_matrix_rejects_non_square() {
        let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 3))
            .edges(vec![(0, 1, 1.0)].into_iter())
            .build()
            .unwrap();
        assert!(matches!(
            DirectedWorkingGraph::from_matrix(&matrix),
            Err(ModularityError::NonSquareMatrix { rows: 2, columns: 3 })
        ));
    }

    #[test]
    fn test_from_matrix_rejects_non_finite_and_non_positive_weights() {
        assert!(matches!(
            DirectedWorkingGraph::from_matrix(&build(2, vec![(0, 1, f64::NAN)])),
            Err(ModularityError::NonFiniteWeight { source_id: 0, destination_id: 1 })
        ));
        assert!(matches!(
            DirectedWorkingGraph::from_matrix(&build(2, vec![(0, 1, -1.0)])),
            Err(ModularityError::NonPositiveWeight { source_id: 0, destination_id: 1 })
        ));
    }

    #[test]
    fn test_local_moving_skips_isolated_nodes() {
        // Node 2 has no arcs at all: local moving must skip it without panicking.
        let graph = graph(3, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let config = LocalMovingConfig { resolution: 1.0, max_local_passes: 10, seed: 3 };
        let (partition, _) = graph.local_moving(config, 0);
        assert_eq!(partition.len(), 3);
        assert_eq!(partition[0], partition[1]);
    }

    #[test]
    fn test_split_returns_early_on_trivial_partitions() {
        // One node: nothing to split.
        let single = graph(1, vec![]);
        let mut one = vec![0usize];
        single.split_disconnected_communities(&mut one);
        assert_eq!(one, vec![0]);

        // Several nodes but a single community: nothing to split.
        let two = graph(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let mut single_community = vec![0usize, 0];
        two.split_disconnected_communities(&mut single_community);
        assert_eq!(single_community, vec![0, 0]);
    }

    #[test]
    fn test_refine_returns_early_on_degenerate_input() {
        let config =
            RefineConfig { resolution: 1.0, theta: 0.01, max_refinement_passes: 5, seed: 1 };
        // Empty graph: no nodes to refine.
        let empty = graph(0, vec![]);
        assert_eq!(directed_refine_partition(&empty, &[], config, 0).0, Vec::<usize>::new());
        // Non-empty graph but empty parent partition: nothing to do.
        let two = graph(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        assert_eq!(directed_refine_partition(&two, &[], config, 0), (vec![0, 1], 0));
    }

    #[test]
    fn test_refine_skips_self_loops_cross_parent_and_zero_degree_nodes() {
        // Parent community 0 = {0, 1, 3}: node 0 carries a self-loop, node 1 has
        // an arc into the other parent community, and node 3 is isolated. The
        // refinement must skip the self-loop, the cross-parent arc, and the
        // zero-degree node without panicking.
        let graph = graph(4, vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0)]);
        let config =
            RefineConfig { resolution: 1.0, theta: 0.01, max_refinement_passes: 5, seed: 2 };
        let parent_partition = vec![0usize, 0, 1, 0];
        let (refined, _) = directed_refine_partition(&graph, &parent_partition, config, 0);
        assert_eq!(refined.len(), 4);
    }
}
