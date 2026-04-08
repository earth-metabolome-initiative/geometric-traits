//! Exact minimum-cycle-basis computation for undirected simple graphs.
//!
//! The implementation is graph-theoretic and chemistry-agnostic:
//!
//! - it decomposes the graph into connected components
//! - it computes an exact minimum cycle basis on each component
//! - it follows the unweighted de Pina / Kavitha-style orthogonal-cycle
//!   construction used by NetworkX's `minimum_cycle_basis`

use alloc::{
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    vec,
    vec::Vec,
};
use core::cmp::Reverse;

use crate::traits::{
    ConnectedComponents, MonopartiteGraph, PositiveInteger, UndirectedMonopartiteMonoplexGraph,
};

/// Result returned by [`MinimumCycleBasis::minimum_cycle_basis`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimumCycleBasisResult<NodeId: PositiveInteger> {
    minimum_cycle_basis: Vec<Vec<NodeId>>,
    cycle_rank: usize,
    total_weight: usize,
}

type MeetingState = (usize, usize, Vec<Option<usize>>, Vec<Option<usize>>);

impl<NodeId: PositiveInteger> MinimumCycleBasisResult<NodeId> {
    /// Returns the cycles in the minimum cycle basis.
    #[inline]
    pub fn minimum_cycle_basis(&self) -> core::slice::Iter<'_, Vec<NodeId>> {
        self.minimum_cycle_basis.iter()
    }

    /// Returns the cyclomatic number of the graph.
    #[inline]
    #[must_use]
    pub fn cycle_rank(&self) -> usize {
        self.cycle_rank
    }

    /// Returns the total weight of the basis.
    ///
    /// For unweighted graphs this is the sum of the cycle lengths.
    #[inline]
    #[must_use]
    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    /// Returns the number of cycles in the basis.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.minimum_cycle_basis.len()
    }

    /// Returns whether the basis is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.minimum_cycle_basis.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for minimum-cycle-basis computation.
pub enum MinimumCycleBasisError {
    /// The graph contains self-loops, which are outside the simple-graph
    /// contract used by the current algorithm.
    #[error(
        "The minimum-cycle-basis algorithm currently supports only simple undirected graphs and does not accept self-loops."
    )]
    SelfLoopsUnsupported,
    /// The orthogonal-cycle construction unexpectedly failed to span a cyclic
    /// connected component.
    #[error(
        "The minimum-cycle-basis algorithm did not generate enough independent candidate cycles for one connected component."
    )]
    CandidateSetIncomplete,
}

impl From<MinimumCycleBasisError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: MinimumCycleBasisError) -> Self {
        Self::MinimumCycleBasisError(error)
    }
}

impl<G: MonopartiteGraph> From<MinimumCycleBasisError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: MinimumCycleBasisError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalGraph<NodeId: PositiveInteger> {
    vertices: Vec<NodeId>,
    edges: Vec<[usize; 2]>,
    adjacency: Vec<Vec<usize>>,
    edge_lookup: BTreeMap<[usize; 2], usize>,
}

impl<NodeId: PositiveInteger> LocalGraph<NodeId> {
    fn from_component(vertices: Vec<NodeId>, edges: Vec<[NodeId; 2]>) -> Self {
        let mut global_to_local = BTreeMap::new();
        for (local_id, &vertex) in vertices.iter().enumerate() {
            global_to_local.insert(vertex, local_id);
        }

        let mut local_edges = Vec::with_capacity(edges.len());
        let mut adjacency = vec![Vec::new(); vertices.len()];
        let mut edge_lookup = BTreeMap::new();

        for [left, right] in edges {
            let left_local = global_to_local[&left];
            let right_local = global_to_local[&right];
            let edge_id = local_edges.len();
            local_edges.push([left_local, right_local]);
            adjacency[left_local].push(right_local);
            adjacency[right_local].push(left_local);
            edge_lookup.insert(normalize_local_edge(left_local, right_local), edge_id);
        }

        Self { vertices, edges: local_edges, adjacency, edge_lookup }
    }

    #[inline]
    fn order(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    fn cycle_rank(&self) -> usize {
        self.edge_count() - self.order() + 1
    }

    #[inline]
    fn edge_id(&self, left_local: usize, right_local: usize) -> usize {
        self.edge_lookup[&normalize_local_edge(left_local, right_local)]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OrthogonalCycle<NodeId: PositiveInteger> {
    cycle: Vec<NodeId>,
    edge_bits: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnionFind {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self { parents: (0..size).collect(), ranks: vec![0; size] }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parents[node] != node {
            let parent = self.parents[node];
            self.parents[node] = self.find(parent);
        }
        self.parents[node]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }

        match self.ranks[left_root].cmp(&self.ranks[right_root]) {
            core::cmp::Ordering::Less => self.parents[left_root] = right_root,
            core::cmp::Ordering::Greater => self.parents[right_root] = left_root,
            core::cmp::Ordering::Equal => {
                self.parents[right_root] = left_root;
                self.ranks[left_root] += 1;
            }
        }

        true
    }
}

/// Trait providing an exact minimum cycle basis for undirected simple graphs.
pub trait MinimumCycleBasis: UndirectedMonopartiteMonoplexGraph {
    /// Computes the minimum cycle basis of the graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph contains self-loops or if orthogonal-cycle
    /// construction unexpectedly fails to span a cyclic connected component.
    ///
    /// # Complexity
    ///
    /// The implementation runs independently on each connected component and
    /// uses the exact de Pina / Kavitha-style orthogonal-cycle construction for
    /// unweighted graphs.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # {
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D, UpperTriangularCSR2D},
    ///     naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(4)
    ///     .symbols((0..4).enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
    ///     GenericUndirectedMonopartiteEdgesBuilder::<
    ///         _,
    ///         UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    ///         SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    ///     >::default()
    ///     .expected_number_of_edges(5)
    ///     .expected_shape(4)
    ///     .edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)].into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// let result = graph.minimum_cycle_basis().unwrap();
    /// assert_eq!(result.cycle_rank(), 2);
    /// assert_eq!(result.total_weight(), 6);
    /// # }
    /// ```
    fn minimum_cycle_basis(
        &self,
    ) -> Result<MinimumCycleBasisResult<Self::NodeId>, MinimumCycleBasisError>
    where
        Self: Sized,
    {
        if self.has_self_loops() {
            return Err(MinimumCycleBasisError::SelfLoopsUnsupported);
        }

        let Ok(components) = self.connected_components() else {
            unreachable!("usize component markers cannot overflow for an allocated graph")
        };

        let mut basis = Vec::new();

        let number_of_components: usize = components.number_of_components();
        for component_id in 0..number_of_components {
            let mut vertices = components.node_ids_of_component(component_id).collect::<Vec<_>>();
            if vertices.len() < 3 {
                continue;
            }
            vertices.sort_unstable();

            let edges = collect_component_edges(self, &vertices);
            if edges.len() < vertices.len() {
                continue;
            }

            let component = LocalGraph::from_component(vertices, edges);
            basis.extend(minimum_cycle_basis_for_component(&component)?);
        }

        basis.sort_unstable_by(|left, right| {
            left.len().cmp(&right.len()).then_with(|| left.cmp(right))
        });
        let total_weight = basis.iter().map(Vec::len).sum::<usize>();
        let cycle_rank = basis.len();

        Ok(MinimumCycleBasisResult { minimum_cycle_basis: basis, cycle_rank, total_weight })
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> MinimumCycleBasis for G {}

fn collect_component_edges<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
    vertices: &[G::NodeId],
) -> Vec<[G::NodeId; 2]> {
    let vertex_set = vertices.iter().copied().collect::<BTreeSet<_>>();
    let mut edges = Vec::new();

    for &left in vertices {
        for right in graph.neighbors(left) {
            if left < right && vertex_set.contains(&right) {
                edges.push([left, right]);
            }
        }
    }

    edges.sort_unstable();
    edges
}

fn minimum_cycle_basis_for_component<NodeId: PositiveInteger>(
    component: &LocalGraph<NodeId>,
) -> Result<Vec<Vec<NodeId>>, MinimumCycleBasisError> {
    let rank = component.cycle_rank();
    if rank == 0 {
        return Ok(Vec::new());
    }

    let tree_edges = kruskal_spanning_tree_flags(component);
    let chords = tree_edges
        .iter()
        .enumerate()
        .filter_map(|(edge_id, &in_tree)| (!in_tree).then_some(edge_id))
        .collect::<Vec<_>>();
    let mut orthogonal_vectors = chords
        .iter()
        .copied()
        .map(|edge_id| singleton_edge_vector(component.edge_count(), edge_id))
        .collect::<Vec<_>>();

    let mut basis = Vec::with_capacity(rank);
    while let Some(base) = orthogonal_vectors.pop() {
        let cycle = minimum_orthogonal_cycle(component, &base)?;
        basis.push(cycle.cycle);

        for orthogonal in &mut orthogonal_vectors {
            if odd_intersection(orthogonal, &cycle.edge_bits) {
                xor_assign(orthogonal, &base);
            }
        }
    }

    if basis.len() == rank {
        Ok(basis)
    } else {
        Err(MinimumCycleBasisError::CandidateSetIncomplete)
    }
}

fn kruskal_spanning_tree_flags<NodeId: PositiveInteger>(
    component: &LocalGraph<NodeId>,
) -> Vec<bool> {
    let mut union_find = UnionFind::new(component.order());
    let mut tree_edges = vec![false; component.edge_count()];

    for (edge_id, [left, right]) in component.edges.iter().copied().enumerate() {
        if union_find.union(left, right) {
            tree_edges[edge_id] = true;
        }
    }

    tree_edges
}

fn minimum_orthogonal_cycle<NodeId: PositiveInteger>(
    component: &LocalGraph<NodeId>,
    orthogonal: &[u64],
) -> Result<OrthogonalCycle<NodeId>, MinimumCycleBasisError> {
    let lifted = build_lifted_graph(component, orthogonal);
    let plane_size = component.order();

    let mut best_distance = None;
    let mut best_meeting = None;
    let mut best_parents0 = Vec::new();
    let mut best_parents1 = Vec::new();
    let mut best_source = 0usize;
    let mut best_target = 0usize;
    for start in 0..plane_size {
        let source = start;
        let target = plane_size + start;
        if let Some((distance, meeting, parents0, parents1)) =
            bidirectional_shortest_path(&lifted, source, target)
        {
            if best_distance.is_none_or(|best| distance < best) {
                best_distance = Some(distance);
                best_meeting = Some(meeting);
                best_parents0 = parents0;
                best_parents1 = parents1;
                best_source = source;
                best_target = target;
            }
        }
    }

    let lifted_path = reconstruct_bidirectional_path(
        best_meeting.ok_or(MinimumCycleBasisError::CandidateSetIncomplete)?,
        &best_parents0,
        &best_parents1,
        best_source,
        best_target,
    );
    let path = lifted_path
        .into_iter()
        .map(|node| if node < plane_size { node } else { node - plane_size })
        .collect::<Vec<_>>();
    let cycle_edges = collapse_lifted_path_edges(&path);
    if cycle_edges.len() < 3 {
        return Err(MinimumCycleBasisError::CandidateSetIncomplete);
    }

    let mut edge_bits = vec![0_u64; component.edge_count().div_ceil(64)];
    for &[left, right] in &cycle_edges {
        set_bit(&mut edge_bits, component.edge_id(left, right));
    }

    let cycle = normalize_cycle(
        cycle_edges.into_iter().map(|[_, right]| component.vertices[right]).collect::<Vec<_>>(),
    );
    Ok(OrthogonalCycle { cycle, edge_bits })
}

fn build_lifted_graph<NodeId: PositiveInteger>(
    component: &LocalGraph<NodeId>,
    orthogonal: &[u64],
) -> Vec<Vec<usize>> {
    let plane_size = component.order();
    let mut lifted = vec![Vec::new(); plane_size * 2];

    for (edge_id, [left, right]) in component.edges.iter().copied().enumerate() {
        if is_bit_set(orthogonal, edge_id) {
            add_undirected_edge(&mut lifted, left, plane_size + right);
            add_undirected_edge(&mut lifted, plane_size + left, right);
        } else {
            add_undirected_edge(&mut lifted, left, right);
            add_undirected_edge(&mut lifted, plane_size + left, plane_size + right);
        }
    }

    lifted
}

fn add_undirected_edge(adjacency: &mut [Vec<usize>], left: usize, right: usize) {
    adjacency[left].push(right);
    adjacency[right].push(left);
}

fn bidirectional_shortest_path(
    adjacency: &[Vec<usize>],
    source: usize,
    target: usize,
) -> Option<MeetingState> {
    if source == target {
        return Some((0, source, vec![None; adjacency.len()], vec![None; adjacency.len()]));
    }

    let node_count = adjacency.len();
    let mut dists0 = vec![None; node_count];
    let mut dists1 = vec![None; node_count];
    let mut seen0 = vec![None; node_count];
    let mut seen1 = vec![None; node_count];
    let mut parents0 = vec![None; node_count];
    let mut parents1 = vec![None; node_count];
    let mut fringe0 = BinaryHeap::new();
    let mut fringe1 = BinaryHeap::new();
    let mut counter = 0usize;

    seen0[source] = Some(0);
    seen1[target] = Some(0);
    fringe0.push(Reverse((0usize, counter, source)));
    counter += 1;
    fringe1.push(Reverse((0usize, counter, target)));
    counter += 1;

    let mut final_distance = None;
    let mut final_meeting = None;
    let mut direction = 1usize;

    while !fringe0.is_empty() && !fringe1.is_empty() {
        direction = 1 - direction;

        if direction == 0 {
            let Some((distance, node)) = pop_next(&mut fringe0, &dists0) else {
                continue;
            };
            dists0[node] = Some(distance);

            if let Some(other_distance) = dists1[node] {
                let total_distance = distance + other_distance;
                if final_distance.is_none_or(|best| total_distance < best) {
                    final_distance = Some(total_distance);
                    final_meeting = Some(node);
                }
                return final_distance
                    .map(|length| (length, final_meeting.unwrap_or(node), parents0, parents1));
            }

            for &neighbor in &adjacency[node] {
                let next_distance = distance + 1;
                debug_assert!(dists0[neighbor].is_none_or(|seen| next_distance >= seen));

                if dists0[neighbor].is_none()
                    && seen0[neighbor].is_none_or(|seen| next_distance < seen)
                {
                    seen0[neighbor] = Some(next_distance);
                    parents0[neighbor] = Some(node);
                    fringe0.push(Reverse((next_distance, counter, neighbor)));
                    counter += 1;

                    if let (Some(left_distance), Some(right_distance)) =
                        (seen0[neighbor], seen1[neighbor])
                    {
                        let total_distance = left_distance + right_distance;
                        if final_distance.is_none_or(|best| total_distance < best) {
                            final_distance = Some(total_distance);
                            final_meeting = Some(neighbor);
                        }
                    }
                }
            }
        } else {
            let Some((distance, node)) = pop_next(&mut fringe1, &dists1) else {
                continue;
            };
            dists1[node] = Some(distance);

            if let Some(other_distance) = dists0[node] {
                let total_distance = distance + other_distance;
                if final_distance.is_none_or(|best| total_distance < best) {
                    final_distance = Some(total_distance);
                    final_meeting = Some(node);
                }
                return final_distance
                    .map(|length| (length, final_meeting.unwrap_or(node), parents0, parents1));
            }

            for &neighbor in &adjacency[node] {
                let next_distance = distance + 1;
                debug_assert!(dists1[neighbor].is_none_or(|seen| next_distance >= seen));

                if dists1[neighbor].is_none()
                    && seen1[neighbor].is_none_or(|seen| next_distance < seen)
                {
                    seen1[neighbor] = Some(next_distance);
                    parents1[neighbor] = Some(node);
                    fringe1.push(Reverse((next_distance, counter, neighbor)));
                    counter += 1;

                    if let (Some(left_distance), Some(right_distance)) =
                        (seen0[neighbor], seen1[neighbor])
                    {
                        let total_distance = left_distance + right_distance;
                        if final_distance.is_none_or(|best| total_distance < best) {
                            final_distance = Some(total_distance);
                            final_meeting = Some(neighbor);
                        }
                    }
                }
            }
        }
    }

    None
}

fn reconstruct_bidirectional_path(
    meeting: usize,
    parents0: &[Option<usize>],
    parents1: &[Option<usize>],
    source: usize,
    target: usize,
) -> Vec<usize> {
    let mut left = Vec::new();
    let mut current = Some(meeting);
    while let Some(node) = current {
        left.push(node);
        if node == source {
            break;
        }
        current = parents0[node];
    }
    left.reverse();

    let mut right = Vec::new();
    let mut current = parents1[meeting];
    while let Some(node) = current {
        right.push(node);
        if node == target {
            break;
        }
        current = parents1[node];
    }

    left.extend(right);
    left
}

fn pop_next(
    fringe: &mut BinaryHeap<Reverse<(usize, usize, usize)>>,
    finalized: &[Option<usize>],
) -> Option<(usize, usize)> {
    while let Some(Reverse((distance, _, node))) = fringe.pop() {
        if finalized[node].is_none() {
            return Some((distance, node));
        }
    }
    None
}

fn collapse_lifted_path_edges(path: &[usize]) -> Vec<[usize; 2]> {
    let mut edge_set = BTreeSet::new();

    for pair in path.windows(2) {
        let edge = [pair[0], pair[1]];
        let reverse = [pair[1], pair[0]];
        if !edge_set.remove(&edge) && !edge_set.remove(&reverse) {
            edge_set.insert(edge);
        }
    }

    let mut cycle_edges = Vec::new();
    for pair in path.windows(2) {
        let edge = [pair[0], pair[1]];
        let reverse = [pair[1], pair[0]];
        if edge_set.remove(&edge) {
            cycle_edges.push(edge);
        } else if edge_set.remove(&reverse) {
            cycle_edges.push(reverse);
        }
    }

    cycle_edges
}

fn singleton_edge_vector(edge_count: usize, edge_id: usize) -> Vec<u64> {
    let mut bits = vec![0_u64; edge_count.div_ceil(64)];
    set_bit(&mut bits, edge_id);
    bits
}

fn odd_intersection(left: &[u64], right: &[u64]) -> bool {
    let mut parity = false;
    for (&left_word, &right_word) in left.iter().zip(right.iter()) {
        parity ^= ((left_word & right_word).count_ones() & 1) == 1;
    }
    parity
}

fn normalize_cycle<NodeId: PositiveInteger>(mut cycle: Vec<NodeId>) -> Vec<NodeId> {
    if cycle.is_empty() {
        return cycle;
    }

    let mut smallest_position = 0usize;
    for (position, node) in cycle.iter().enumerate().skip(1) {
        if node < &cycle[smallest_position] {
            smallest_position = position;
        }
    }
    cycle.rotate_left(smallest_position);

    if cycle.len() > 2 && cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }

    cycle
}

#[inline]
fn normalize_local_edge(left: usize, right: usize) -> [usize; 2] {
    if left <= right { [left, right] } else { [right, left] }
}

#[inline]
fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / 64] |= 1_u64 << (bit % 64);
}

#[inline]
fn is_bit_set(bits: &[u64], bit: usize) -> bool {
    ((bits[bit / 64] >> (bit % 64)) & 1) == 1
}

#[inline]
fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, right_word) in left.iter_mut().zip(right.iter()) {
        *left_word ^= *right_word;
    }
}
