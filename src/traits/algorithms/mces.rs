//! Maximum Common Edge Subgraph (MCES) builder.
//!
//! Composes the full MCES pipeline: line graph construction, modular product,
//! partition-aware maximum clique search, clique ranking, and similarity
//! computation.
//!
//! Uses a typestate builder pattern with zero-cost defaults. All closures and
//! rankers are monomorphized at compile time.
//!
//! # Example
//!
//! ```ignore
//! let result = McesBuilder::new(&g1, &g2)
//!     .compute_unlabeled();
//! let similarity = result.johnson_similarity();
//! ```

mod connected_tree_lower_bound;

use alloc::{collections::BTreeMap, vec::Vec};

use connected_tree_lower_bound::connected_tree_lower_bound;
use num_traits::AsPrimitive;

use super::{
    clique_ranking::{
        ChainedRanker, CliqueInfo, CliqueRanker, CliqueRankerExt, EagerCliqueInfo,
        FragmentCountRanker, LargestFragmentMetric, LargestFragmentMetricRanker, MatchedEdgePair,
    },
    graph_similarities::GraphSimilarities,
    labeled_line_graph::LabeledLineGraph,
    line_graph::LineGraph,
    maximum_clique::{
        PartitionInfo, PartitionSide, SearchOutcome, all_best_search,
        choose_partition_side_by_atom_counts, generic_search, greedy_lower_bound, partial_search,
        partial_search_u32_with_bounds, partial_u32_best_size_with_budget,
    },
    modular_product::{ModularProduct, ModularProductGraph},
    node_ordering::{DegreeScorer, DescendingScoreSorter, NodeSorter, PageRankScorer},
    weighted_assignment::Crouse,
};
use crate::{
    impls::{BitSquareMatrix, EdgeContexts, ValuedCSR2D},
    traits::{
        Edges, Matrix2D, MatrixMut, MonopartiteEdges, MonoplexMonopartiteGraph, PositiveInteger,
        SparseMatrix2D, SparseMatrixMut, SparseValuedMatrix2D, SquareMatrix, TypedNode,
        ValuedMatrix,
    },
};

// ============================================================================
// Default ZSTs
// ============================================================================

/// Default pair filter: accept all vertex pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AcceptAllPairs;

/// Default disambiguation: arbitrary orientation (always `true`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArbitraryDisambiguate;

/// Default edge-context constraint: accept all bond pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AcceptAllEdgeContexts;

// ============================================================================
// Traits for builder parameters
// ============================================================================

/// Pair filter for modular product vertex pair selection.
pub trait McesPairFilter {
    /// Returns `true` if the pair `(i, j)` should be included in the product.
    fn filter(&mut self, i: usize, j: usize) -> bool;
}

impl McesPairFilter for AcceptAllPairs {
    #[inline]
    fn filter(&mut self, _i: usize, _j: usize) -> bool {
        true
    }
}

/// Wrapper for user-provided pair filter closures.
pub struct CustomPairFilter<F>(pub F);

impl<F: FnMut(usize, usize) -> bool> McesPairFilter for CustomPairFilter<F> {
    #[inline]
    fn filter(&mut self, i: usize, j: usize) -> bool {
        (self.0)(i, j)
    }
}

/// Additional constraint over original edge pairs for labeled MCES.
pub trait McesEdgeContexts {
    /// Validates that the stored rows match the original graph edge counts.
    fn validate(&self, first_edges: usize, second_edges: usize);

    /// Returns `true` if the original edge pair `(i, j)` is compatible.
    fn compatible(&self, i: usize, j: usize) -> bool;
}

impl McesEdgeContexts for AcceptAllEdgeContexts {
    #[inline]
    fn validate(&self, _first_edges: usize, _second_edges: usize) {}

    #[inline]
    fn compatible(&self, _i: usize, _j: usize) -> bool {
        true
    }
}

/// Wrapper for precomputed edge-context memberships.
pub struct ConfiguredEdgeContexts<'g, Signature, SparseIndex = usize> {
    first: &'g EdgeContexts<Signature, SparseIndex>,
    second: &'g EdgeContexts<Signature, SparseIndex>,
}

impl<Signature, SparseIndex> McesEdgeContexts for ConfiguredEdgeContexts<'_, Signature, SparseIndex>
where
    Signature: PartialEq,
    SparseIndex: PositiveInteger,
{
    fn validate(&self, first_edges: usize, second_edges: usize) {
        assert_eq!(
            self.first.len(),
            first_edges,
            "edge contexts for the first graph must have one row per original edge",
        );
        assert_eq!(
            self.second.len(),
            second_edges,
            "edge contexts for the second graph must have one row per original edge",
        );
    }

    #[inline]
    fn compatible(&self, i: usize, j: usize) -> bool {
        self.first.compatible_with(i, self.second, j)
    }
}

/// Disambiguation strategy for isolated edge vertex matching.
pub trait McesDisambiguate<N> {
    /// Returns `true` for mapping `a↔c, b↔d`; `false` for `a↔d, b↔c`.
    fn disambiguate(&mut self, a: N, b: N, c: N, d: N) -> bool;
}

impl<N> McesDisambiguate<N> for ArbitraryDisambiguate {
    #[inline]
    fn disambiguate(&mut self, _a: N, _b: N, _c: N, _d: N) -> bool {
        true
    }
}

/// Wrapper for user-provided disambiguation closures.
pub struct CustomDisambiguate<F>(pub F);

impl<N, F: FnMut(N, N, N, N) -> bool> McesDisambiguate<N> for CustomDisambiguate<F> {
    #[inline]
    fn disambiguate(&mut self, a: N, b: N, c: N, d: N) -> bool {
        (self.0)(a, b, c, d)
    }
}

/// Edge comparator for labeled modular product.
pub trait McesEdgeComparator<V1, V2> {
    /// Returns `true` if the edge values are compatible.
    fn compare(&self, a: Option<V1>, b: Option<V2>) -> bool;
}

/// Default edge comparator: strict `PartialEq` equality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StrictEqualityComparator;

impl<V: PartialEq> McesEdgeComparator<V, V> for StrictEqualityComparator {
    #[inline]
    fn compare(&self, a: Option<V>, b: Option<V>) -> bool {
        a == b
    }
}

/// Wrapper for user-provided edge comparator closures.
pub struct CustomEdgeComparator<F>(pub F);

impl<V1, V2, F: Fn(Option<V1>, Option<V2>) -> bool> McesEdgeComparator<V1, V2>
    for CustomEdgeComparator<F>
{
    #[inline]
    fn compare(&self, a: Option<V1>, b: Option<V2>) -> bool {
        (self.0)(a, b)
    }
}

// ============================================================================
// Delta-Y detection helpers
// ============================================================================

/// Computes the sorted degree sequence of an edge-induced subgraph.
fn mces_degree_sequence<N: Copy + AsPrimitive<usize>>(
    num_vertices: usize,
    edges: &[(N, N)],
) -> Vec<usize> {
    let mut counts = vec![0usize; num_vertices];
    for &(u, v) in edges {
        counts[u.as_()] += 1;
        counts[v.as_()] += 1;
    }
    let mut seq: Vec<usize> = counts.into_iter().filter(|&d| d > 0).collect();
    seq.sort_unstable();
    seq
}

/// Returns `true` if the two matched edge-induced subgraphs exhibit a Delta-Y
/// exchange (different degree sequences in the original graphs).
fn has_delta_y_on_edges<N: Copy + AsPrimitive<usize>>(
    first_edges: &[(N, N)],
    second_edges: &[(N, N)],
    num_vertices_first: usize,
    num_vertices_second: usize,
) -> bool {
    mces_degree_sequence(num_vertices_first, first_edges)
        != mces_degree_sequence(num_vertices_second, second_edges)
}

/// Returns `true` if the modular-product clique exhibits a Delta-Y exchange
/// when mapped back to the original graphs.
fn clique_has_delta_y<N: Copy + AsPrimitive<usize>>(
    clique: &[usize],
    vertex_pairs: &[(usize, usize)],
    first_edge_map: &[(N, N)],
    second_edge_map: &[(N, N)],
    num_vertices_first: usize,
    num_vertices_second: usize,
) -> bool {
    let first_edges: Vec<(N, N)> =
        clique.iter().map(|&v| first_edge_map[vertex_pairs[v].0]).collect();
    let second_edges: Vec<(N, N)> =
        clique.iter().map(|&v| second_edge_map[vertex_pairs[v].1]).collect();
    has_delta_y_on_edges(&first_edges, &second_edges, num_vertices_first, num_vertices_second)
}

// ============================================================================
// Pre-screening (Tier 1)
// ============================================================================

/// Screening estimate: upper bound on matched vertices and edges.
struct ScreeningEstimate {
    /// Estimated matched vertices (sum of min counts per label type).
    vg1g2: usize,
    /// Estimated matched edges × 2 (sum of min degrees before dividing).
    eg1g2_times2: usize,
}

impl ScreeningEstimate {
    /// RASCAL-style similarity upper bound.
    #[allow(clippy::cast_precision_loss)]
    fn similarity(&self, v1: usize, e1: usize, v2: usize, e2: usize) -> f64 {
        let denom = (v1 + e1) * (v2 + e2);
        if denom == 0 {
            return 1.0;
        }
        let num = self.vg1g2 + self.eg1g2_times2 / 2;
        (num * num) as f64 / denom as f64
    }

    /// Myopic-style distance lower bound.
    fn distance(&self, e1: usize, e2: usize) -> usize {
        // dist = E1 + E2 - 2*eg1g2, and eg1g2 = eg1g2_times2 / 2
        // so dist = E1 + E2 - eg1g2_times2
        (e1 + e2).saturating_sub(self.eg1g2_times2)
    }

    /// Returns `true` if the estimate is rejected by either threshold.
    fn is_rejected(
        &self,
        v1: usize,
        e1: usize,
        v2: usize,
        e2: usize,
        sim_threshold: Option<f64>,
        dist_threshold: Option<f64>,
    ) -> bool {
        if let Some(t) = sim_threshold
            && self.similarity(v1, e1, v2, e2) < t
        {
            return true;
        }
        if let Some(t) = dist_threshold {
            #[allow(clippy::cast_precision_loss)]
            if (self.distance(e1, e2) as f64) > t {
                return true;
            }
        }
        false
    }
}

/// Tier 1 screening: degree-sequence bound, O(V log V).
///
/// Groups vertices by label, sorts degrees descending within each group,
/// and greedily pairs them to estimate the maximum number of matchable
/// vertices and edges.
fn tier1_screening<L: Ord>(
    degrees_by_label_first: &BTreeMap<L, Vec<usize>>,
    degrees_by_label_second: &BTreeMap<L, Vec<usize>>,
) -> ScreeningEstimate {
    let mut vg1g2 = 0usize;
    let mut eg1g2_times2 = 0usize;

    for (label, degs1) in degrees_by_label_first {
        if let Some(degs2) = degrees_by_label_second.get(label) {
            vg1g2 += degs1.len().min(degs2.len());
            for (d1, d2) in degs1.iter().zip(degs2.iter()) {
                eg1g2_times2 += (*d1).min(*d2);
            }
        }
    }

    ScreeningEstimate { vg1g2, eg1g2_times2 }
}

/// Extracts degree sequences grouped by vertex label, sorted descending.
///
/// For unlabeled graphs, use `()` as the label for all vertices.
fn extract_degree_groups<G, L, F>(graph: &G, vertex_label: F) -> BTreeMap<L, Vec<usize>>
where
    G: crate::traits::MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    L: Ord,
    F: Fn(G::NodeId) -> L,
{
    let mut groups: BTreeMap<L, Vec<usize>> = BTreeMap::new();
    for node_id in graph.node_ids() {
        let label = vertex_label(node_id);
        let degree = graph.out_degree(node_id).as_();
        groups.entry(label).or_default().push(degree);
    }
    // Sort each group descending.
    for degs in groups.values_mut() {
        degs.sort_unstable_by(|a, b| b.cmp(a));
    }
    groups
}

/// Extracts per-label degree sequences together with atom indices.
fn extract_degree_sequences<G, L, F>(graph: &G, vertex_label: F) -> BTreeMap<L, Vec<(usize, usize)>>
where
    G: crate::traits::MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    L: Ord,
    F: Fn(G::NodeId) -> L,
{
    let mut groups: BTreeMap<L, Vec<(usize, usize)>> = BTreeMap::new();
    for node_id in graph.node_ids() {
        let label = vertex_label(node_id);
        let degree = graph.out_degree(node_id).as_();
        groups.entry(label).or_default().push((degree, node_id.as_()));
    }
    for seq in groups.values_mut() {
        seq.sort_unstable_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    }
    groups
}

fn build_incident_bond_labels<N: Copy + AsPrimitive<usize>>(
    num_nodes: usize,
    edge_map: &[(N, N)],
    bond_label_indices: &[usize],
) -> Vec<Vec<usize>> {
    let mut incident = vec![Vec::new(); num_nodes];
    for (&(src, dst), &label_index) in edge_map.iter().zip(bond_label_indices) {
        incident[src.as_()].push(label_index);
        incident[dst.as_()].push(label_index);
    }
    for labels in &mut incident {
        labels.sort_unstable();
    }
    incident
}

fn incident_label_overlap(first: &[usize], second: &[usize]) -> usize {
    let mut first_index = 0usize;
    let mut second_index = 0usize;
    let mut overlap = 0usize;

    while first_index < first.len() && second_index < second.len() {
        match first[first_index].cmp(&second[second_index]) {
            core::cmp::Ordering::Less => {
                first_index += 1;
            }
            core::cmp::Ordering::Greater => {
                second_index += 1;
            }
            core::cmp::Ordering::Equal => {
                let label = first[first_index];
                let first_start = first_index;
                while first_index < first.len() && first[first_index] == label {
                    first_index += 1;
                }
                let second_start = second_index;
                while second_index < second.len() && second[second_index] == label {
                    second_index += 1;
                }
                overlap += (first_index - first_start).min(second_index - second_start);
            }
        }
    }

    overlap
}

#[allow(clippy::cast_precision_loss)]
fn assignment_score_via_crouse(
    first_atoms: &[(usize, usize)],
    second_atoms: &[(usize, usize)],
    first_incident_bond_labels: &[Vec<usize>],
    second_incident_bond_labels: &[Vec<usize>],
) -> usize {
    if first_atoms.is_empty() || second_atoms.is_empty() {
        return 0;
    }

    let rows = first_atoms.len();
    let columns = second_atoms.len();
    let mut scores = vec![0usize; rows * columns];
    let mut max_score = 0usize;

    for (row_index, &(_, first_atom_index)) in first_atoms.iter().enumerate() {
        for (column_index, &(_, second_atom_index)) in second_atoms.iter().enumerate() {
            let score = incident_label_overlap(
                &first_incident_bond_labels[first_atom_index],
                &second_incident_bond_labels[second_atom_index],
            );
            scores[row_index * columns + column_index] = score;
            max_score = max_score.max(score);
        }
    }

    let mut matrix: ValuedCSR2D<usize, usize, usize, f64> =
        SparseMatrixMut::with_sparse_shaped_capacity((rows, columns), rows * columns);
    let max_real_cost = (max_score + 1) as f64;

    for row_index in 0..rows {
        for column_index in 0..columns {
            let score = scores[row_index * columns + column_index];
            let cost = max_real_cost - score as f64;
            MatrixMut::add(&mut matrix, (row_index, column_index, cost))
                .expect("tier2 assignment matrix must be built in sorted row-major order");
        }
    }

    let non_edge_cost = max_real_cost + 1.0;
    let max_cost = non_edge_cost + 1.0;
    matrix
        .crouse(non_edge_cost, max_cost)
        .expect("dense tier2 assignment must be feasible")
        .into_iter()
        .map(|(row_index, column_index)| scores[row_index * columns + column_index])
        .sum()
}

fn tier2_screening<L: Ord>(
    degrees_by_label_first: &BTreeMap<L, Vec<(usize, usize)>>,
    degrees_by_label_second: &BTreeMap<L, Vec<(usize, usize)>>,
    first_incident_bond_labels: &[Vec<usize>],
    second_incident_bond_labels: &[Vec<usize>],
) -> ScreeningEstimate {
    let mut vg1g2 = 0usize;
    let mut eg1g2_times2 = 0usize;

    for (label, atoms1) in degrees_by_label_first {
        if let Some(atoms2) = degrees_by_label_second.get(label) {
            vg1g2 += atoms1.len().min(atoms2.len());
            eg1g2_times2 += assignment_score_via_crouse(
                atoms1,
                atoms2,
                first_incident_bond_labels,
                second_incident_bond_labels,
            );
        }
    }

    ScreeningEstimate { vg1g2, eg1g2_times2 }
}

type BondLabel<G> = (
    <<G as crate::traits::MonopartiteGraph>::NodeSymbol as TypedNode>::NodeType,
    Option<
        <<<G as MonoplexMonopartiteGraph>::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix as ValuedMatrix>::Value,
    >,
    <<G as crate::traits::MonopartiteGraph>::NodeSymbol as TypedNode>::NodeType,
);

/// Computes the current bond label for each original graph edge.
///
/// The label is the canonical pair of endpoint node types plus an optional
/// original edge value, matching the intrinsic bond-pair compatibility used by
/// labeled MCES. When `ignore_edge_values` is enabled, the edge-value slot is
/// collapsed to `None`.
fn compute_bond_labels<G>(
    graph: &G,
    edge_map: &[(G::NodeId, G::NodeId)],
    ignore_edge_values: bool,
) -> Vec<BondLabel<G>>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    G::NodeSymbol: TypedNode,
    <G::NodeSymbol as TypedNode>::NodeType: Copy + Ord,
    <G::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix:
        SparseValuedMatrix2D<RowIndex = G::NodeId, ColumnIndex = G::NodeId>,
    <<G::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix as ValuedMatrix>::Value:
        Copy + PartialEq,
{
    let node_types: Vec<<G::NodeSymbol as TypedNode>::NodeType> =
        graph.nodes().map(|sym| sym.node_type()).collect();
    edge_map
        .iter()
        .map(|&(src, dst)| {
            let t1 = node_types[src.as_()];
            let t2 = node_types[dst.as_()];
            let edge_value = if ignore_edge_values {
                None
            } else {
                Some(
                    graph
                        .edges()
                        .matrix()
                        .sparse_value_at(src, dst)
                        .expect("line graph edge_map must refer to an existing original edge"),
                )
            };
            if t1 <= t2 { (t1, edge_value, t2) } else { (t2, edge_value, t1) }
        })
        .collect()
}

/// Maps the shared label universe from two graphs to dense indices.
fn intern_shared_labels<L: PartialEq + Copy>(
    first: &[L],
    second: &[L],
) -> (Vec<usize>, Vec<usize>, usize) {
    let mut all_labels: Vec<L> = Vec::new();
    for label in first.iter().chain(second.iter()).copied() {
        if !all_labels.contains(&label) {
            all_labels.push(label);
        }
    }
    let num_labels = all_labels.len().max(1);

    let first_indices = first
        .iter()
        .map(|label| all_labels.iter().position(|candidate| candidate == label).unwrap())
        .collect();
    let second_indices = second
        .iter()
        .map(|label| all_labels.iter().position(|candidate| candidate == label).unwrap())
        .collect();

    (first_indices, second_indices, num_labels)
}

/// Search mode for the clique stage of MCES.
///
/// `PartialEnumeration` mirrors RDKit's default partitioned behavior more
/// closely: it keeps strict tie-pruning, retains equal-size accepted maxima
/// encountered during that search, and ranks the retained set afterward.
///
/// `AllBest` enumerates all accepted tied maximum cliques before ranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McesSearchMode {
    /// Strict pruning plus retained tied-best cliques for later ranking.
    ///
    /// On the non-partitioned fallback path, this currently degenerates to the
    /// legacy single accepted maximum behavior.
    PartialEnumeration,
    /// Enumerate all accepted tied maximum cliques.
    AllBest,
}

/// Initial ordering policy for modular-product vertices before clique search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialProductVertexOrdering {
    /// Leave product vertices in construction order.
    None,
    /// Order product vertices lexicographically by intrinsic edge signatures
    /// computed independently in the two input graphs.
    ///
    /// This is a source-graph ordering, not a product-graph centrality score.
    EdgeSignature,
    /// Order product vertices by Weisfeiler-Lehman colors of the line-graph
    /// vertices, seeded by intrinsic source-edge signatures.
    ///
    /// This is still a source-edge ordering. It uses line-graph refinement to
    /// propagate a larger edge neighborhood context before sorting product
    /// vertices lexicographically by the pair of refined edge colors.
    LineGraphWL,
    /// Order product vertices by descending degree on the modular product.
    Degree,
    /// Order product vertices by descending PageRank on the modular product.
    PageRank,
}

fn endpoint_degrees_from_edge_map<N: Copy + AsPrimitive<usize>>(
    edge_map: &[(N, N)],
    num_vertices: usize,
) -> Vec<usize> {
    let mut degrees = vec![0usize; num_vertices];
    for &(src, dst) in edge_map {
        degrees[src.as_()] += 1;
        degrees[dst.as_()] += 1;
    }
    degrees
}

fn incident_edges_from_edge_map<N: Copy + AsPrimitive<usize>>(
    edge_map: &[(N, N)],
    num_vertices: usize,
) -> Vec<Vec<usize>> {
    let mut incident = vec![Vec::new(); num_vertices];
    for (edge_index, &(src, dst)) in edge_map.iter().enumerate() {
        incident[src.as_()].push(edge_index);
        incident[dst.as_()].push(edge_index);
    }
    incident
}

fn sorted_neighbor_labels<L: Copy + Ord>(
    incident_edges: &[usize],
    edge_index: usize,
    base_labels: &[L],
) -> Vec<L> {
    let mut labels: Vec<L> = incident_edges
        .iter()
        .copied()
        .filter(|&neighbor_edge_index| neighbor_edge_index != edge_index)
        .map(|neighbor_edge_index| base_labels[neighbor_edge_index])
        .collect();
    labels.sort_unstable();
    labels
}

type EdgeEndpointProfile<L> = (usize, Vec<L>);
type CanonicalEdgeSignature<L> = (L, EdgeEndpointProfile<L>, EdgeEndpointProfile<L>);
type UnlabeledBaseLabel = (usize, usize);
type UnlabeledEdgeSignature = CanonicalEdgeSignature<UnlabeledBaseLabel>;
type LabeledEdgeSignature = CanonicalEdgeSignature<usize>;
type UnlabeledWlEdgeSignature = (usize, UnlabeledEdgeSignature);
type LabeledWlEdgeSignature = (usize, LabeledEdgeSignature);

fn edge_signatures_from_base_labels<N, L>(
    edge_map: &[(N, N)],
    num_vertices: usize,
    base_labels: &[L],
) -> Vec<CanonicalEdgeSignature<L>>
where
    N: Copy + AsPrimitive<usize>,
    L: Copy + Ord,
{
    let degrees = endpoint_degrees_from_edge_map(edge_map, num_vertices);
    let incident = incident_edges_from_edge_map(edge_map, num_vertices);

    edge_map
        .iter()
        .enumerate()
        .map(|(edge_index, &(src, dst))| {
            let left_profile = (
                degrees[src.as_()],
                sorted_neighbor_labels(&incident[src.as_()], edge_index, base_labels),
            );
            let right_profile = (
                degrees[dst.as_()],
                sorted_neighbor_labels(&incident[dst.as_()], edge_index, base_labels),
            );
            let (lo, hi) = if left_profile <= right_profile {
                (left_profile, right_profile)
            } else {
                (right_profile, left_profile)
            };
            (base_labels[edge_index], lo, hi)
        })
        .collect()
}

fn unlabeled_edge_signatures<N: Copy + AsPrimitive<usize>>(
    edge_map: &[(N, N)],
    num_vertices: usize,
) -> Vec<UnlabeledEdgeSignature> {
    let degrees = endpoint_degrees_from_edge_map(edge_map, num_vertices);
    let base_labels: Vec<UnlabeledBaseLabel> = edge_map
        .iter()
        .map(|&(src, dst)| {
            let left = degrees[src.as_()];
            let right = degrees[dst.as_()];
            if left <= right { (left, right) } else { (right, left) }
        })
        .collect();
    edge_signatures_from_base_labels(edge_map, num_vertices, &base_labels)
}

fn labeled_edge_signatures<N: Copy + AsPrimitive<usize>>(
    edge_map: &[(N, N)],
    num_vertices: usize,
    label_indices: &[usize],
) -> Vec<LabeledEdgeSignature> {
    edge_signatures_from_base_labels(edge_map, num_vertices, label_indices)
}

fn reorder_product_by_edge_signatures<S1: Ord, S2: Ord>(
    matrix: &BitSquareMatrix,
    vertex_pairs: &[(usize, usize)],
    first_edge_signatures: &[S1],
    second_edge_signatures: &[S2],
) -> (BitSquareMatrix, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..vertex_pairs.len()).collect();
    order.sort_unstable_by(|&left_index, &right_index| {
        let (left_first, left_second) = vertex_pairs[left_index];
        let (right_first, right_second) = vertex_pairs[right_index];
        first_edge_signatures[left_first]
            .cmp(&first_edge_signatures[right_first])
            .then(second_edge_signatures[left_second].cmp(&second_edge_signatures[right_second]))
            .then(left_first.cmp(&right_first))
            .then(left_second.cmp(&right_second))
    });

    let mut permuted = BitSquareMatrix::new(order.len());
    for new_left in 0..order.len() {
        for new_right in new_left + 1..order.len() {
            if matrix.has_entry(order[new_left], order[new_right]) {
                permuted.set_symmetric(new_left, new_right);
            }
        }
    }
    let permuted_pairs = order.into_iter().map(|old| vertex_pairs[old]).collect();
    (permuted, permuted_pairs)
}

fn dense_rank<T>(keys: &[T]) -> Vec<usize>
where
    T: Ord + Clone,
{
    let mut keyed_indices: Vec<(T, usize)> =
        keys.iter().cloned().enumerate().map(|(index, key)| (key, index)).collect();
    keyed_indices.sort_unstable_by(|(left_key, left_index), (right_key, right_index)| {
        left_key.cmp(right_key).then_with(|| left_index.cmp(right_index))
    });

    let mut colors = vec![0usize; keys.len()];
    let mut current_color = 0usize;

    for (offset, (_, index)) in keyed_indices.iter().enumerate() {
        if offset > 0 && keyed_indices[offset - 1].0 != keyed_indices[offset].0 {
            current_color += 1;
        }
        colors[*index] = current_color;
    }

    colors
}

fn wl_refine_with_edge_colors<M, SeedColor, EdgeColor, F>(
    graph: &M,
    seed_colors: &[SeedColor],
    mut edge_colors: F,
) -> Vec<usize>
where
    M: Matrix2D<RowIndex = usize, ColumnIndex = usize>
        + SparseMatrix2D<RowIndex = usize, ColumnIndex = usize>,
    SeedColor: Ord + Clone,
    EdgeColor: Ord + Clone,
    F: FnMut(usize, usize) -> EdgeColor,
{
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct RefinementKey<EdgeColor> {
        current_color: usize,
        neighborhood: Vec<(EdgeColor, usize)>,
    }

    let node_count = graph.number_of_rows();
    assert_eq!(
        seed_colors.len(),
        node_count,
        "seed colors must contain exactly one entry per dense node id",
    );

    let mut colors = dense_rank(seed_colors);
    loop {
        let mut keys = Vec::with_capacity(node_count);
        for node in 0..node_count {
            let mut neighborhood: Vec<(EdgeColor, usize)> = graph
                .sparse_row(node)
                .map(|neighbor| (edge_colors(node, neighbor), colors[neighbor]))
                .collect();
            neighborhood.sort_unstable();
            keys.push(RefinementKey { current_color: colors[node], neighborhood });
        }

        let next_colors = dense_rank(&keys);
        if next_colors == colors {
            return colors;
        }
        colors = next_colors;
    }
}

fn unlabeled_line_graph_wl_signatures<N, M>(
    line_graph: &M,
    edge_map: &[(N, N)],
    num_vertices: usize,
) -> Vec<UnlabeledWlEdgeSignature>
where
    N: Copy + AsPrimitive<usize>,
    M: Matrix2D<RowIndex = usize, ColumnIndex = usize>
        + SparseMatrix2D<RowIndex = usize, ColumnIndex = usize>,
{
    let base_signatures = unlabeled_edge_signatures(edge_map, num_vertices);
    let wl_colors = wl_refine_with_edge_colors(line_graph, &base_signatures, |_, _| ());
    wl_colors.into_iter().zip(base_signatures).collect()
}

fn labeled_line_graph_wl_signatures<N, M>(
    line_graph: &M,
    edge_map: &[(N, N)],
    num_vertices: usize,
    label_indices: &[usize],
) -> Vec<LabeledWlEdgeSignature>
where
    N: Copy + AsPrimitive<usize>,
    M: Matrix2D<RowIndex = usize, ColumnIndex = usize>
        + SparseMatrix2D<RowIndex = usize, ColumnIndex = usize>
        + SparseValuedMatrix2D<RowIndex = usize, ColumnIndex = usize>,
    M::Value: Ord + Clone,
{
    let base_signatures = labeled_edge_signatures(edge_map, num_vertices, label_indices);
    let wl_colors = wl_refine_with_edge_colors(line_graph, &base_signatures, |src, dst| {
        line_graph
            .sparse_value_at(src, dst)
            .expect("line graph row neighbor must correspond to an existing edge color")
    });
    wl_colors.into_iter().zip(base_signatures).collect()
}

fn reorder_product_for_search<N>(
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    _first_edge_map: &[(N, N)],
    _second_edge_map: &[(N, N)],
    ordering: InitialProductVertexOrdering,
) -> (BitSquareMatrix, Vec<(usize, usize)>)
where
    N: Copy + AsPrimitive<usize>,
{
    let order = match ordering {
        InitialProductVertexOrdering::None => return (matrix, vertex_pairs),
        InitialProductVertexOrdering::EdgeSignature => {
            panic!("edge-signature ordering requires source-graph edge signatures")
        }
        InitialProductVertexOrdering::LineGraphWL => {
            panic!("line-graph WL ordering requires source-graph line-graph refinement")
        }
        InitialProductVertexOrdering::Degree => {
            let graph = ModularProductGraph::new(matrix.clone(), vertex_pairs.clone());
            DescendingScoreSorter::new(DegreeScorer).sort_nodes(&graph)
        }
        InitialProductVertexOrdering::PageRank => {
            let graph = ModularProductGraph::new(matrix.clone(), vertex_pairs.clone());
            DescendingScoreSorter::new(PageRankScorer::default()).sort_nodes(&graph)
        }
    };

    let mut permuted = BitSquareMatrix::new(order.len());
    for new_left in 0..order.len() {
        for new_right in new_left + 1..order.len() {
            if matrix.has_entry(order[new_left], order[new_right]) {
                permuted.set_symmetric(new_left, new_right);
            }
        }
    }
    let permuted_pairs = order.into_iter().map(|old| vertex_pairs[old]).collect();
    (permuted, permuted_pairs)
}

struct SourceEdgeOrderingComputers<FEdge, FWl> {
    edge_signatures: FEdge,
    wl_signatures: FWl,
}

fn reorder_product_with_source_edge_ordering<N, EdgeSignature, WlSignature, FEdge, FWl>(
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    first_edge_map: &[(N, N)],
    second_edge_map: &[(N, N)],
    ordering: InitialProductVertexOrdering,
    source_orderings: SourceEdgeOrderingComputers<FEdge, FWl>,
) -> (BitSquareMatrix, Vec<(usize, usize)>)
where
    N: Copy + AsPrimitive<usize>,
    EdgeSignature: Ord,
    WlSignature: Ord,
    FEdge: FnOnce() -> (Vec<EdgeSignature>, Vec<EdgeSignature>),
    FWl: FnOnce() -> (Vec<WlSignature>, Vec<WlSignature>),
{
    let SourceEdgeOrderingComputers { edge_signatures, wl_signatures } = source_orderings;

    match ordering {
        InitialProductVertexOrdering::EdgeSignature => {
            let (first_edge_signatures, second_edge_signatures) = edge_signatures();
            reorder_product_by_edge_signatures(
                &matrix,
                &vertex_pairs,
                &first_edge_signatures,
                &second_edge_signatures,
            )
        }
        InitialProductVertexOrdering::LineGraphWL => {
            let (first_edge_signatures, second_edge_signatures) = wl_signatures();
            reorder_product_by_edge_signatures(
                &matrix,
                &vertex_pairs,
                &first_edge_signatures,
                &second_edge_signatures,
            )
        }
        _ => {
            reorder_product_for_search(
                matrix,
                vertex_pairs,
                first_edge_map,
                second_edge_map,
                ordering,
            )
        }
    }
}

fn accepted_cliques<F>(
    matrix: &BitSquareMatrix,
    search_mode: McesSearchMode,
    search_budget: Option<usize>,
    accept_clique: F,
) -> SearchOutcome
where
    F: FnMut(&[usize]) -> bool,
{
    let max_nodes = search_budget.unwrap_or(usize::MAX);
    match search_mode {
        McesSearchMode::PartialEnumeration => {
            let SearchOutcome { cliques, nodes, completed } =
                generic_search(matrix, false, max_nodes, accept_clique);
            // PartialEnumeration on the non-partition path historically returns
            // a single best clique (possibly empty), so keep that shape.
            SearchOutcome {
                cliques: vec![cliques.into_iter().next().unwrap_or_default()],
                nodes,
                completed,
            }
        }
        McesSearchMode::AllBest => generic_search(matrix, true, max_nodes, accept_clique),
    }
}

fn accepted_partitioned_cliques<F>(
    matrix: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    search_mode: McesSearchMode,
    search_budget: Option<usize>,
    mut accept_clique: F,
) -> SearchOutcome
where
    F: FnMut(&[usize]) -> bool,
{
    let max_nodes = search_budget.unwrap_or(usize::MAX);
    match search_mode {
        McesSearchMode::PartialEnumeration => {
            let best_size_seed =
                partial_best_size_seed(matrix, partition, initial_lower_bound, &mut accept_clique);

            partial_search_u32_with_bounds(
                matrix,
                partition,
                // Keep the ordinary non-empty baseline in the partition state,
                // but do not let stronger external lower bounds reshape the
                // partition search. The connected-tree lower bound should act
                // as an incumbent seed only for `PartialEnumeration`.
                usize::from(matrix.order() > 0),
                best_size_seed,
                max_nodes,
                &mut accept_clique,
            )
        }
        McesSearchMode::AllBest => {
            let initial_lower_bound =
                partial_search(matrix, partition, initial_lower_bound, &mut accept_clique)
                    .first()
                    .map_or(0, Vec::len);
            all_best_search(matrix, partition, initial_lower_bound, max_nodes, accept_clique)
        }
    }
}

const PARTIAL_GREEDY_DELTA_THRESHOLD: usize = 2;
const PARTIAL_SEED_DFS_BUDGET: usize = 5_000;

fn alternate_partition_info<'a>(partition: &PartitionInfo<'a>) -> PartitionInfo<'a> {
    let partition_side = match partition.partition_side {
        PartitionSide::First => PartitionSide::Second,
        PartitionSide::Second => PartitionSide::First,
    };
    PartitionInfo {
        pairs: partition.pairs,
        g1_labels: partition.g1_labels,
        g2_labels: partition.g2_labels,
        num_labels: partition.num_labels,
        partition_side,
    }
}

fn partition_info_with_side<'a>(
    partition: &PartitionInfo<'a>,
    partition_side: PartitionSide,
) -> PartitionInfo<'a> {
    PartitionInfo {
        pairs: partition.pairs,
        g1_labels: partition.g1_labels,
        g2_labels: partition.g2_labels,
        num_labels: partition.num_labels,
        partition_side,
    }
}

fn partial_initial_seed_size<F>(
    matrix: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: &mut F,
) -> (usize, usize, usize)
where
    F: FnMut(&[usize]) -> bool,
{
    let current_greedy =
        greedy_lower_bound(matrix, partition, initial_lower_bound, &mut *accept_clique);
    let alternate_partition = alternate_partition_info(partition);
    let alternate_greedy =
        greedy_lower_bound(matrix, &alternate_partition, initial_lower_bound, &mut *accept_clique);
    let initial_seed_size =
        if alternate_greedy >= current_greedy.saturating_add(PARTIAL_GREEDY_DELTA_THRESHOLD) {
            alternate_greedy
        } else {
            current_greedy
        };
    (initial_seed_size, current_greedy, alternate_greedy)
}

fn partial_best_size_seed<F>(
    matrix: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: &mut F,
) -> usize
where
    F: FnMut(&[usize]) -> bool,
{
    // Keep the shipped partition side and state lower bound unchanged. Only
    // improve the incumbent seed. The seed search runs on the more promising
    // side, but the real search still runs on the shipped order, side, and
    // state lower bound.
    let (initial_seed_size, current_greedy, alternate_greedy) =
        partial_initial_seed_size(matrix, partition, initial_lower_bound, &mut *accept_clique);
    let seed_side = if alternate_greedy > current_greedy {
        alternate_partition_info(partition).partition_side
    } else {
        partition.partition_side
    };
    let seed_partition = partition_info_with_side(partition, seed_side);
    let seed_best_size = partial_u32_best_size_with_budget(
        matrix,
        &seed_partition,
        initial_lower_bound,
        initial_seed_size.saturating_sub(1),
        PARTIAL_SEED_DFS_BUDGET,
        &mut *accept_clique,
    );

    initial_seed_size.max(seed_best_size).saturating_sub(1)
}

#[allow(clippy::too_many_arguments)]
fn rank_cliques_into_result<N, D, R>(
    outcome: SearchOutcome,
    vertex_pairs: &[(usize, usize)],
    first_edge_map: &[(N, N)],
    second_edge_map: &[(N, N)],
    disambiguate: &mut D,
    ranker: &R,
    first_vertices: usize,
    first_edges: usize,
    second_vertices: usize,
    second_edges: usize,
) -> McesResult<N>
where
    N: Eq + Copy + Ord + core::fmt::Debug,
    D: McesDisambiguate<N>,
    R: CliqueRanker<EagerCliqueInfo<N>>,
{
    let SearchOutcome { cliques, nodes, completed } = outcome;
    let mut infos: Vec<EagerCliqueInfo<N>> = cliques
        .into_iter()
        .map(|clique| {
            EagerCliqueInfo::new(
                clique,
                vertex_pairs,
                first_edge_map,
                second_edge_map,
                |a, b, c, d| disambiguate.disambiguate(a, b, c, d),
            )
        })
        .collect();

    infos.sort_by(|left, right| ranker.compare(left, right));
    build_result(
        infos,
        first_vertices,
        first_edges,
        second_vertices,
        second_edges,
        nodes,
        completed,
    )
}

/// Default ranker: fragment count → largest fragment.
pub type DefaultRanker = ChainedRanker<FragmentCountRanker, LargestFragmentMetricRanker>;

fn default_ranker() -> DefaultRanker {
    default_ranker_with_metric(LargestFragmentMetric::Edges)
}

fn default_ranker_with_metric(metric: LargestFragmentMetric) -> DefaultRanker {
    FragmentCountRanker.then(LargestFragmentMetricRanker::new(metric))
}

// ============================================================================
// McesResult
// ============================================================================

/// Result of an MCES computation.
///
/// Contains the best-ranked clique's matched edges, vertex matches, and
/// all data needed for similarity computation. Implements [`GraphSimilarities`]
/// for convenient access to Johnson, Tanimoto, Dice, etc.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct McesResult<N> {
    matched_edges: Vec<MatchedEdgePair<N>>,
    vertex_matches: Vec<(N, N)>,
    fragment_count: usize,
    largest_fragment_size: usize,
    common_edges: usize,
    common_vertices: usize,
    first_graph_vertices: usize,
    first_graph_edges: usize,
    second_graph_vertices: usize,
    second_graph_edges: usize,
    all_cliques: Vec<EagerCliqueInfo<N>>,
    search_nodes: usize,
    search_completed: bool,
}

impl<N: Eq + Copy + Ord + core::fmt::Debug> McesResult<N> {
    /// Matched edge pairs from the best-ranked clique.
    #[inline]
    #[must_use]
    pub fn matched_edges(&self) -> &[MatchedEdgePair<N>] {
        &self.matched_edges
    }

    /// Deterministic count of branch-and-bound nodes spent in the clique
    /// search.
    ///
    /// This is a noise-free effort metric: it depends only on the input and the
    /// search, never on wall-clock time, so it is identical on every rerun. It
    /// is `0` for inputs rejected by similarity or distance pre-screening (no
    /// search runs).
    #[inline]
    #[must_use]
    pub fn search_nodes(&self) -> usize {
        self.search_nodes
    }

    /// Whether the clique search ran to completion.
    ///
    /// `true` when the search tree was fully explored (the returned MCES is
    /// proven maximum). `false` when a configured search budget aborted the
    /// search early, in which case the returned clique is a valid lower bound
    /// rather than a proven maximum. See [`McesBuilder::with_search_budget`].
    #[inline]
    #[must_use]
    pub fn search_completed(&self) -> bool {
        self.search_completed
    }

    /// Matched vertex pairs from the best-ranked clique.
    #[inline]
    #[must_use]
    pub fn vertex_matches(&self) -> &[(N, N)] {
        &self.vertex_matches
    }

    /// Number of connected fragments in the best-ranked clique.
    #[inline]
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.fragment_count
    }

    /// Edge count of the largest fragment in the best-ranked clique.
    #[inline]
    #[must_use]
    pub fn largest_fragment_size(&self) -> usize {
        self.largest_fragment_size
    }

    /// All ranked clique infos (best first).
    ///
    /// In [`McesSearchMode::PartialEnumeration`], this contains the retained
    /// tied-best subset already accepted by the partitioned search, ranked
    /// best-first. In [`McesSearchMode::AllBest`], it contains the full tied
    /// maximum set already accepted by the search, ranked best-first.
    #[inline]
    #[must_use]
    pub fn all_cliques(&self) -> &[EagerCliqueInfo<N>] {
        &self.all_cliques
    }
}

impl<N: Eq + Copy + Ord + core::fmt::Debug> GraphSimilarities for McesResult<N> {
    #[inline]
    fn common_edges(&self) -> usize {
        self.common_edges
    }
    #[inline]
    fn common_vertices(&self) -> usize {
        self.common_vertices
    }
    #[inline]
    fn first_graph_vertices(&self) -> usize {
        self.first_graph_vertices
    }
    #[inline]
    fn first_graph_edges(&self) -> usize {
        self.first_graph_edges
    }
    #[inline]
    fn second_graph_vertices(&self) -> usize {
        self.second_graph_vertices
    }
    #[inline]
    fn second_graph_edges(&self) -> usize {
        self.second_graph_edges
    }
}

// ============================================================================
// McesBuilder
// ============================================================================

/// Builder for MCES computation with typestate generics.
///
/// All parameters have sensible defaults. Override with `.with_*()` methods.
/// Call `.compute_unlabeled()` to run the pipeline.
pub struct McesBuilder<'g, G, PF, XC, EC, D, R> {
    first: &'g G,
    second: &'g G,
    pair_filter: PF,
    edge_contexts: XC,
    edge_comparator: EC,
    disambiguate: D,
    ranker: R,
    product_vertex_ordering: InitialProductVertexOrdering,
    use_partition: bool,
    search_mode: McesSearchMode,
    delta_y: bool,
    ignore_edge_values: bool,
    similarity_threshold: Option<f64>,
    distance_threshold: Option<f64>,
    search_budget: Option<usize>,
}

impl<'g, G>
    McesBuilder<
        'g,
        G,
        AcceptAllPairs,
        AcceptAllEdgeContexts,
        StrictEqualityComparator,
        ArbitraryDisambiguate,
        DefaultRanker,
    >
{
    /// Creates a new MCES builder with default parameters.
    #[must_use]
    pub fn new(first: &'g G, second: &'g G) -> Self {
        Self {
            first,
            second,
            pair_filter: AcceptAllPairs,
            edge_contexts: AcceptAllEdgeContexts,
            edge_comparator: StrictEqualityComparator,
            disambiguate: ArbitraryDisambiguate,
            ranker: default_ranker(),
            product_vertex_ordering: InitialProductVertexOrdering::None,
            use_partition: true,
            search_mode: McesSearchMode::PartialEnumeration,
            delta_y: true,
            ignore_edge_values: false,
            similarity_threshold: None,
            distance_threshold: None,
            search_budget: None,
        }
    }
}

impl<G, PF, XC, EC, D> McesBuilder<'_, G, PF, XC, EC, D, DefaultRanker> {
    /// Chooses which fragment-size metric the built-in default ranker uses.
    ///
    /// This only affects the default tie-breaking chain
    /// (`FragmentCountRanker -> largest fragment`).
    /// If you need a fully custom policy, use [`McesBuilder::with_ranker`]
    /// instead.
    ///
    /// The default is [`LargestFragmentMetric::Edges`]. For RDKit-oriented
    /// comparisons, [`LargestFragmentMetric::Atoms`] is often the more relevant
    /// choice because RDKit's `LargestFragSize` is atom-based.
    #[must_use]
    pub fn with_largest_fragment_metric(mut self, metric: LargestFragmentMetric) -> Self {
        self.ranker = default_ranker_with_metric(metric);
        self
    }
}

impl<'g, G, PF, XC, EC, D, R> McesBuilder<'g, G, PF, XC, EC, D, R> {
    /// Sets a custom pair filter for modular product construction.
    ///
    /// `filter(i, j)` is called for each `(i, j) ∈ V(LG1) × V(LG2)` to
    /// decide inclusion.
    ///
    /// For labeled MCES, this is applied after the built-in bond-label
    /// compatibility check, so it can only further restrict the candidate
    /// bond pairs.
    #[must_use]
    pub fn with_pair_filter<F: FnMut(usize, usize) -> bool>(
        self,
        f: F,
    ) -> McesBuilder<'g, G, CustomPairFilter<F>, XC, EC, D, R> {
        McesBuilder {
            first: self.first,
            second: self.second,
            pair_filter: CustomPairFilter(f),
            edge_contexts: self.edge_contexts,
            edge_comparator: self.edge_comparator,
            disambiguate: self.disambiguate,
            ranker: self.ranker,
            product_vertex_ordering: self.product_vertex_ordering,
            use_partition: self.use_partition,
            search_mode: self.search_mode,
            delta_y: self.delta_y,
            ignore_edge_values: self.ignore_edge_values,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
            search_budget: self.search_budget,
        }
    }

    /// Sets a custom edge comparator for labeled modular product construction.
    ///
    /// The comparator receives `Option<Value>` from each graph's labeled line
    /// graph adjacency and returns `true` if the edge values are compatible.
    #[must_use]
    pub fn with_edge_comparator<F>(
        self,
        f: F,
    ) -> McesBuilder<'g, G, PF, XC, CustomEdgeComparator<F>, D, R> {
        McesBuilder {
            first: self.first,
            second: self.second,
            pair_filter: self.pair_filter,
            edge_contexts: self.edge_contexts,
            edge_comparator: CustomEdgeComparator(f),
            disambiguate: self.disambiguate,
            ranker: self.ranker,
            product_vertex_ordering: self.product_vertex_ordering,
            use_partition: self.use_partition,
            search_mode: self.search_mode,
            delta_y: self.delta_y,
            ignore_edge_values: self.ignore_edge_values,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
            search_budget: self.search_budget,
        }
    }

    /// Sets a custom disambiguation closure for isolated edge vertex matching.
    #[must_use]
    pub fn with_disambiguate<F>(
        self,
        f: F,
    ) -> McesBuilder<'g, G, PF, XC, EC, CustomDisambiguate<F>, R> {
        McesBuilder {
            first: self.first,
            second: self.second,
            pair_filter: self.pair_filter,
            edge_contexts: self.edge_contexts,
            edge_comparator: self.edge_comparator,
            disambiguate: CustomDisambiguate(f),
            ranker: self.ranker,
            product_vertex_ordering: self.product_vertex_ordering,
            use_partition: self.use_partition,
            search_mode: self.search_mode,
            delta_y: self.delta_y,
            ignore_edge_values: self.ignore_edge_values,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
            search_budget: self.search_budget,
        }
    }

    /// Sets a custom clique ranker.
    ///
    /// The ranker is only used to choose among cliques with the same maximum
    /// edge count. In other words, the maximum clique search still optimizes
    /// matched edges first; the ranker only breaks ties afterward.
    ///
    /// For ad-hoc policies, use
    /// [`FnRanker`](crate::traits::algorithms::clique_ranking::FnRanker). For
    /// reusable lexicographic policies, chain rankers with
    /// [`CliqueRankerExt::then`].
    ///
    /// Note that `matched_edges().len()` is already identical across the
    /// cliques being ranked. If you want an edge-centric tiebreaker, rank by a
    /// fragment-edge statistic such as [`CliqueInfo::largest_fragment_size()`]
    /// rather than total matched edges.
    ///
    /// # Examples
    ///
    /// The examples below use `AllBest` so the custom ranker can choose among
    /// tied maxima when more than one is retained.
    ///
    /// Rank by matched nodes:
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     naive_structs::UndiGraph,
    ///     prelude::*,
    ///     traits::{
    ///         VocabularyBuilder,
    ///         algorithms::randomized_graphs::{cycle_graph, path_graph},
    ///     },
    /// };
    ///
    /// fn wrap_undi(g: SymmetricCSR2D<CSR2D<usize, usize, usize>>) -> UndiGraph<usize> {
    ///     let n = g.order();
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(n)
    ///         .symbols((0..n).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, g))
    /// }
    ///
    /// let g1 = wrap_undi(cycle_graph(4));
    /// let g2 = wrap_undi(path_graph(4));
    ///
    /// let result = McesBuilder::new(&g1, &g2)
    ///     .with_search_mode(McesSearchMode::AllBest)
    ///     .with_ranker(FnRanker::new(|a: &EagerCliqueInfo<usize>, b: &EagerCliqueInfo<usize>| {
    ///         b.vertex_matches().len().cmp(&a.vertex_matches().len())
    ///     }))
    ///     .compute_unlabeled();
    ///
    /// assert_eq!(result.matched_edges().len(), 3);
    /// ```
    ///
    /// Rank by fragment edges:
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     naive_structs::UndiGraph,
    ///     prelude::*,
    ///     traits::{
    ///         VocabularyBuilder,
    ///         algorithms::randomized_graphs::{cycle_graph, path_graph},
    ///     },
    /// };
    ///
    /// fn wrap_undi(g: SymmetricCSR2D<CSR2D<usize, usize, usize>>) -> UndiGraph<usize> {
    ///     let n = g.order();
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(n)
    ///         .symbols((0..n).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, g))
    /// }
    ///
    /// let g1 = wrap_undi(cycle_graph(4));
    /// let g2 = wrap_undi(path_graph(4));
    ///
    /// let result = McesBuilder::new(&g1, &g2)
    ///     .with_search_mode(McesSearchMode::AllBest)
    ///     .with_ranker(FragmentCountRanker.then(FnRanker::new(
    ///         |a: &EagerCliqueInfo<usize>, b: &EagerCliqueInfo<usize>| {
    ///             b.largest_fragment_size().cmp(&a.largest_fragment_size())
    ///         },
    ///     )))
    ///     .compute_unlabeled();
    ///
    /// assert_eq!(result.matched_edges().len(), 3);
    /// ```
    ///
    /// Rank by a mixed policy: fewer fragments, then more matched nodes, then
    /// larger fragment edges:
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     naive_structs::UndiGraph,
    ///     prelude::*,
    ///     traits::{
    ///         VocabularyBuilder,
    ///         algorithms::randomized_graphs::{cycle_graph, path_graph},
    ///     },
    /// };
    ///
    /// fn wrap_undi(g: SymmetricCSR2D<CSR2D<usize, usize, usize>>) -> UndiGraph<usize> {
    ///     let n = g.order();
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(n)
    ///         .symbols((0..n).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, g))
    /// }
    ///
    /// let g1 = wrap_undi(cycle_graph(4));
    /// let g2 = wrap_undi(path_graph(4));
    ///
    /// let mixed_ranker = FragmentCountRanker
    ///     .then(FnRanker::new(|a: &EagerCliqueInfo<usize>, b: &EagerCliqueInfo<usize>| {
    ///         b.vertex_matches().len().cmp(&a.vertex_matches().len())
    ///     }))
    ///     .then(FnRanker::new(|a: &EagerCliqueInfo<usize>, b: &EagerCliqueInfo<usize>| {
    ///         b.largest_fragment_size().cmp(&a.largest_fragment_size())
    ///     }));
    ///
    /// let result = McesBuilder::new(&g1, &g2)
    ///     .with_search_mode(McesSearchMode::AllBest)
    ///     .with_ranker(mixed_ranker)
    ///     .compute_unlabeled();
    ///
    /// assert_eq!(result.matched_edges().len(), 3);
    /// ```
    #[must_use]
    pub fn with_ranker<R2>(self, ranker: R2) -> McesBuilder<'g, G, PF, XC, EC, D, R2> {
        McesBuilder {
            first: self.first,
            second: self.second,
            pair_filter: self.pair_filter,
            edge_contexts: self.edge_contexts,
            edge_comparator: self.edge_comparator,
            disambiguate: self.disambiguate,
            ranker,
            product_vertex_ordering: self.product_vertex_ordering,
            use_partition: self.use_partition,
            search_mode: self.search_mode,
            delta_y: self.delta_y,
            ignore_edge_values: self.ignore_edge_values,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
            search_budget: self.search_budget,
        }
    }

    /// Adds precomputed per-edge contexts to labeled MCES.
    ///
    /// This does not compute any contexts itself. It only enforces the
    /// following compatibility rule during labeled bond-pair admission:
    /// - both context rows empty => compatible
    /// - exactly one row empty => incompatible
    /// - both non-empty => compatible iff the two rows intersect
    ///
    /// Unlabeled MCES ignores this setting.
    #[must_use]
    pub fn with_edge_contexts<Signature, SparseIndex>(
        self,
        first: &'g EdgeContexts<Signature, SparseIndex>,
        second: &'g EdgeContexts<Signature, SparseIndex>,
    ) -> McesBuilder<'g, G, PF, ConfiguredEdgeContexts<'g, Signature, SparseIndex>, EC, D, R> {
        McesBuilder {
            first: self.first,
            second: self.second,
            pair_filter: self.pair_filter,
            edge_contexts: ConfiguredEdgeContexts { first, second },
            edge_comparator: self.edge_comparator,
            disambiguate: self.disambiguate,
            ranker: self.ranker,
            product_vertex_ordering: self.product_vertex_ordering,
            use_partition: self.use_partition,
            search_mode: self.search_mode,
            delta_y: self.delta_y,
            ignore_edge_values: self.ignore_edge_values,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
            search_budget: self.search_budget,
        }
    }

    /// Selects a built-in initial modular-product ordering policy.
    #[must_use]
    pub fn with_initial_product_vertex_ordering(
        mut self,
        ordering: InitialProductVertexOrdering,
    ) -> Self {
        self.product_vertex_ordering = ordering;
        self
    }

    /// Enables or disables partition-aware maximum clique search (default:
    /// enabled).
    #[must_use]
    pub fn with_partition(mut self, enabled: bool) -> Self {
        self.use_partition = enabled;
        self
    }

    /// Selects how the clique stage explores tied maximum solutions.
    ///
    /// The default is [`McesSearchMode::PartialEnumeration`], which mirrors the
    /// RDKit-style partitioned default more closely while avoiding full tied
    /// best enumeration.
    #[must_use]
    pub fn with_search_mode(mut self, search_mode: McesSearchMode) -> Self {
        self.search_mode = search_mode;
        self
    }

    /// Enables or disables Delta-Y exchange filtering (default: enabled).
    ///
    /// When enabled, cliques whose matched edge subgraphs have different sorted
    /// degree sequences in the two original graphs are discarded. This catches
    /// the Whitney K₃/K₁,₃ exception.
    #[must_use]
    pub fn with_delta_y(mut self, enabled: bool) -> Self {
        self.delta_y = enabled;
        self
    }

    /// Enables or disables ignoring original graph edge values in labeled bond
    /// identity (default: disabled).
    ///
    /// When enabled, labeled MCES matches bonds using only the canonical
    /// endpoint node-type pair and ignores the original edge value. This
    /// mirrors RDKit's `ignoreBondOrders` behavior when edge values encode bond
    /// order.
    #[must_use]
    pub fn with_ignore_edge_values(mut self, enabled: bool) -> Self {
        self.ignore_edge_values = enabled;
        self
    }

    /// Sets a minimum similarity threshold for pre-screening (RASCAL-style).
    ///
    /// Before running the expensive pipeline, a cheap upper bound on similarity
    /// is computed from degree sequences. If the bound is below this threshold,
    /// the pipeline is skipped and an empty result is returned.
    ///
    /// Typical values: 0.5–0.7.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = Some(threshold);
        self
    }

    /// Sets a maximum distance threshold for pre-screening (myopic-style).
    ///
    /// Before running the expensive pipeline, a cheap lower bound on edit
    /// distance is computed from degree sequences. If the bound exceeds this
    /// threshold, the pipeline is skipped and an empty result is returned.
    #[must_use]
    pub fn with_distance_threshold(mut self, threshold: f64) -> Self {
        self.distance_threshold = Some(threshold);
        self
    }

    /// Sets a deterministic search budget for the maximum-clique stage.
    ///
    /// The budget is a cap on the number of branch-and-bound nodes the search
    /// may visit. When the cap is reached the search stops and returns the best
    /// clique found so far, a valid lower bound rather than a proven maximum,
    /// and [`McesResult::search_completed`] reports `false`. The node count is
    /// deterministic, so a given input and budget always abort at the same
    /// point. `None` (the default) means unbounded. `Some(0)` aborts before the
    /// first node.
    ///
    /// The same node count is reported via [`McesResult::search_nodes`] on
    /// every search, so it doubles as a reproducible effort metric.
    ///
    /// The cap bounds the main retained clique search only. The bounded
    /// lower-bound seed heuristic that precedes it runs regardless, so a
    /// `Some(0)` budget still performs that fixed preprocessing.
    #[must_use]
    pub fn with_search_budget(mut self, max_nodes: usize) -> Self {
        self.search_budget = Some(max_nodes);
        self
    }
}

// ============================================================================
// compute_unlabeled
// ============================================================================

impl<G, PF, XC, EC, D, R> McesBuilder<'_, G, PF, XC, EC, D, R>
where
    G: LineGraph,
    G::NodeId: Eq + Copy + Ord + core::fmt::Debug + AsPrimitive<usize>,
    PF: McesPairFilter,
    D: McesDisambiguate<G::NodeId>,
    R: CliqueRanker<EagerCliqueInfo<G::NodeId>>,
{
    /// Runs the unlabeled MCES pipeline.
    ///
    /// 1. Builds line graphs for both input graphs.
    /// 2. Constructs the modular product with the configured pair filter.
    /// 3. Finds maximum cliques according to the configured search mode
    ///    (partition-aware if enabled).
    /// 4. Builds `EagerCliqueInfo` for each clique (with vertex matching).
    /// 5. Ranks cliques and returns the best as `McesResult`.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn compute_unlabeled(mut self) -> McesResult<G::NodeId> {
        // 1. Build line graphs.
        let lg1 = self.first.line_graph();
        let lg2 = self.second.line_graph();

        // Graph sizes for similarity.
        let first_vertices: usize = self.first.number_of_nodes().as_();
        let first_edges = lg1.number_of_vertices();
        let second_vertices: usize = self.second.number_of_nodes().as_();
        let second_edges = lg2.number_of_vertices();

        // 2. Pre-screening (tier 1).
        if self.similarity_threshold.is_some() || self.distance_threshold.is_some() {
            let groups1 = extract_degree_groups(self.first, |_| ());
            let groups2 = extract_degree_groups(self.second, |_| ());
            let estimate = tier1_screening(&groups1, &groups2);
            if estimate.is_rejected(
                first_vertices,
                first_edges,
                second_vertices,
                second_edges,
                self.similarity_threshold,
                self.distance_threshold,
            ) {
                return build_result(
                    Vec::new(),
                    first_vertices,
                    first_edges,
                    second_vertices,
                    second_edges,
                    0,
                    true,
                );
            }
        }

        let mut product_vertex_pairs = Vec::new();
        let mut edge_pair_allowed = vec![false; first_edges * second_edges];
        for i in 0..first_edges {
            for j in 0..second_edges {
                let allowed = self.pair_filter.filter(i, j);
                edge_pair_allowed[i * second_edges + j] = allowed;
                if allowed {
                    product_vertex_pairs.push((i, j));
                }
            }
        }

        // 2. Modular product.
        let mp = lg1.graph().modular_product(lg2.graph(), &product_vertex_pairs);
        let (mp_matrix, mp_vertex_pairs) = reorder_product_with_source_edge_ordering(
            mp,
            product_vertex_pairs,
            lg1.edge_map(),
            lg2.edge_map(),
            self.product_vertex_ordering,
            SourceEdgeOrderingComputers {
                edge_signatures: || {
                    (
                        unlabeled_edge_signatures(lg1.edge_map(), first_vertices),
                        unlabeled_edge_signatures(lg2.edge_map(), second_vertices),
                    )
                },
                wl_signatures: || {
                    (
                        unlabeled_line_graph_wl_signatures(
                            lg1.graph(),
                            lg1.edge_map(),
                            first_vertices,
                        ),
                        unlabeled_line_graph_wl_signatures(
                            lg2.graph(),
                            lg2.edge_map(),
                            second_vertices,
                        ),
                    )
                },
            },
        );
        // 3. Maximum cliques (unlabeled: all bonds get label 0).
        let search_outcome = if self.use_partition {
            let g1_labels = vec![0usize; first_edges];
            let g2_labels = vec![0usize; second_edges];
            let info = PartitionInfo {
                pairs: &mp_vertex_pairs,
                g1_labels: &g1_labels,
                g2_labels: &g2_labels,
                num_labels: 1,
                partition_side: choose_partition_side_by_atom_counts(
                    first_vertices,
                    second_vertices,
                ),
            };
            let initial_lower_bound = connected_tree_lower_bound(
                self.first,
                self.second,
                lg1.edge_map(),
                lg2.edge_map(),
                &vec![true; first_vertices * second_vertices],
                &edge_pair_allowed,
            )
            .unwrap_or(0)
            .max(usize::from(mp_matrix.order() > 0));
            accepted_partitioned_cliques(
                &mp_matrix,
                &info,
                initial_lower_bound,
                self.search_mode,
                self.search_budget,
                |clique| {
                    !self.delta_y
                        || !clique_has_delta_y(
                            clique,
                            &mp_vertex_pairs,
                            lg1.edge_map(),
                            lg2.edge_map(),
                            first_vertices,
                            second_vertices,
                        )
                },
            )
        } else {
            accepted_cliques(&mp_matrix, self.search_mode, self.search_budget, |clique| {
                !self.delta_y
                    || !clique_has_delta_y(
                        clique,
                        &mp_vertex_pairs,
                        lg1.edge_map(),
                        lg2.edge_map(),
                        first_vertices,
                        second_vertices,
                    )
            })
        };

        rank_cliques_into_result(
            search_outcome,
            &mp_vertex_pairs,
            lg1.edge_map(),
            lg2.edge_map(),
            &mut self.disambiguate,
            &self.ranker,
            first_vertices,
            first_edges,
            second_vertices,
            second_edges,
        )
    }
}

// ============================================================================
// compute_labeled
// ============================================================================

impl<G, PF, XC, EC, D, R> McesBuilder<'_, G, PF, XC, EC, D, R>
where
    G: LabeledLineGraph,
    G::NodeId: Eq + Copy + Ord + core::fmt::Debug + AsPrimitive<usize>,
    G::NodeSymbol: TypedNode,
    <G::NodeSymbol as TypedNode>::NodeType: Copy + Ord,
    <G::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix:
        SparseValuedMatrix2D<RowIndex = G::NodeId, ColumnIndex = G::NodeId>,
    <<G::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix as ValuedMatrix>::Value:
        Copy + PartialEq,
    PF: McesPairFilter,
    XC: McesEdgeContexts,
    EC: McesEdgeComparator<
            <G::NodeSymbol as TypedNode>::NodeType,
            <G::NodeSymbol as TypedNode>::NodeType,
        >,
    D: McesDisambiguate<G::NodeId>,
    R: CliqueRanker<EagerCliqueInfo<G::NodeId>>,
{
    /// Runs the labeled MCES pipeline.
    ///
    /// Uses [`LabeledLineGraph`] to construct line graphs with node-type edge
    /// labels, then builds a labeled modular product over the admissible
    /// original-edge pairs using the configured edge comparator.
    ///
    /// Only bond-label-compatible pairs enter the modular product. The current
    /// bond label is the canonical endpoint node-type pair together with the
    /// original edge value unless [`McesBuilder::with_ignore_edge_values`] is
    /// enabled. Precomputed edge contexts, when provided, further restrict
    /// which original edge pairs are allowed into the product before any
    /// caller-provided pair filter runs.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn compute_labeled(mut self) -> McesResult<G::NodeId> {
        // 1. Build labeled line graphs.
        let lg1 = self.first.labeled_line_graph();
        let lg2 = self.second.labeled_line_graph();

        // Graph sizes for similarity.
        let first_vertices: usize = self.first.number_of_nodes().as_();
        let first_edges = lg1.number_of_vertices();
        let second_vertices: usize = self.second.number_of_nodes().as_();
        let second_edges = lg2.number_of_vertices();

        // 1b. Compute bond labels once and reuse them for screening,
        // vertex-pair filtering, and the partition-aware clique bound.
        let g1_bond_labels =
            compute_bond_labels(self.first, lg1.edge_map(), self.ignore_edge_values);
        let g2_bond_labels =
            compute_bond_labels(self.second, lg2.edge_map(), self.ignore_edge_values);
        let (g1_label_indices, g2_label_indices, num_labels) =
            intern_shared_labels(&g1_bond_labels, &g2_bond_labels);
        let first_node_types: Vec<_> =
            self.first.nodes().map(|symbol| symbol.node_type()).collect();
        let second_node_types: Vec<_> =
            self.second.nodes().map(|symbol| symbol.node_type()).collect();

        // 1c. Pre-screening (tier 2).
        // Mirror RDKit's second screening stage: within each atom bucket,
        // solve a rectangular assignment over incident bond-label overlap.
        if self.similarity_threshold.is_some() || self.distance_threshold.is_some() {
            let groups1 =
                extract_degree_sequences(self.first, |node_id| first_node_types[node_id.as_()]);
            let groups2 =
                extract_degree_sequences(self.second, |node_id| second_node_types[node_id.as_()]);
            let first_incident_bond_labels =
                build_incident_bond_labels(first_vertices, lg1.edge_map(), &g1_label_indices);
            let second_incident_bond_labels =
                build_incident_bond_labels(second_vertices, lg2.edge_map(), &g2_label_indices);
            let estimate = tier2_screening(
                &groups1,
                &groups2,
                &first_incident_bond_labels,
                &second_incident_bond_labels,
            );
            if estimate.is_rejected(
                first_vertices,
                first_edges,
                second_vertices,
                second_edges,
                self.similarity_threshold,
                self.distance_threshold,
            ) {
                return build_result(
                    Vec::new(),
                    first_vertices,
                    first_edges,
                    second_vertices,
                    second_edges,
                    0,
                    true,
                );
            }
        }

        // 2. Reuse the bond labels for the partition-aware clique bound.
        self.edge_contexts.validate(first_edges, second_edges);

        let mut product_vertex_pairs = Vec::new();
        let mut edge_pair_allowed = vec![false; first_edges * second_edges];
        {
            let edge_contexts = &self.edge_contexts;
            let pair_filter = &mut self.pair_filter;
            for i in 0..first_edges {
                for j in 0..second_edges {
                    let allowed = g1_label_indices[i] == g2_label_indices[j]
                        && edge_contexts.compatible(i, j)
                        && pair_filter.filter(i, j);
                    edge_pair_allowed[i * second_edges + j] = allowed;
                    if allowed {
                        product_vertex_pairs.push((i, j));
                    }
                }
            }
        }

        let none_none_compatible =
            self.edge_comparator.compare(None::<<G::NodeSymbol as TypedNode>::NodeType>, None);
        let mut junction_compatible = Vec::with_capacity(first_vertices * second_vertices);
        {
            let edge_comparator = &self.edge_comparator;
            for &first_type in &first_node_types {
                for &second_type in &second_node_types {
                    junction_compatible
                        .push(edge_comparator.compare(Some(first_type), Some(second_type)));
                }
            }
        }

        // 2. Labeled modular product.
        let edge_comparator = &self.edge_comparator;
        let mp = lg1.graph().labeled_modular_product(lg2.graph(), &product_vertex_pairs, |a, b| {
            edge_comparator.compare(a, b)
        });
        let (mp_matrix, mp_vertex_pairs) = reorder_product_with_source_edge_ordering(
            mp,
            product_vertex_pairs,
            lg1.edge_map(),
            lg2.edge_map(),
            self.product_vertex_ordering,
            SourceEdgeOrderingComputers {
                edge_signatures: || {
                    (
                        labeled_edge_signatures(lg1.edge_map(), first_vertices, &g1_label_indices),
                        labeled_edge_signatures(lg2.edge_map(), second_vertices, &g2_label_indices),
                    )
                },
                wl_signatures: || {
                    (
                        labeled_line_graph_wl_signatures(
                            lg1.graph(),
                            lg1.edge_map(),
                            first_vertices,
                            &g1_label_indices,
                        ),
                        labeled_line_graph_wl_signatures(
                            lg2.graph(),
                            lg2.edge_map(),
                            second_vertices,
                            &g2_label_indices,
                        ),
                    )
                },
            },
        );
        // 3. Maximum cliques (label-aware partition bound).
        let search_outcome = if self.use_partition {
            let info = PartitionInfo {
                pairs: &mp_vertex_pairs,
                g1_labels: &g1_label_indices,
                g2_labels: &g2_label_indices,
                num_labels,
                partition_side: choose_partition_side_by_atom_counts(
                    first_vertices,
                    second_vertices,
                ),
            };
            let initial_lower_bound = if none_none_compatible {
                connected_tree_lower_bound(
                    self.first,
                    self.second,
                    lg1.edge_map(),
                    lg2.edge_map(),
                    &junction_compatible,
                    &edge_pair_allowed,
                )
                .unwrap_or(0)
            } else {
                0
            }
            .max(usize::from(mp_matrix.order() > 0));
            accepted_partitioned_cliques(
                &mp_matrix,
                &info,
                initial_lower_bound,
                self.search_mode,
                self.search_budget,
                |clique| {
                    !self.delta_y
                        || !clique_has_delta_y(
                            clique,
                            &mp_vertex_pairs,
                            lg1.edge_map(),
                            lg2.edge_map(),
                            first_vertices,
                            second_vertices,
                        )
                },
            )
        } else {
            accepted_cliques(&mp_matrix, self.search_mode, self.search_budget, |clique| {
                !self.delta_y
                    || !clique_has_delta_y(
                        clique,
                        &mp_vertex_pairs,
                        lg1.edge_map(),
                        lg2.edge_map(),
                        first_vertices,
                        second_vertices,
                    )
            })
        };

        rank_cliques_into_result(
            search_outcome,
            &mp_vertex_pairs,
            lg1.edge_map(),
            lg2.edge_map(),
            &mut self.disambiguate,
            &self.ranker,
            first_vertices,
            first_edges,
            second_vertices,
            second_edges,
        )
    }
}

/// Constructs an `McesResult` from ranked clique infos.
#[allow(clippy::too_many_arguments)]
fn build_result<N>(
    infos: Vec<EagerCliqueInfo<N>>,
    first_graph_vertices: usize,
    first_graph_edges: usize,
    second_graph_vertices: usize,
    second_graph_edges: usize,
    search_nodes: usize,
    search_completed: bool,
) -> McesResult<N>
where
    N: Eq + Copy + Ord + core::fmt::Debug,
{
    let (
        matched_edges,
        vertex_matches,
        fragment_count,
        largest_fragment_size,
        common_edges,
        common_vertices,
    ) = if let Some(best) = infos.first() {
        (
            best.matched_edges().to_vec(),
            best.vertex_matches().to_vec(),
            best.fragment_count(),
            best.largest_fragment_size(),
            best.matched_edges().len(),
            best.vertex_matches().len(),
        )
    } else {
        (Vec::new(), Vec::new(), 0, 0, 0, 0)
    };

    McesResult {
        matched_edges,
        vertex_matches,
        fragment_count,
        largest_fragment_size,
        common_edges,
        common_vertices,
        first_graph_vertices,
        first_graph_edges,
        second_graph_vertices,
        second_graph_edges,
        all_cliques: infos,
        search_nodes,
        search_completed,
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;

    use super::{
        ScreeningEstimate, assignment_score_via_crouse, dense_rank, incident_label_overlap,
        intern_shared_labels, reorder_product_by_edge_signatures, tier1_screening, tier2_screening,
        unlabeled_edge_signatures, wl_refine_with_edge_colors,
    };
    use crate::{impls::BitSquareMatrix, traits::SparseMatrix2D};

    #[test]
    fn test_dense_rank_assigns_compact_ranks_by_sorted_value() {
        // Ranks are 0-based by sorted distinct value: 5 -> 0, 10 -> 1, 20 -> 2.
        assert_eq!(dense_rank(&[10u32, 10, 5, 20, 5]), vec![1, 1, 0, 2, 0]);
        assert!(dense_rank::<u32>(&[]).is_empty());
        assert_eq!(dense_rank(&[7u32, 7, 7]), vec![0, 0, 0]);
    }

    #[test]
    fn test_wl_refine_iterates_to_fixpoint_over_multiple_rounds() {
        // A 5-vertex path 0-1-2-3-4 with a uniform seed needs several
        // refinement rounds: a single round only separates the
        // endpoints from the interior, so reaching the distance-to-end
        // coloring exercises the iterate-again branch (`colors =
        // next_colors;`).
        let path = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let refined = wl_refine_with_edge_colors(&path, &[0usize; 5], |_, _| ());
        assert_eq!(refined, vec![0, 1, 2, 1, 0]);

        // An already-stable seed converges in one round and is returned as is
        // (after dense-ranking), with no further refinement.
        let stable = wl_refine_with_edge_colors(&path, &[0, 1, 2, 1, 0], |_, _| ());
        assert_eq!(stable, vec![0, 1, 2, 1, 0]);
    }

    #[test]
    fn test_intern_shared_labels_reuses_equal_labels() {
        let first = [(0_u8, 7_u8, 1_u8), (1, 9, 2)];
        let second = [(1_u8, 9_u8, 2_u8), (0, 11, 2)];

        let (first_indices, second_indices, num_labels) = intern_shared_labels(&first, &second);

        assert_eq!(num_labels, 3);
        assert_eq!(first_indices[1], second_indices[0]);
        assert_ne!(first_indices[0], second_indices[1]);
    }

    #[test]
    fn test_screening_estimate_threshold_boundaries_are_inclusive() {
        let estimate = ScreeningEstimate { vg1g2: 2, eg1g2_times2: 2 };
        let similarity = estimate.similarity(6, 15, 2, 1);
        let distance = estimate.distance(15, 1);

        assert!((similarity - (1.0 / 7.0)).abs() < 1.0e-12);
        assert_eq!(distance, 14);
        assert!(!estimate.is_rejected(6, 15, 2, 1, Some(similarity), Some(14.0)));
        assert!(estimate.is_rejected(6, 15, 2, 1, Some(similarity + 1.0e-12), None));
        assert!(estimate.is_rejected(6, 15, 2, 1, None, Some(13.0)));
    }

    #[test]
    fn test_tier1_screening_pairs_degrees_within_matching_labels() {
        let first = BTreeMap::from([('a', vec![4, 2]), ('b', vec![3]), ('c', vec![10])]);
        let second = BTreeMap::from([('a', vec![3, 1]), ('b', vec![5]), ('d', vec![7])]);

        let estimate = tier1_screening(&first, &second);

        assert_eq!(estimate.vg1g2, 3);
        assert_eq!(estimate.eg1g2_times2, 7);
    }

    #[test]
    fn test_incident_label_overlap_counts_multiset_intersection() {
        assert_eq!(incident_label_overlap(&[0, 0, 1, 3], &[0, 1, 1, 2, 3]), 3);
        assert_eq!(incident_label_overlap(&[2, 2], &[1, 1]), 0);
    }

    #[test]
    fn test_tier2_screening_uses_rectangular_assignment_score() {
        let first = BTreeMap::from([('c', vec![(3, 0), (1, 1)])]);
        let second = BTreeMap::from([('c', vec![(2, 0), (1, 1), (1, 2)])]);
        let first_incident = vec![vec![0, 0, 1], vec![2]];
        let second_incident = vec![vec![0, 1], vec![0, 2], vec![2]];

        let estimate = tier2_screening(&first, &second, &first_incident, &second_incident);

        assert_eq!(estimate.vg1g2, 2);
        assert_eq!(estimate.eg1g2_times2, 3);
    }

    #[test]
    fn test_assignment_score_via_crouse_returns_zero_for_empty_side() {
        let first_incident = vec![vec![0, 1]];
        let second_incident = vec![vec![0, 1]];

        assert_eq!(
            assignment_score_via_crouse(&[], &[(2, 0)], &first_incident, &second_incident),
            0
        );
        assert_eq!(
            assignment_score_via_crouse(&[(2, 0)], &[], &first_incident, &second_incident),
            0
        );
    }

    #[test]
    fn test_reorder_product_by_edge_signatures_applies_non_identity_permutation() {
        let mut matrix = BitSquareMatrix::new(3);
        matrix.set_symmetric(0, 1);
        matrix.set_symmetric(1, 2);
        let vertex_pairs = vec![(0usize, 0usize), (1, 0), (2, 0)];
        let first_edge_map = vec![(0usize, 1usize), (1, 2), (2, 3)];
        let second_edge_map = vec![(0usize, 1usize)];
        let first_edge_signatures = unlabeled_edge_signatures(&first_edge_map, 4);
        let second_edge_signatures = unlabeled_edge_signatures(&second_edge_map, 2);

        let (permuted, permuted_pairs) = reorder_product_by_edge_signatures(
            &matrix,
            &vertex_pairs,
            &first_edge_signatures,
            &second_edge_signatures,
        );

        assert_eq!(permuted_pairs, vec![(0, 0), (2, 0), (1, 0)]);
        assert!(!permuted.has_entry(0, 1));
        assert!(permuted.has_entry(1, 2));
        assert!(permuted.has_entry(0, 2));
    }
}
