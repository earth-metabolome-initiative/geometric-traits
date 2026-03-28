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

use alloc::{boxed::Box, collections::BTreeMap, vec::Vec};

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
        MaximumClique, PartitionInfo, PartitionSide, all_best_search, choose_partition_side,
        partial_search,
    },
    modular_product::ModularProduct,
};
use crate::{
    impls::{BitSquareMatrix, EdgeContexts},
    traits::{
        Edges, MonopartiteEdges, MonoplexMonopartiteGraph, PositiveInteger, SparseMatrix2D,
        SparseValuedMatrix2D, SquareMatrix, TypedNode, ValuedMatrix,
    },
};

// ============================================================================
// Default ZSTs
// ============================================================================

/// Default pair filter: accept all vertex pairs.
pub struct AcceptAllPairs;

/// Default disambiguation: arbitrary orientation (always `true`).
pub struct ArbitraryDisambiguate;

/// Default edge-context constraint: accept all bond pairs.
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
        if let Some(t) = sim_threshold {
            if self.similarity(v1, e1, v2, e2) < t {
                return true;
            }
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

type ProductVertexOrdering<'g> =
    dyn FnMut(usize, usize, (usize, usize), (usize, usize)) -> (usize, usize) + 'g;

fn reorder_product_for_search<N>(
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    first_edge_map: &[(N, N)],
    second_edge_map: &[(N, N)],
    ordering: Option<&mut Box<ProductVertexOrdering<'_>>>,
) -> (BitSquareMatrix, Vec<(usize, usize)>)
where
    N: Copy + AsPrimitive<usize>,
{
    let Some(ordering) = ordering else {
        return (matrix, vertex_pairs);
    };

    let mut ranked_indices: Vec<((usize, usize), usize)> = vertex_pairs
        .iter()
        .enumerate()
        .map(|(index, &(first_lg, second_lg))| {
            let first_edge = (first_edge_map[first_lg].0.as_(), first_edge_map[first_lg].1.as_());
            let second_edge =
                (second_edge_map[second_lg].0.as_(), second_edge_map[second_lg].1.as_());
            (((*ordering)(first_lg, second_lg, first_edge, second_edge)), index)
        })
        .collect();
    ranked_indices.sort_unstable();

    if ranked_indices.iter().enumerate().all(|(new, (_, old))| new == *old) {
        return (matrix, vertex_pairs);
    }

    let order: Vec<usize> = ranked_indices.into_iter().map(|(_, index)| index).collect();
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

fn accepted_cliques<M, F>(
    matrix: &M,
    search_mode: McesSearchMode,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    M: MaximumClique,
    F: FnMut(&[usize]) -> bool,
{
    match search_mode {
        McesSearchMode::PartialEnumeration => vec![matrix.maximum_clique_where(accept_clique)],
        McesSearchMode::AllBest => matrix.all_maximum_cliques_where(accept_clique),
    }
}

fn accepted_partitioned_cliques<F>(
    matrix: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    search_mode: McesSearchMode,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    match search_mode {
        McesSearchMode::PartialEnumeration => {
            let initial_lower_bound = usize::from(matrix.order() > 0);
            partial_search(matrix, partition, initial_lower_bound, accept_clique)
        }
        McesSearchMode::AllBest => all_best_search(matrix, partition, 0, accept_clique),
    }
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
}

impl<N: Eq + Copy + Ord + core::fmt::Debug> McesResult<N> {
    /// Matched edge pairs from the best-ranked clique.
    #[inline]
    #[must_use]
    pub fn matched_edges(&self) -> &[MatchedEdgePair<N>] {
        &self.matched_edges
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
    product_vertex_ordering: Option<Box<ProductVertexOrdering<'g>>>,
    use_partition: bool,
    search_mode: McesSearchMode,
    delta_y: bool,
    ignore_edge_values: bool,
    partition_orientation_heuristic: bool,
    similarity_threshold: Option<f64>,
    distance_threshold: Option<f64>,
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
            product_vertex_ordering: None,
            use_partition: true,
            search_mode: McesSearchMode::PartialEnumeration,
            delta_y: true,
            ignore_edge_values: false,
            partition_orientation_heuristic: true,
            similarity_threshold: None,
            distance_threshold: None,
        }
    }
}

impl<'g, G, PF, XC, EC, D> McesBuilder<'g, G, PF, XC, EC, D, DefaultRanker> {
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
            partition_orientation_heuristic: self.partition_orientation_heuristic,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
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
            partition_orientation_heuristic: self.partition_orientation_heuristic,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
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
            partition_orientation_heuristic: self.partition_orientation_heuristic,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
        }
    }

    /// Sets a custom clique ranker.
    ///
    /// The ranker is only used to choose among cliques with the same maximum
    /// edge count. In other words, the maximum clique search still optimizes
    /// matched edges first; the ranker only breaks ties afterward.
    ///
    /// For ad-hoc policies, use [`FnRanker`]. For reusable lexicographic
    /// policies, chain rankers with [`CliqueRankerExt::then`].
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
            partition_orientation_heuristic: self.partition_orientation_heuristic,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
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
            partition_orientation_heuristic: self.partition_orientation_heuristic,
            similarity_threshold: self.similarity_threshold,
            distance_threshold: self.distance_threshold,
        }
    }

    /// Enables or disables partition-aware maximum clique search (default:
    /// enabled).
    #[must_use]
    pub fn with_partition(mut self, enabled: bool) -> Self {
        self.use_partition = enabled;
        self
    }

    /// Reorders modular-product vertices before clique search.
    ///
    /// The closure receives:
    /// - the line-graph vertex index from the first graph
    /// - the line-graph vertex index from the second graph
    /// - the original edge endpoints from the first graph
    /// - the original edge endpoints from the second graph
    ///
    /// It must return a lexicographic ordering key. This only affects
    /// search-order-sensitive behavior, such as which tied maxima
    /// [`McesSearchMode::PartialEnumeration`] encounters first.
    #[must_use]
    pub fn with_product_vertex_ordering<F>(mut self, ordering: F) -> Self
    where
        F: FnMut(usize, usize, (usize, usize), (usize, usize)) -> (usize, usize) + 'g,
    {
        self.product_vertex_ordering = Some(Box::new(ordering));
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

    /// Enables or disables RDKit-style partition-side selection.
    ///
    /// When enabled, the partition-aware clique search partitions on the
    /// smaller line graph first. If both line graphs have the same number of
    /// bond vertices, it falls back to the flatter initial partition profile
    /// as a tie-breaker. When disabled, the search always partitions by the
    /// first graph's bond ids.
    #[must_use]
    pub fn with_partition_orientation_heuristic(mut self, enabled: bool) -> Self {
        self.partition_orientation_heuristic = enabled;
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
                );
            }
        }

        // 2. Modular product.
        let mp =
            lg1.graph().modular_product_filtered(lg2.graph(), |i, j| self.pair_filter.filter(i, j));
        let (mp_matrix, mp_vertex_pairs) = reorder_product_for_search(
            mp.matrix().clone(),
            mp.vertex_pairs().to_vec(),
            lg1.edge_map(),
            lg2.edge_map(),
            self.product_vertex_ordering.as_mut(),
        );

        // 3. Maximum cliques (unlabeled: all bonds get label 0).
        let cliques = if self.use_partition {
            let g1_labels = vec![0usize; first_edges];
            let g2_labels = vec![0usize; second_edges];
            let info = PartitionInfo {
                pairs: &mp_vertex_pairs,
                g1_labels: &g1_labels,
                g2_labels: &g2_labels,
                num_labels: 1,
                partition_side: if self.partition_orientation_heuristic {
                    choose_partition_side(&mp_vertex_pairs, g1_labels.len(), g2_labels.len())
                } else {
                    PartitionSide::First
                },
            };
            match self.search_mode {
                McesSearchMode::PartialEnumeration => {
                    accepted_partitioned_cliques(&mp_matrix, &info, self.search_mode, |clique| {
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
                }
                McesSearchMode::AllBest => {
                    let initial_lower_bound = partial_search(
                        &mp_matrix,
                        &info,
                        usize::from(mp_matrix.order() > 0),
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
                    .first()
                    .map_or(0, Vec::len);
                    all_best_search(&mp_matrix, &info, initial_lower_bound, |clique| {
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
                }
            }
        } else {
            accepted_cliques(&mp_matrix, self.search_mode, |clique| {
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

        // 4. Build EagerCliqueInfo for each clique.
        let mut infos: Vec<EagerCliqueInfo<G::NodeId>> = cliques
            .into_iter()
            .map(|c| {
                EagerCliqueInfo::new(
                    c,
                    &mp_vertex_pairs,
                    lg1.edge_map(),
                    lg2.edge_map(),
                    |a, b, c, d| self.disambiguate.disambiguate(a, b, c, d),
                )
            })
            .collect();

        // 5. Rank.
        infos.sort_by(|a, b| self.ranker.compare(a, b));

        // 6. Build result from best clique.
        build_result(infos, first_vertices, first_edges, second_vertices, second_edges)
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
    /// labels, then [`labeled_modular_product_filtered`] with the configured
    /// edge comparator.
    ///
    /// Only bond-label-compatible pairs enter the modular product. The current
    /// bond label is the canonical endpoint node-type pair together with the
    /// original edge value unless [`McesBuilder::with_ignore_edge_values`] is
    /// enabled. Precomputed edge contexts, when provided, further restrict
    /// which original edge pairs are allowed into the product before any
    /// caller-provided pair filter runs.
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

        // 1b. Pre-screening (tier 1).
        // TODO: group by node type for tighter labeled screening.
        // Currently uses unlabeled grouping (all vertices in one group).
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
                );
            }
        }

        // 2. Compute bond labels once and reuse them for vertex-pair filtering
        // and the partition-aware clique bound.
        let g1_bond_labels =
            compute_bond_labels(self.first, lg1.edge_map(), self.ignore_edge_values);
        let g2_bond_labels =
            compute_bond_labels(self.second, lg2.edge_map(), self.ignore_edge_values);
        let (g1_label_indices, g2_label_indices, num_labels) =
            intern_shared_labels(&g1_bond_labels, &g2_bond_labels);
        self.edge_contexts.validate(first_edges, second_edges);

        // 2. Labeled modular product.
        let edge_contexts = &self.edge_contexts;
        let pair_filter = &mut self.pair_filter;
        let edge_comparator = &self.edge_comparator;
        let mp = lg1.graph().labeled_modular_product_filtered(
            lg2.graph(),
            |i, j| {
                g1_label_indices[i] == g2_label_indices[j]
                    && edge_contexts.compatible(i, j)
                    && pair_filter.filter(i, j)
            },
            |a, b| edge_comparator.compare(a, b),
        );
        let (mp_matrix, mp_vertex_pairs) = reorder_product_for_search(
            mp.matrix().clone(),
            mp.vertex_pairs().to_vec(),
            lg1.edge_map(),
            lg2.edge_map(),
            self.product_vertex_ordering.as_mut(),
        );

        // 3. Maximum cliques (label-aware partition bound).
        let cliques = if self.use_partition {
            let info = PartitionInfo {
                pairs: &mp_vertex_pairs,
                g1_labels: &g1_label_indices,
                g2_labels: &g2_label_indices,
                num_labels,
                partition_side: if self.partition_orientation_heuristic {
                    choose_partition_side(
                        &mp_vertex_pairs,
                        g1_label_indices.len(),
                        g2_label_indices.len(),
                    )
                } else {
                    PartitionSide::First
                },
            };
            match self.search_mode {
                McesSearchMode::PartialEnumeration => {
                    accepted_partitioned_cliques(&mp_matrix, &info, self.search_mode, |clique| {
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
                }
                McesSearchMode::AllBest => {
                    let initial_lower_bound = partial_search(
                        &mp_matrix,
                        &info,
                        usize::from(mp_matrix.order() > 0),
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
                    .first()
                    .map_or(0, Vec::len);
                    all_best_search(&mp_matrix, &info, initial_lower_bound, |clique| {
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
                }
            }
        } else {
            accepted_cliques(&mp_matrix, self.search_mode, |clique| {
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

        // 4. Build EagerCliqueInfo for each clique.
        let mut infos: Vec<EagerCliqueInfo<G::NodeId>> = cliques
            .into_iter()
            .map(|c| {
                EagerCliqueInfo::new(
                    c,
                    &mp_vertex_pairs,
                    lg1.edge_map(),
                    lg2.edge_map(),
                    |a, b, c, d| self.disambiguate.disambiguate(a, b, c, d),
                )
            })
            .collect();

        // 5. Rank.
        infos.sort_by(|a, b| self.ranker.compare(a, b));

        // 6. Build result from best clique.
        build_result(infos, first_vertices, first_edges, second_vertices, second_edges)
    }
}

/// Constructs an `McesResult` from ranked clique infos.
fn build_result<N>(
    infos: Vec<EagerCliqueInfo<N>>,
    first_graph_vertices: usize,
    first_graph_edges: usize,
    second_graph_vertices: usize,
    second_graph_edges: usize,
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
    }
}

#[cfg(test)]
mod tests {
    use super::intern_shared_labels;

    #[test]
    fn test_intern_shared_labels_reuses_equal_labels() {
        let first = [(0_u8, 7_u8, 1_u8), (1, 9, 2)];
        let second = [(1_u8, 9_u8, 2_u8), (0, 11, 2)];

        let (first_indices, second_indices, num_labels) = intern_shared_labels(&first, &second);

        assert_eq!(num_labels, 3);
        assert_eq!(first_indices[1], second_indices[0]);
        assert_ne!(first_indices[0], second_indices[1]);
    }
}
