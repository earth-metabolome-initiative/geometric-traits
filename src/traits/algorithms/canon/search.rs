//! First individualization-refinement canonizer for simple undirected labeled
//! graphs.
//!
//! This implementation intentionally starts small:
//!
//! - seed the partition from vertex labels
//! - refine to the stable labeled equitable partition
//! - depth-first individualization on residual non-singleton cells
//! - compare leaves using a deterministic graph certificate
//! - record a refinement trace for future `bliss`-style subtree comparison
//! - sibling-level orbit pruning from discovered automorphisms
//!
//! It still does **not** yet implement the full richer pruning machinery
//! present in `bliss`, such as failure recording and the remaining streamed
//! certificate / bad-node logic. It does include a partial `bliss`-style
//! component-recursion path.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

use num_traits::AsPrimitive;

use crate::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::GenericVocabularyBuilder,
    traits::{MonoplexGraph, SparseValuedMatrix2D, VocabularyBuilder},
};

use super::{
    BacktrackableOrderedPartition, PartitionCellId, RefinementTrace,
    refine_partition_to_labeled_equitable_with_trace,
    refine_partition_to_labeled_equitable_with_trace_from_splitters,
};
use crate::traits::MonoplexMonopartiteGraph;

/// Splitting heuristic used to choose the next non-singleton cell to
/// individualize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanonSplittingHeuristic {
    /// The first non-singleton cell in current cell order.
    First,
    /// The first smallest non-singleton cell.
    FirstSmallest,
    /// The first largest non-singleton cell.
    FirstLargest,
    /// The first non-singleton cell maximizing the number of partially
    /// connected non-singleton neighbour cells.
    FirstMaxNeighbours,
    /// The first smallest non-singleton cell maximizing the number of
    /// partially connected non-singleton neighbour cells.
    #[default]
    FirstSmallestMaxNeighbours,
    /// The first largest non-singleton cell maximizing the number of
    /// partially connected non-singleton neighbour cells.
    FirstLargestMaxNeighbours,
}

/// Canonical certificate for a simple undirected graph with vertex and edge
/// labels.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel> {
    /// Vertex labels in canonical order.
    pub vertex_labels: Vec<VertexLabel>,
    /// Edge labels over the upper triangle in canonical order.
    ///
    /// `None` denotes the absence of an edge.
    pub upper_triangle_edge_labels: Vec<Option<EdgeLabel>>,
}

/// Result of canonical labeling for a simple undirected labeled graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalLabelingResult<VertexLabel, EdgeLabel> {
    /// Original dense vertex identifiers in canonical order.
    pub order: Vec<usize>,
    /// The canonical graph certificate induced by `order`.
    pub certificate: LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>,
    /// Search statistics collected while finding the canonical labeling.
    pub stats: CanonicalSearchStats,
}

/// Search statistics for the current individualization-refinement canonizer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CanonicalSearchStats {
    /// Number of search-tree nodes visited, including the root and leaves.
    pub search_nodes: usize,
    /// Number of discrete leaves reached.
    pub leaf_nodes: usize,
    /// Number of automorphisms inferred from equal canonical leaves.
    pub automorphisms_found: usize,
    /// Number of sibling branch choices skipped because they fell into a known
    /// orbit discovered earlier at the same search node.
    pub pruned_sibling_orbits: usize,
    /// Number of root-branch choices skipped because they fell into a known
    /// orbit discovered earlier in the search.
    pub pruned_root_orbits: usize,
    /// Number of search nodes pruned because their path signature was already
    /// worse than the best leaf found so far.
    pub pruned_path_signatures: usize,
}

/// Options for the current canonizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CanonicalLabelingOptions {
    /// Splitting heuristic for selecting the next target cell.
    pub splitting_heuristic: CanonSplittingHeuristic,
}

#[derive(Clone, Debug)]
struct KnownOrbits {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl KnownOrbits {
    fn new(order: usize) -> Self {
        Self { parents: (0..order).collect(), ranks: vec![0; order] }
    }

    fn same_set(&mut self, left: usize, right: usize) -> bool {
        self.find(left) == self.find(right)
    }

    fn is_minimal_representative(&mut self, element: usize) -> bool {
        let root = self.find(element);
        (0..self.parents.len())
            .all(|candidate| self.find(candidate) != root || candidate >= element)
    }

    fn ingest_leaf_automorphism(&mut self, left_order: &[usize], right_order: &[usize]) {
        debug_assert_eq!(left_order.len(), right_order.len());
        for (&left, &right) in left_order.iter().zip(right_order.iter()) {
            self.union(left, right);
        }
    }

    fn find(&mut self, element: usize) -> usize {
        let parent = self.parents[element];
        if parent == element {
            return element;
        }
        let root = self.find(parent);
        self.parents[element] = root;
        root
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.ranks[left_root] < self.ranks[right_root] {
            core::mem::swap(&mut left_root, &mut right_root);
        }
        self.parents[right_root] = left_root;
        if self.ranks[left_root] == self.ranks[right_root] {
            self.ranks[left_root] = self.ranks[left_root].saturating_add(1);
        }
    }

    fn ingest_known_orbits(&mut self, other: &mut Self) {
        let order = other.parents.len();
        for left in 0..order {
            for right in (left + 1)..order {
                if other.same_set(left, right) {
                    self.union(left, right);
                }
            }
        }
    }

    fn ingest_automorphism(&mut self, automorphism: &[usize]) {
        debug_assert_eq!(self.parents.len(), automorphism.len());
        for (vertex, &image) in automorphism.iter().enumerate() {
            self.union(vertex, image);
        }
    }
}

struct SearchState<VertexLabel, EdgeLabel> {
    stats: CanonicalSearchStats,
    first_order: Option<Vec<usize>>,
    first_certificate: Option<LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>>,
    first_path_invariants: Option<Vec<RefinementTrace<EdgeLabel>>>,
    first_choice_path: Option<Vec<usize>>,
    best_order: Option<Vec<usize>>,
    best_certificate: Option<LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>>,
    best_path_invariants: Option<Vec<RefinementTrace<EdgeLabel>>>,
    best_choice_path: Option<Vec<usize>>,
    first_path_orbits_global: KnownOrbits,
    first_path_orbits_by_depth: Vec<KnownOrbits>,
    best_path_orbits: KnownOrbits,
    long_prune_records: Vec<LongPruneRecord>,
}

struct SearchOutcome<VertexLabel, EdgeLabel> {
    result: CanonicalLabelingResult<VertexLabel, EdgeLabel>,
    path_invariants: Vec<RefinementTrace<EdgeLabel>>,
    sibling_orbits: KnownOrbits,
    choice_path: Vec<usize>,
}

struct SearchReturn<VertexLabel, EdgeLabel> {
    best: Option<SearchOutcome<VertexLabel, EdgeLabel>>,
    first_path_automorphism: Option<Vec<usize>>,
    best_path_backjump_depth: Option<usize>,
}

#[derive(Clone, Debug)]
struct LongPruneRecord {
    fixed: Vec<bool>,
    mcrs: Vec<bool>,
}

#[derive(Clone, Debug)]
struct ComponentEndpoint {
    discrete_cell_limit: usize,
    first_checked: bool,
    best_checked: bool,
    creation_choice_path: Vec<usize>,
    created_on_best_path: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeSubdivisionVertexLabel<VertexLabel, EdgeLabel> {
    Original(VertexLabel),
    Edge(EdgeLabel),
}

/// Computes a canonical labeling for a simple undirected graph with total-order
/// vertex and edge labels.
///
/// This is the first search-based canonizer in the crate. It assumes:
///
/// - dense node identifiers `0..graph.number_of_nodes()`
/// - simple undirected structure
/// - every vertex has a label
/// - every present edge has a label
///
/// The implementation is correct but intentionally smaller than `bliss`. It
/// currently uses queue-based equitable refinement, partition-independent
/// initial invariants, a `bliss`-style target-cell heuristic family, partial
/// component recursion, and orbit pruning inferred from equal child
/// certificates and first-path matches.
#[must_use]
pub fn canonical_label_labeled_simple_graph<G, VertexLabel, EdgeLabel, VF, EF>(
    graph: &G,
    mut vertex_label: VF,
    mut edge_label: EF,
) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    VF: FnMut(G::NodeId) -> VertexLabel,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    canonical_label_labeled_simple_graph_with_options(
        graph,
        &mut vertex_label,
        &mut edge_label,
        CanonicalLabelingOptions::default(),
    )
}

/// Computes a canonical labeling for a simple undirected graph with total-order
/// vertex and edge labels using explicit canonizer options.
#[must_use]
pub fn canonical_label_labeled_simple_graph_with_options<G, VertexLabel, EdgeLabel, VF, EF>(
    graph: &G,
    mut vertex_label: VF,
    mut edge_label: EF,
    options: CanonicalLabelingOptions,
) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    VF: FnMut(G::NodeId) -> VertexLabel,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let reduced =
        reduce_edge_labeled_graph_to_vertex_labeled(graph, &mut vertex_label, &mut edge_label);
    let reduced_matrix = crate::traits::Edges::matrix(reduced.graph.edges());
    let reduced_result = canonical_label_labeled_simple_graph_core(
        &reduced.graph,
        |node| reduced.vertex_labels[node].clone(),
        |left, right| reduced_matrix.sparse_value_at(left, right).unwrap(),
        options,
    );
    let order = reduced_result
        .order
        .into_iter()
        .filter(|&vertex| vertex < reduced.original_vertex_count)
        .collect::<Vec<_>>();
    let certificate = build_certificate(
        graph,
        &reduced.original_nodes,
        &reduced.original_vertex_labels,
        &mut edge_label,
        &order,
    );

    CanonicalLabelingResult { order, certificate, stats: reduced_result.stats }
}

fn canonical_label_labeled_simple_graph_core<G, VertexLabel, EdgeLabel, VF, EF>(
    graph: &G,
    mut vertex_label: VF,
    mut edge_label: EF,
    options: CanonicalLabelingOptions,
) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    VF: FnMut(G::NodeId) -> VertexLabel,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let order = graph.number_of_nodes().as_();
    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), order);
    debug_assert!(nodes.iter().enumerate().all(|(index, node)| node.as_() == index));

    let vertex_labels = nodes.iter().copied().map(&mut vertex_label).collect::<Vec<_>>();
    let mut partition = BacktrackableOrderedPartition::new(order);
    if order > 1 {
        let label_ids = dense_label_ids(vertex_labels.iter().cloned());
        let degrees =
            nodes.iter().copied().map(|node| graph.successors(node).count()).collect::<Vec<_>>();
        let _ =
            refine_partition_according_to_unsigned_invariant_like_bliss(&mut partition, |vertex| {
                label_ids[&vertex_labels[vertex]]
            });
        let _ =
            refine_partition_according_to_unsigned_invariant_like_bliss(&mut partition, |vertex| {
                degrees[vertex]
            });
    }
    let (_, root_trace) =
        refine_partition_to_labeled_equitable_with_trace(graph, &mut partition, |left, right| {
            edge_label(left, right)
        });
    let root_was_discrete = partition.is_discrete();
    let mut state = SearchState {
        stats: CanonicalSearchStats::default(),
        first_order: None,
        first_certificate: None,
        first_path_invariants: None,
        first_choice_path: None,
        best_order: None,
        best_certificate: None,
        best_path_invariants: None,
        best_choice_path: None,
        first_path_orbits_global: KnownOrbits::new(order),
        first_path_orbits_by_depth: Vec::new(),
        best_path_orbits: KnownOrbits::new(order),
        long_prune_records: Vec::new(),
    };
    let mut path_invariants = vec![root_trace];
    let mut choice_path = Vec::new();
    let mut component_endpoints = Vec::new();
    let search_return = search_canonical_labeling(
        graph,
        &nodes,
        &vertex_labels,
        &mut edge_label,
        &mut partition,
        &mut state,
        &mut path_invariants,
        &mut choice_path,
        &mut component_endpoints,
        0,
        true,
        false,
        true,
        0,
        options.splitting_heuristic,
    );
    let outcome = search_return
        .best
        .expect("a canonizer over any finite graph must produce at least one leaf");
    if !root_was_discrete {
        state.stats.leaf_nodes += 1;
    }
    finish_result(outcome.result, state.stats)
}

type ReducedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, ()>>;
type ReducedGraph = GenericGraph<SortedVec<usize>, ReducedEdges>;

struct ReducedEdgeLabeledGraph<VertexLabel, EdgeLabel, NodeId> {
    graph: ReducedGraph,
    vertex_labels: Vec<EdgeSubdivisionVertexLabel<VertexLabel, EdgeLabel>>,
    original_nodes: Vec<NodeId>,
    original_vertex_labels: Vec<VertexLabel>,
    original_vertex_count: usize,
}

fn reduce_edge_labeled_graph_to_vertex_labeled<G, VertexLabel, EdgeLabel, VF, EF>(
    graph: &G,
    vertex_label: &mut VF,
    edge_label: &mut EF,
) -> ReducedEdgeLabeledGraph<VertexLabel, EdgeLabel, G::NodeId>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    VF: FnMut(G::NodeId) -> VertexLabel,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let original_nodes = graph.node_ids().collect::<Vec<_>>();
    let original_vertex_count = original_nodes.len();
    let original_vertex_labels =
        original_nodes.iter().copied().map(vertex_label).collect::<Vec<_>>();
    let mut normalized_edges = Vec::new();
    for source in 0..original_vertex_count {
        let source_node = original_nodes[source];
        for destination_node in graph.successors(source_node) {
            let destination = destination_node.as_();
            if source >= destination {
                continue;
            }
            normalized_edges.push((source, destination, edge_label(source_node, destination_node)));
        }
    }
    normalized_edges.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });

    let mut reduced_vertex_labels = original_vertex_labels
        .iter()
        .cloned()
        .map(EdgeSubdivisionVertexLabel::Original)
        .collect::<Vec<_>>();
    let mut reduced_edges = Vec::new();
    for (edge_index, &(source, destination, ref label)) in normalized_edges.iter().enumerate() {
        let edge_vertex = original_vertex_count + edge_index;
        reduced_vertex_labels.push(EdgeSubdivisionVertexLabel::Edge(label.clone()));
        reduced_edges.push((source, edge_vertex, ()));
        reduced_edges.push((destination, edge_vertex, ()));
    }

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(reduced_vertex_labels.len())
        .symbols((0..reduced_vertex_labels.len()).enumerate())
        .build()
        .expect("dense reduced graph vocabulary should build");
    reduced_edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    let graph = GenericGraph::from((
        nodes,
        SymmetricCSR2D::from_sorted_upper_triangular_entries(
            reduced_vertex_labels.len(),
            reduced_edges,
        )
        .expect("edge-subdivision reduction should produce a simple graph"),
    ));

    ReducedEdgeLabeledGraph {
        graph,
        vertex_labels: reduced_vertex_labels,
        original_nodes,
        original_vertex_labels,
        original_vertex_count,
    }
}

fn search_canonical_labeling<G, VertexLabel, EdgeLabel, EF>(
    graph: &G,
    nodes: &[G::NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    partition: &mut BacktrackableOrderedPartition,
    state: &mut SearchState<VertexLabel, EdgeLabel>,
    path_invariants: &mut Vec<RefinementTrace<EdgeLabel>>,
    choice_path: &mut Vec<usize>,
    component_endpoints: &mut Vec<ComponentEndpoint>,
    active_component_endpoint_len: usize,
    current_node_is_on_first_path: bool,
    current_node_is_on_best_path: bool,
    count_current_node: bool,
    depth: usize,
    splitting_heuristic: CanonSplittingHeuristic,
) -> SearchReturn<VertexLabel, EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    if count_current_node {
        state.stats.search_nodes += 1;
    }

    if partition.is_discrete() {
        if let Some(best_path) = state.best_path_invariants.as_deref() {
            if depth == 1
                && !current_node_is_on_first_path
                && !current_node_is_on_best_path
                && path_invariants_prefix_cmp(path_invariants, best_path)
                    == core::cmp::Ordering::Less
            {
                state.stats.pruned_path_signatures += 1;
                return SearchReturn {
                    best: None,
                    first_path_automorphism: None,
                    best_path_backjump_depth: None,
                };
            }
        }
        state.stats.leaf_nodes += 1;
        let order = partition
            .cells()
            .map(|cell| {
                debug_assert!(cell.is_unit());
                cell.elements()[0]
            })
            .collect::<Vec<_>>();
        let certificate = build_certificate(graph, nodes, vertex_labels, edge_label, &order);
        let leaf_order = order.clone();
        let result =
            CanonicalLabelingResult { order, certificate, stats: CanonicalSearchStats::default() };
        let had_first_certificate = state.first_certificate.is_some();
        let candidate_equals_first =
            state.first_certificate.as_ref().is_some_and(|first| result.certificate == *first);
        let comparison_to_best = compare_candidate_to_best(
            path_invariants,
            &result.certificate,
            state.best_path_invariants.as_deref(),
            state.best_certificate.as_ref(),
        );
        if state.first_certificate.is_none() {
            state.first_order = Some(result.order.clone());
            state.first_certificate = Some(result.certificate.clone());
            state.first_path_invariants = Some(path_invariants.clone());
            state.first_choice_path = Some(choice_path.clone());
        }
        if comparison_to_best == core::cmp::Ordering::Greater {
            state.best_order = Some(result.order.clone());
            state.best_certificate = Some(result.certificate.clone());
            state.best_path_invariants = Some(path_invariants.clone());
            state.best_choice_path = Some(choice_path.clone());
            state.best_path_orbits = KnownOrbits::new(nodes.len());
        }
        return SearchReturn {
            best: Some(SearchOutcome {
                result,
                path_invariants: path_invariants.clone(),
                sibling_orbits: KnownOrbits::new(nodes.len()),
                choice_path: choice_path.clone(),
            }),
            first_path_automorphism: if had_first_certificate
                && candidate_equals_first
                && !current_node_is_on_first_path
            {
                state
                    .first_order
                    .as_ref()
                    .map(|first_order| leaf_automorphism(first_order, &leaf_order))
            } else {
                None
            },
            best_path_backjump_depth: None,
        };
    }

    let node_backtrack_point = partition.set_backtrack_point();
    let previous_component_endpoint_len = component_endpoints.len();
    let mut active_component_endpoint_len = active_component_endpoint_len;
    let mut node_is_on_first_path = current_node_is_on_first_path
        || choice_path_is_prefix_of(state.first_choice_path.as_deref(), choice_path);
    let mut node_is_on_best_path = current_node_is_on_best_path
        || choice_path_is_prefix_of(state.best_choice_path.as_deref(), choice_path);
    let target_cell = prepare_component_recursion_and_choose_target_cell(
        graph,
        nodes,
        partition,
        splitting_heuristic,
        node_is_on_best_path,
        choice_path,
        component_endpoints,
        &mut active_component_endpoint_len,
    );
    let candidate_choices = candidate_split_elements(
        graph,
        edge_label,
        partition,
        target_cell,
        path_invariants.as_slice(),
        state.best_path_invariants.as_deref(),
        &mut state.stats,
    );
    let long_prune_redundant = if !node_is_on_first_path && depth >= 1 {
        compute_long_prune_redundant(
            choice_path.as_slice(),
            &candidate_choices,
            &state.long_prune_records,
        )
    } else {
        BTreeSet::new()
    };
    let mut local_orbits = KnownOrbits::new(nodes.len());
    let mut explored_choices = Vec::new();
    let mut best: Option<SearchOutcome<VertexLabel, EdgeLabel>> = None;
    let mut best_choice: Option<usize> = None;
    let mut node_first_path_automorphism: Option<Vec<usize>> = None;
    let target_cell_len = partition.cell_len(target_cell);
    if let Some(best_path) = state.best_path_invariants.as_deref() {
        if target_cell_len != 2
            && path_invariants_prefix_cmp(path_invariants, best_path) == core::cmp::Ordering::Less
        {
            state.stats.pruned_path_signatures += 1;
            partition.goto_backtrack_point(node_backtrack_point);
            component_endpoints.truncate(previous_component_endpoint_len);
            return SearchReturn {
                best: None,
                first_path_automorphism: None,
                best_path_backjump_depth: None,
            };
        }
    }

    for (candidate_index, element) in candidate_choices.iter().copied().enumerate() {
        node_is_on_first_path = current_node_is_on_first_path
            || choice_path_is_prefix_of(state.first_choice_path.as_deref(), choice_path);
        node_is_on_best_path = current_node_is_on_best_path
            || choice_path_is_prefix_of(state.best_choice_path.as_deref(), choice_path);
        if state.first_choice_path.is_some()
            && node_is_on_first_path
            && !state.first_path_orbits_global.is_minimal_representative(element)
        {
            state.stats.pruned_sibling_orbits += 1;
            if depth == 0 {
                state.stats.pruned_root_orbits += 1;
            }
            continue;
        }

        if state.best_choice_path.is_some()
            && node_is_on_best_path
            && !state.best_path_orbits.is_minimal_representative(element)
        {
            state.stats.pruned_sibling_orbits += 1;
            if depth == 0 {
                state.stats.pruned_root_orbits += 1;
            }
            continue;
        }

        if long_prune_redundant.contains(&element) {
            state.stats.pruned_sibling_orbits += 1;
            continue;
        }

        if explored_choices.iter().any(|&previous| local_orbits.same_set(previous, element)) {
            state.stats.pruned_sibling_orbits += 1;
            if depth == 0 {
                state.stats.pruned_root_orbits += 1;
            }
            continue;
        }

        explored_choices.push(element);
        let backtrack_point = partition.set_backtrack_point();
        let individualized = partition.individualize(target_cell, element);
        let (_, child_trace) = refine_partition_to_labeled_equitable_with_trace_from_splitters(
            graph,
            partition,
            |left, right| edge_label(left, right),
            [individualized],
        );
        path_invariants.push(child_trace);
        choice_path.push(element);
        let had_first_path_before_child = state.first_choice_path.is_some();
        let had_best_before_child = state.best_certificate.is_some();
        let previous_best_order = state.best_order.clone();
        let previous_best_path_invariants = state.best_path_invariants.clone();
        let previous_best_choice_path = state.best_choice_path.clone();
        let child_path_matches_first_prefix =
            choice_path_is_prefix_of(state.first_choice_path.as_deref(), choice_path);
        let child_is_on_first_path = node_is_on_first_path && child_path_matches_first_prefix;
        let child_path_is_equal_to_first_prefix =
            state.first_path_invariants.as_ref().is_some_and(|first_path_invariants| {
                path_invariants_prefix_equal_strict(path_invariants, first_path_invariants)
            });
        let child_path_cmp_to_best_prefix =
            state.best_path_invariants.as_ref().map(|best_path_invariants| {
                path_invariants_prefix_cmp(path_invariants, best_path_invariants)
            });
        let child_path_is_equal_to_best_prefix =
            child_path_cmp_to_best_prefix.is_some_and(|cmp| cmp == core::cmp::Ordering::Equal);
        let child_path_is_not_worse_than_best_prefix =
            child_path_cmp_to_best_prefix.is_some_and(|cmp| cmp != core::cmp::Ordering::Less);
        let child_is_on_best_path = node_is_on_best_path
            && choice_path_is_prefix_of(state.best_choice_path.as_deref(), choice_path);
        let local_best_matches_first_path = best.as_ref().is_some_and(|current_best| {
            state
                .first_certificate
                .as_ref()
                .is_some_and(|first| current_best.result.certificate == *first)
        });
        let mut descended_via_first_component_boundary = false;
        let mut break_after_component_endpoint_automorphism = false;
        let mut component_endpoint_first_path_automorphism: Option<Vec<usize>> = None;
        let reached_component_endpoint =
            active_component_endpoint_mut(component_endpoints, active_component_endpoint_len)
                .is_some_and(|endpoint| {
                    if partition.is_discrete()
                        || partition.number_of_discrete_cells() != endpoint.discrete_cell_limit
                    {
                        return false;
                    }
                    let mut continue_to_next_component = false;
                    let endpoint_created_on_first_path =
                        state.first_choice_path.as_ref().is_some_and(|first_choice_path| {
                            endpoint.creation_choice_path.len() <= first_choice_path.len()
                                && endpoint
                                    .creation_choice_path
                                    .iter()
                                    .zip(first_choice_path.iter())
                                    .all(|(left, right)| left == right)
                        });
                    if state.first_certificate.is_none() || child_path_is_equal_to_first_prefix {
                        if !endpoint.first_checked {
                            endpoint.first_checked = true;
                            if depth > 0 {
                                state.stats.search_nodes += 1;
                            }
                            continue_to_next_component = true;
                        } else if endpoint_created_on_first_path && !child_is_on_first_path {
                            if let Some(first_choice_path) = state.first_choice_path.as_ref() {
                                let automorphism = state.first_order.as_ref().map(|first_order| {
                                    first_path_automorphism_for_current_partition(
                                        partition,
                                        first_order,
                                    )
                                });
                                if let Some((
                                    first_difference_depth,
                                    (&current_choice, &first_choice),
                                )) = choice_path
                                    .iter()
                                    .zip(first_choice_path.iter())
                                    .enumerate()
                                    .find(|(_, (left, right))| left != right)
                                {
                                    state.stats.search_nodes += 1;
                                    first_path_orbits_at_depth_mut(
                                        state,
                                        first_difference_depth,
                                        nodes.len(),
                                    )
                                    .union(current_choice, first_choice);
                                    if let Some(automorphism) = automorphism.as_deref() {
                                        first_path_orbits_at_depth_mut(
                                            state,
                                            first_difference_depth,
                                            nodes.len(),
                                        )
                                        .ingest_automorphism(automorphism);
                                        state
                                            .first_path_orbits_global
                                            .ingest_automorphism(automorphism);
                                        local_orbits.ingest_automorphism(automorphism);
                                        component_endpoint_first_path_automorphism =
                                            Some(automorphism.to_vec());
                                    } else if first_difference_depth == 0 {
                                        state
                                            .first_path_orbits_global
                                            .union(current_choice, first_choice);
                                    }
                                }
                            }
                            state.stats.automorphisms_found += 1;
                            break_after_component_endpoint_automorphism = true;
                            return true;
                        }
                    }
                    if child_path_is_not_worse_than_best_prefix {
                        if !endpoint.best_checked {
                            endpoint.best_checked = true;
                            continue_to_next_component = true;
                        } else if endpoint.created_on_best_path
                            && !current_node_is_on_best_path
                            && child_path_is_equal_to_best_prefix
                        {
                            state.stats.automorphisms_found += 1;
                            return true;
                        }
                    }
                    if continue_to_next_component {
                        descended_via_first_component_boundary = true;
                    }
                    false
                });
        if reached_component_endpoint {
            choice_path.pop();
            path_invariants.pop();
            partition.goto_backtrack_point(backtrack_point);
            if break_after_component_endpoint_automorphism {
                if node_is_on_first_path {
                    continue;
                }
                node_first_path_automorphism = component_endpoint_first_path_automorphism;
                break;
            }
            continue;
        }
        let early_component_first_path_automorphism = !node_is_on_first_path
            && active_component_endpoint(component_endpoints, active_component_endpoint_len)
                .is_some_and(|endpoint| endpoint.first_checked && local_best_matches_first_path)
            && !partition.is_discrete()
            && child_path_is_equal_to_first_prefix
            && partition.non_singleton_cells().count() == 1
            && target_cell_len == 2;
        let child_active_component_endpoint_len =
            if descended_via_first_component_boundary && active_component_endpoint_len > 1 {
                active_component_endpoint_len.saturating_sub(1)
            } else {
                active_component_endpoint_len
            };
        if early_component_first_path_automorphism {
            if let Some(first_choice_path) = state.first_choice_path.as_ref() {
                state.stats.automorphisms_found += 1;
                if let Some((first_difference_depth, (&current_choice, &first_choice))) =
                    choice_path
                        .iter()
                        .zip(first_choice_path.iter())
                        .enumerate()
                        .find(|(_, (left, right))| left != right)
                {
                    first_path_orbits_at_depth_mut(state, first_difference_depth, nodes.len())
                        .union(current_choice, first_choice);
                    if first_difference_depth == 0 {
                        state.first_path_orbits_global.union(current_choice, first_choice);
                    }
                    local_orbits.union(current_choice, first_choice);
                }
            }
            choice_path.pop();
            path_invariants.pop();
            partition.goto_backtrack_point(backtrack_point);
            break;
        }
        let child_return = search_canonical_labeling(
            graph,
            nodes,
            vertex_labels,
            edge_label,
            partition,
            state,
            path_invariants,
            choice_path,
            component_endpoints,
            child_active_component_endpoint_len,
            if descended_via_first_component_boundary { false } else { child_is_on_first_path },
            if descended_via_first_component_boundary { false } else { child_is_on_best_path },
            depth == 0 || !descended_via_first_component_boundary,
            depth + 1,
            splitting_heuristic,
        );
        let child_first_path_automorphism = child_return.first_path_automorphism.clone();
        let child_best_path_backjump_depth = child_return.best_path_backjump_depth;
        let candidate_equals_first_path = child_return.best.as_ref().is_some_and(|candidate| {
            state
                .first_certificate
                .as_ref()
                .is_some_and(|first| candidate.result.certificate == *first)
        });
        choice_path.pop();
        path_invariants.pop();
        partition.goto_backtrack_point(backtrack_point);
        let Some(candidate) = child_return.best else {
            if let Some(automorphism) = child_first_path_automorphism.as_deref() {
                first_path_orbits_at_depth_mut(state, depth, nodes.len())
                    .ingest_automorphism(automorphism);
                state.first_path_orbits_global.ingest_automorphism(automorphism);
                state.long_prune_records.push(long_prune_record_from_automorphism(automorphism));
                local_orbits.ingest_automorphism(automorphism);
                if !node_is_on_first_path {
                    node_first_path_automorphism = child_first_path_automorphism;
                    break;
                }
            }
            if let Some(backjump_depth) = child_best_path_backjump_depth {
                if backjump_depth < depth {
                    partition.goto_backtrack_point(node_backtrack_point);
                    component_endpoints.truncate(previous_component_endpoint_len);
                    return SearchReturn {
                        best: best
                            .map(|best| SearchOutcome { sibling_orbits: local_orbits, ..best }),
                        first_path_automorphism: if node_is_on_first_path {
                            None
                        } else {
                            node_first_path_automorphism
                        },
                        best_path_backjump_depth: Some(backjump_depth),
                    };
                }
                if backjump_depth == depth {
                    continue;
                }
            }
            continue;
        };
        if let Some(backjump_depth) = child_best_path_backjump_depth {
            if backjump_depth < depth {
                partition.goto_backtrack_point(node_backtrack_point);
                component_endpoints.truncate(previous_component_endpoint_len);
                return SearchReturn {
                    best: best.map(|best| SearchOutcome { sibling_orbits: local_orbits, ..best }),
                    first_path_automorphism: if node_is_on_first_path {
                        None
                    } else {
                        node_first_path_automorphism
                    },
                    best_path_backjump_depth: Some(backjump_depth),
                };
            }
            if backjump_depth == depth {
                continue;
            }
        }
        let mut candidate_sibling_orbits = candidate.sibling_orbits.clone();
        local_orbits.ingest_known_orbits(&mut candidate_sibling_orbits);

        let candidate_matches_first_path_automorphism = candidate_equals_first_path
            && had_first_path_before_child
            && !child_is_on_first_path
            && (child_path_is_equal_to_first_prefix || component_endpoints.is_empty());
        if candidate_matches_first_path_automorphism {
            if let Some(first_choice_path) = state.first_choice_path.as_ref().cloned() {
                state.stats.automorphisms_found += 1;
                if let Some(&first_choice) = first_choice_path.get(depth) {
                    let first_order = state.first_order.as_ref().cloned();
                    {
                        let first_path_orbits =
                            first_path_orbits_at_depth_mut(state, depth, nodes.len());
                        first_path_orbits.union(element, first_choice);
                        if let Some(first_order) = first_order.as_deref() {
                            first_path_orbits
                                .ingest_leaf_automorphism(first_order, &candidate.result.order);
                            local_orbits
                                .ingest_leaf_automorphism(first_order, &candidate.result.order);
                        }
                    }
                    state.first_path_orbits_global.union(element, first_choice);
                    if let Some(first_order) = first_order.as_deref() {
                        state
                            .first_path_orbits_global
                            .ingest_leaf_automorphism(first_order, &candidate.result.order);
                    }
                }
                if component_endpoints.is_empty() {}
            }
        }
        let candidate_first_path_automorphism = child_first_path_automorphism.or_else(|| {
            if candidate_matches_first_path_automorphism {
                state
                    .first_order
                    .as_ref()
                    .map(|first_order| leaf_automorphism(first_order, &candidate.result.order))
            } else {
                None
            }
        });
        let prune_remaining_siblings_after_first_path_match = if component_endpoints.is_empty() {
            candidate_equals_first_path
                && child_path_is_equal_to_first_prefix
                && candidate_choices[(candidate_index + 1)..].iter().copied().all(|remaining| {
                    explored_choices
                        .iter()
                        .any(|&previous| local_orbits.same_set(previous, remaining))
                        || (state.first_choice_path.is_some()
                            && node_is_on_first_path
                            && !state.first_path_orbits_global.is_minimal_representative(remaining))
                })
        } else {
            candidate_equals_first_path
                && child_path_is_equal_to_first_prefix
                && !node_is_on_first_path
        };
        let comparison_to_local_best = compare_candidate_to_best(
            &candidate.path_invariants,
            &candidate.result.certificate,
            best.as_ref().map(|current_best| current_best.path_invariants.as_slice()),
            best.as_ref().map(|current_best| &current_best.result.certificate),
        );
        let candidate_matches_previous_best_path =
            previous_best_path_invariants.as_ref().is_some_and(|best_path_invariants| {
                path_invariants_lex_cmp(&candidate.path_invariants, best_path_invariants)
                    == core::cmp::Ordering::Equal
            });
        let candidate_matches_previous_best_automorphism = previous_best_order
            .as_ref()
            .map(|best_order| leaf_automorphism(best_order, &candidate.result.order))
            .filter(|automorphism| {
                is_labeled_graph_automorphism(graph, nodes, vertex_labels, edge_label, automorphism)
            });
        let candidate_equals_local_best_certificate = best.as_ref().is_some_and(|current_best| {
            current_best.result.certificate == candidate.result.certificate
        });

        if candidate_equals_local_best_certificate {
            let current_best =
                best.as_ref().expect("equal local-best certificates require an existing best");
            state.stats.automorphisms_found += 1;
            local_orbits
                .union(best_choice.expect("equal sibling branches require a best choice"), element);
            local_orbits
                .ingest_leaf_automorphism(&current_best.result.order, &candidate.result.order);
        }

        if (candidate_matches_previous_best_path
            || candidate_matches_previous_best_automorphism.is_some())
            && had_best_before_child
            && !child_is_on_best_path
        {
            if let (Some(best_order), Some(best_choice_path)) =
                (previous_best_order.as_ref(), previous_best_choice_path.as_ref())
            {
                state.stats.automorphisms_found += 1;
                let automorphism = candidate_matches_previous_best_automorphism
                    .unwrap_or_else(|| leaf_automorphism(best_order, &candidate.result.order));
                state.best_path_orbits.ingest_automorphism(&automorphism);
                state.first_path_orbits_global.ingest_automorphism(&automorphism);
                state.long_prune_records.push(long_prune_record_from_automorphism(&automorphism));
                local_orbits.ingest_automorphism(&automorphism);

                let gca_with_first =
                    common_prefix_len(&candidate.choice_path, state.first_choice_path.as_deref());
                let gca_with_best =
                    common_prefix_len(&candidate.choice_path, Some(best_choice_path.as_slice()));
                let backjump_depth = if gca_with_first < candidate.choice_path.len()
                    && !state
                        .first_path_orbits_global
                        .is_minimal_representative(candidate.choice_path[gca_with_first])
                {
                    Some(gca_with_first)
                } else if gca_with_best < candidate.choice_path.len()
                    && !state
                        .best_path_orbits
                        .is_minimal_representative(candidate.choice_path[gca_with_best])
                {
                    Some(gca_with_best)
                } else {
                    None
                };

                if let Some(backjump_depth) = backjump_depth {
                    if backjump_depth < depth {
                        partition.goto_backtrack_point(node_backtrack_point);
                        component_endpoints.truncate(previous_component_endpoint_len);
                        return SearchReturn {
                            best: best
                                .map(|best| SearchOutcome { sibling_orbits: local_orbits, ..best }),
                            first_path_automorphism: if node_is_on_first_path {
                                None
                            } else {
                                node_first_path_automorphism
                            },
                            best_path_backjump_depth: Some(backjump_depth),
                        };
                    }
                    if backjump_depth == depth {
                        continue;
                    }
                }
            }
        }

        match (&mut best, comparison_to_local_best) {
            (_, core::cmp::Ordering::Greater) | (None, _) => {
                best = Some(candidate);
                best_choice = Some(element);
            }
            _ => {}
        }

        if let Some(automorphism) = candidate_first_path_automorphism.as_deref() {
            if !candidate_matches_first_path_automorphism {
                first_path_orbits_at_depth_mut(state, depth, nodes.len())
                    .ingest_automorphism(automorphism);
                state.first_path_orbits_global.ingest_automorphism(automorphism);
                local_orbits.ingest_automorphism(automorphism);
            }
            if !node_is_on_first_path {
                node_first_path_automorphism = candidate_first_path_automorphism;
                break;
            }
        }

        if prune_remaining_siblings_after_first_path_match {
            break;
        }
    }

    partition.goto_backtrack_point(node_backtrack_point);
    component_endpoints.truncate(previous_component_endpoint_len);
    SearchReturn {
        best: best.map(|best| SearchOutcome { sibling_orbits: local_orbits, ..best }),
        first_path_automorphism: if node_is_on_first_path {
            None
        } else {
            node_first_path_automorphism
        },
        best_path_backjump_depth: None,
    }
}

fn first_path_orbits_at_depth_mut<VertexLabel, EdgeLabel>(
    state: &mut SearchState<VertexLabel, EdgeLabel>,
    depth: usize,
    order: usize,
) -> &mut KnownOrbits {
    while state.first_path_orbits_by_depth.len() <= depth {
        state.first_path_orbits_by_depth.push(KnownOrbits::new(order));
    }
    &mut state.first_path_orbits_by_depth[depth]
}

fn first_path_automorphism_for_current_partition(
    partition: &BacktrackableOrderedPartition,
    first_order: &[usize],
) -> Vec<usize> {
    let order = first_order.len();
    let current_elements = partition.ordered_elements();
    debug_assert_eq!(order, current_elements.len());

    let mut first_labeling = vec![0usize; order];
    for (index, &vertex) in first_order.iter().enumerate() {
        first_labeling[vertex] = index;
    }

    let mut automorphism = (0..order).collect::<Vec<_>>();
    for (index, &current_vertex) in current_elements.iter().enumerate() {
        if partition.cell_len(partition.cell_of(current_vertex)) == 1 {
            automorphism[first_order[index]] = current_vertex;
        }
    }

    for cell in partition.non_singleton_cells() {
        for &vertex in cell.elements() {
            let image_vertex = current_elements[first_labeling[vertex]];
            if partition.cell_len(partition.cell_of(image_vertex)) == 1 {
                automorphism[image_vertex] = vertex;
            } else {
                automorphism[vertex] = vertex;
            }
        }
    }

    automorphism
}

fn leaf_automorphism(first_order: &[usize], current_order: &[usize]) -> Vec<usize> {
    debug_assert_eq!(first_order.len(), current_order.len());
    let mut automorphism = vec![0usize; first_order.len()];
    for (&first_vertex, &current_vertex) in first_order.iter().zip(current_order.iter()) {
        automorphism[first_vertex] = current_vertex;
    }
    automorphism
}

fn active_component_endpoint(
    component_endpoints: &[ComponentEndpoint],
    active_component_endpoint_len: usize,
) -> Option<&ComponentEndpoint> {
    active_component_endpoint_len.checked_sub(1).and_then(|index| component_endpoints.get(index))
}

fn active_component_endpoint_mut(
    component_endpoints: &mut [ComponentEndpoint],
    active_component_endpoint_len: usize,
) -> Option<&mut ComponentEndpoint> {
    active_component_endpoint_len
        .checked_sub(1)
        .and_then(|index| component_endpoints.get_mut(index))
}

fn prepare_component_recursion_and_choose_target_cell<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &mut BacktrackableOrderedPartition,
    splitting_heuristic: CanonSplittingHeuristic,
    current_node_is_on_best_path: bool,
    choice_path: &[usize],
    component_endpoints: &mut Vec<ComponentEndpoint>,
    active_component_endpoint_len: &mut usize,
) -> PartitionCellId
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    loop {
        let active_level = partition
            .highest_non_singleton_component_level()
            .expect("a non-discrete partition must contain at least one non-singleton cell");
        let total_elements_at_level = partition
            .non_singleton_cells()
            .filter(|cell| partition.cell_component_level(cell.id()) == active_level)
            .map(|cell| cell.len())
            .sum::<usize>();
        let Some((component_cells, component_elements, preferred_cell)) =
            find_first_component_at_level(
                graph,
                nodes,
                partition,
                active_level,
                splitting_heuristic,
            )
        else {
            return choose_target_cell_at_level(
                graph,
                nodes,
                partition,
                splitting_heuristic,
                active_level,
            );
        };
        if component_elements < total_elements_at_level {
            let _ = partition.promote_cells_to_new_component_level(&component_cells);
            component_endpoints.push(ComponentEndpoint {
                discrete_cell_limit: partition.number_of_discrete_cells() + component_elements,
                first_checked: false,
                best_checked: false,
                creation_choice_path: choice_path.to_vec(),
                created_on_best_path: current_node_is_on_best_path,
            });
            *active_component_endpoint_len = component_endpoints.len();
            continue;
        }
        return preferred_cell;
    }
}

fn find_first_component_at_level<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    component_level: usize,
    splitting_heuristic: CanonSplittingHeuristic,
) -> Option<(Vec<PartitionCellId>, usize, PartitionCellId)>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    let seed_cell = partition
        .non_singleton_cells()
        .find(|cell| partition.cell_component_level(cell.id()) == component_level)
        .map(|cell| cell.id())?;
    let mut component_cells = vec![seed_cell];
    let mut seen = BTreeSet::from([seed_cell.index()]);
    let mut preferred_cell = seed_cell;
    let mut preferred_first = partition.cell_first(seed_cell);
    let mut preferred_size = partition.cell_len(seed_cell);
    let mut preferred_nuconn = 0usize;
    let mut cursor = 0usize;

    while cursor < component_cells.len() {
        let cell_id = component_cells[cursor];
        cursor += 1;
        let representative = nodes[partition.cell_elements(cell_id)[0]];
        let mut neighbour_counts = BTreeMap::<PartitionCellId, usize>::new();

        for neighbour in graph.successors(representative) {
            let neighbour_cell = partition.cell_of(neighbour.as_());
            if partition.cell_len(neighbour_cell) == 1 {
                continue;
            }
            *neighbour_counts.entry(neighbour_cell).or_default() += 1;
        }

        let mut nuconn = 1usize;
        for (neighbour_cell, count) in neighbour_counts {
            if count == partition.cell_len(neighbour_cell) {
                continue;
            }
            nuconn += 1;
            if seen.insert(neighbour_cell.index()) {
                component_cells.push(neighbour_cell);
            }
        }

        let cell_first = partition.cell_first(cell_id);
        let cell_size = partition.cell_len(cell_id);
        let replace_preferred = match splitting_heuristic {
            CanonSplittingHeuristic::First => cell_first < preferred_first,
            CanonSplittingHeuristic::FirstSmallest => {
                cell_size < preferred_size
                    || (cell_size == preferred_size && cell_first < preferred_first)
            }
            CanonSplittingHeuristic::FirstLargest => {
                cell_size > preferred_size
                    || (cell_size == preferred_size && cell_first < preferred_first)
            }
            CanonSplittingHeuristic::FirstMaxNeighbours => {
                nuconn > preferred_nuconn
                    || (nuconn == preferred_nuconn && cell_first < preferred_first)
            }
            CanonSplittingHeuristic::FirstSmallestMaxNeighbours => {
                nuconn > preferred_nuconn
                    || (nuconn == preferred_nuconn
                        && (cell_size < preferred_size
                            || (cell_size == preferred_size && cell_first < preferred_first)))
            }
            CanonSplittingHeuristic::FirstLargestMaxNeighbours => {
                nuconn > preferred_nuconn
                    || (nuconn == preferred_nuconn
                        && (cell_size > preferred_size
                            || (cell_size == preferred_size && cell_first < preferred_first)))
            }
        };

        if replace_preferred {
            preferred_cell = cell_id;
            preferred_first = cell_first;
            preferred_size = cell_size;
            preferred_nuconn = nuconn;
        }
    }

    let component_elements =
        component_cells.iter().map(|&cell| partition.cell_len(cell)).sum::<usize>();
    Some((component_cells, component_elements, preferred_cell))
}

fn choose_target_cell_at_level<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    splitting_heuristic: CanonSplittingHeuristic,
    component_level: usize,
) -> PartitionCellId
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    let candidates = partition
        .non_singleton_cells()
        .filter(|cell| partition.cell_component_level(cell.id()) == component_level)
        .map(|cell| cell.id())
        .collect::<Vec<_>>();
    choose_target_cell_among(graph, nodes, partition, splitting_heuristic, &candidates)
}

fn choose_target_cell_among<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    splitting_heuristic: CanonSplittingHeuristic,
    candidates: &[PartitionCellId],
) -> PartitionCellId
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    match splitting_heuristic {
        CanonSplittingHeuristic::First => candidates
            .iter()
            .copied()
            .min_by_key(|&cell| partition.cell_first(cell))
            .expect("a non-discrete partition must contain at least one non-singleton cell"),
        CanonSplittingHeuristic::FirstSmallest => candidates
            .iter()
            .copied()
            .min_by(|&left, &right| {
                partition
                    .cell_len(left)
                    .cmp(&partition.cell_len(right))
                    .then(partition.cell_first(left).cmp(&partition.cell_first(right)))
            })
            .expect("a non-discrete partition must contain at least one non-singleton cell"),
        CanonSplittingHeuristic::FirstLargest => {
            let mut best = *candidates
                .first()
                .expect("a non-discrete partition must contain at least one non-singleton cell");
            for &cell in &candidates[1..] {
                if partition.cell_len(cell) > partition.cell_len(best)
                    || (partition.cell_len(cell) == partition.cell_len(best)
                        && partition.cell_first(cell) < partition.cell_first(best))
                {
                    best = cell;
                }
            }
            best
        }
        CanonSplittingHeuristic::FirstMaxNeighbours => {
            let mut best = *candidates
                .first()
                .expect("a non-discrete partition must contain at least one non-singleton cell");
            let mut best_value = nontrivial_neighbour_cell_count(graph, nodes, partition, best);
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count(graph, nodes, partition, cell);
                if value > best_value
                    || (value == best_value
                        && partition.cell_first(cell) < partition.cell_first(best))
                {
                    best = cell;
                    best_value = value;
                }
            }
            best
        }
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours => {
            let mut best = *candidates
                .first()
                .expect("a non-discrete partition must contain at least one non-singleton cell");
            let mut best_value = nontrivial_neighbour_cell_count(graph, nodes, partition, best);
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count(graph, nodes, partition, cell);
                if value > best_value
                    || (value == best_value
                        && (partition.cell_len(cell) < partition.cell_len(best)
                            || (partition.cell_len(cell) == partition.cell_len(best)
                                && partition.cell_first(cell) < partition.cell_first(best))))
                {
                    best = cell;
                    best_value = value;
                }
            }
            best
        }
        CanonSplittingHeuristic::FirstLargestMaxNeighbours => {
            let mut best = *candidates
                .first()
                .expect("a non-discrete partition must contain at least one non-singleton cell");
            let mut best_value = nontrivial_neighbour_cell_count(graph, nodes, partition, best);
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count(graph, nodes, partition, cell);
                if value > best_value
                    || (value == best_value
                        && (partition.cell_len(cell) > partition.cell_len(best)
                            || (partition.cell_len(cell) == partition.cell_len(best)
                                && partition.cell_first(cell) < partition.cell_first(best))))
                {
                    best = cell;
                    best_value = value;
                }
            }
            best
        }
    }
}

#[cfg(test)]
fn choose_target_cell<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    splitting_heuristic: CanonSplittingHeuristic,
) -> PartitionCellId
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    let candidates = partition.non_singleton_cells().map(|cell| cell.id()).collect::<Vec<_>>();
    choose_target_cell_among(graph, nodes, partition, splitting_heuristic, &candidates)
}

fn nontrivial_neighbour_cell_count<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    cell_id: PartitionCellId,
) -> usize
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    let representative = nodes[partition.cell_elements(cell_id)[0]];
    let mut counts = BTreeMap::<usize, (usize, usize)>::new();

    for neighbour in graph.successors(representative) {
        let neighbour_cell = partition.cell_of(neighbour.as_());
        let neighbour_cell_len = partition.cell_elements(neighbour_cell).len();
        if neighbour_cell_len <= 1 {
            continue;
        }
        let entry = counts.entry(neighbour_cell.index()).or_insert((0, neighbour_cell_len));
        entry.0 += 1;
    }

    counts
        .into_iter()
        .filter(|(_, (count, neighbour_cell_len))| *count < *neighbour_cell_len)
        .count()
}

fn candidate_split_elements<G, EdgeLabel, EF>(
    _graph: &G,
    _edge_label: &mut EF,
    partition: &mut BacktrackableOrderedPartition,
    target_cell: PartitionCellId,
    _current_path_invariants: &[RefinementTrace<EdgeLabel>],
    _best_path_invariants: Option<&[RefinementTrace<EdgeLabel>]>,
    _stats: &mut CanonicalSearchStats,
) -> Vec<usize>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
    EdgeLabel: Ord + Clone,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let mut elements = partition.cell_elements(target_cell).to_vec();
    elements.sort_unstable();
    elements
}

fn choice_path_is_prefix_of(reference: Option<&[usize]>, choice_path: &[usize]) -> bool {
    reference.is_some_and(|reference| {
        choice_path.len() <= reference.len()
            && choice_path.iter().zip(reference.iter()).all(|(left, right)| left == right)
    })
}

fn common_prefix_len(current: &[usize], reference: Option<&[usize]>) -> usize {
    let Some(reference) = reference else {
        return 0;
    };
    current.iter().zip(reference.iter()).take_while(|(left, right)| left == right).count()
}

fn long_prune_record_from_automorphism(automorphism: &[usize]) -> LongPruneRecord {
    let mut fixed = vec![false; automorphism.len()];
    let mut mcrs = vec![false; automorphism.len()];
    let mut visited = vec![false; automorphism.len()];

    for (vertex, &image) in automorphism.iter().enumerate() {
        fixed[vertex] = image == vertex;
        if !visited[vertex] {
            mcrs[vertex] = true;
            let mut current = image;
            while current != vertex {
                visited[current] = true;
                current = automorphism[current];
            }
        }
        visited[vertex] = false;
    }

    LongPruneRecord { fixed, mcrs }
}

fn compute_long_prune_redundant(
    choice_path: &[usize],
    candidates: &[usize],
    records: &[LongPruneRecord],
) -> BTreeSet<usize> {
    let mut redundant = BTreeSet::new();

    for record in records {
        if choice_path.iter().all(|&choice| record.fixed.get(choice).copied().unwrap_or(false)) {
            for &candidate in candidates {
                if !record.mcrs.get(candidate).copied().unwrap_or(true) {
                    redundant.insert(candidate);
                }
            }
        }
    }

    redundant
}

fn build_certificate<G, VertexLabel, EdgeLabel, EF>(
    graph: &G,
    nodes: &[G::NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    order: &[usize],
) -> LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let ordered_vertex_labels = order.iter().map(|&vertex| vertex_labels[vertex].clone()).collect();
    let mut upper_triangle_edge_labels = Vec::new();

    for left_index in 0..order.len() {
        for right_index in (left_index + 1)..order.len() {
            let left = nodes[order[left_index]];
            let right = nodes[order[right_index]];
            if graph.has_successor(left, right) {
                upper_triangle_edge_labels.push(Some(edge_label(left, right)));
            } else {
                upper_triangle_edge_labels.push(None);
            }
        }
    }

    LabeledSimpleGraphCertificate {
        vertex_labels: ordered_vertex_labels,
        upper_triangle_edge_labels,
    }
}

fn is_labeled_graph_automorphism<G, VertexLabel, EdgeLabel, EF>(
    graph: &G,
    nodes: &[G::NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    automorphism: &[usize],
) -> bool
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    if automorphism.len() != nodes.len() {
        return false;
    }

    for (vertex, &image) in automorphism.iter().enumerate() {
        if vertex_labels[vertex] != vertex_labels[image] {
            return false;
        }
    }

    for left in 0..nodes.len() {
        for right in (left + 1)..nodes.len() {
            let mapped_left = automorphism[left];
            let mapped_right = automorphism[right];
            let left_node = nodes[left];
            let right_node = nodes[right];
            let mapped_left_node = nodes[mapped_left];
            let mapped_right_node = nodes[mapped_right];
            let has_edge = graph.has_successor(left_node, right_node);
            let has_mapped_edge = graph.has_successor(mapped_left_node, mapped_right_node);
            if has_edge != has_mapped_edge {
                return false;
            }
            if has_edge
                && edge_label(left_node, right_node)
                    != edge_label(mapped_left_node, mapped_right_node)
            {
                return false;
            }
        }
    }

    true
}

fn finish_result<VertexLabel, EdgeLabel>(
    mut result: CanonicalLabelingResult<VertexLabel, EdgeLabel>,
    stats: CanonicalSearchStats,
) -> CanonicalLabelingResult<VertexLabel, EdgeLabel> {
    result.stats = stats;
    result
}

fn dense_label_ids<Label>(labels: impl Iterator<Item = Label>) -> BTreeMap<Label, usize>
where
    Label: Ord,
{
    labels
        .collect::<alloc::collections::BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, label)| (label, index + 1))
        .collect()
}

fn refine_partition_according_to_unsigned_invariant_like_bliss<F>(
    partition: &mut BacktrackableOrderedPartition,
    mut invariant_of: F,
) -> bool
where
    F: FnMut(usize) -> usize,
{
    let cells = partition.non_singleton_cells().map(|cell| cell.id()).collect::<Vec<_>>();
    let mut refined = false;

    for cell in cells {
        if partition.cell_elements(cell).len() <= 1 {
            continue;
        }
        let produced = partition
            .split_cell_by_unsigned_invariant_like_bliss(cell, |vertex| invariant_of(vertex));
        refined |= produced.len() > 1;
    }

    refined
}

fn path_invariants_prefix_cmp<EdgeLabel>(
    current: &[RefinementTrace<EdgeLabel>],
    best: &[RefinementTrace<EdgeLabel>],
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    let limit = current.len().min(best.len());
    for index in 0..limit {
        let cmp = compare_refinement_trace(&current[index], &best[index]);
        if cmp != core::cmp::Ordering::Equal {
            return cmp;
        }
    }
    core::cmp::Ordering::Equal
}

fn path_invariants_lex_cmp<EdgeLabel>(
    current: &[RefinementTrace<EdgeLabel>],
    best: &[RefinementTrace<EdgeLabel>],
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    let prefix_cmp = path_invariants_prefix_cmp(current, best);
    if prefix_cmp != core::cmp::Ordering::Equal {
        return prefix_cmp;
    }
    current.len().cmp(&best.len())
}

fn path_invariants_prefix_equal_strict<EdgeLabel>(
    current: &[RefinementTrace<EdgeLabel>],
    reference: &[RefinementTrace<EdgeLabel>],
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    current.len() <= reference.len()
        && current.iter().zip(reference.iter()).all(|(left, right)| {
            compare_refinement_trace(left, right) == core::cmp::Ordering::Equal
        })
}

fn compare_refinement_trace<EdgeLabel>(
    current: &RefinementTrace<EdgeLabel>,
    best: &RefinementTrace<EdgeLabel>,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    let event_cmp = current.events.cmp(&best.events);
    if event_cmp != core::cmp::Ordering::Equal {
        return event_cmp;
    }

    let length_cmp = current.subcertificate_length.cmp(&best.subcertificate_length);
    if length_cmp != core::cmp::Ordering::Equal {
        return length_cmp;
    }

    current.eqref_hash.cmp(&best.eqref_hash)
}

fn compare_candidate_to_best<VertexLabel, EdgeLabel>(
    candidate_path: &[RefinementTrace<EdgeLabel>],
    _candidate_certificate: &LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>,
    best_path: Option<&[RefinementTrace<EdgeLabel>]>,
    _best_certificate: Option<&LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>>,
) -> core::cmp::Ordering
where
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
{
    match (best_path, _best_certificate) {
        (None, None) => core::cmp::Ordering::Greater,
        (Some(best_path), Some(_)) => {
            let path_cmp = path_invariants_lex_cmp(candidate_path, best_path);
            path_cmp
        }
        _ => core::cmp::Ordering::Greater,
    }
}
