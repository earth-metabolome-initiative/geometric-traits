//! `bliss`-aligned individualization-refinement canonizer for simple
//! undirected labeled graphs.
//!
//! The search follows the same broad structure as `bliss`:
//!
//! - seed the partition from vertex labels and simple unsigned invariants
//! - refine to a labeled equitable partition
//! - depth-first individualization on residual non-singleton cells
//! - track first-path / best-path state and discovered automorphisms during
//!   search
//! - use long-prune records and component-recursion-style endpoints on the hot
//!   path
//!
//! This is still a recursive Rust port rather than a line-by-line translation
//! of `AbstractGraph::search()`. The main remaining structural gap is that it
//! does not yet implement the full `bliss` failure-recording and streamed
//! certificate/bad-node machinery.

use alloc::{collections::BTreeMap, rc::Rc, vec::Vec};

use num_traits::AsPrimitive;

use super::{
    BacktrackableOrderedPartition, PartitionCellId, RefinementTrace,
    refine::{
        RefinementTraceStorage, RefinementWorkspace,
        refine_partition_to_labeled_equitable_with_trace_from_splitters_in_workspace,
    },
};
use crate::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::GenericVocabularyBuilder,
    traits::{MonoplexMonopartiteGraph, VocabularyBuilder},
};

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
    minima: Vec<usize>,
}

impl KnownOrbits {
    #[inline]
    fn new(order: usize) -> Self {
        Self { parents: (0..order).collect(), ranks: vec![0; order], minima: (0..order).collect() }
    }

    #[inline]
    fn same_set(&mut self, left: usize, right: usize) -> bool {
        self.find(left) == self.find(right)
    }

    #[inline]
    fn is_minimal_representative(&mut self, element: usize) -> bool {
        let root = self.find(element);
        self.minima[root] == element
    }

    fn ingest_leaf_automorphism(&mut self, left_order: &[usize], right_order: &[usize]) {
        debug_assert_eq!(left_order.len(), right_order.len());
        for (&left, &right) in left_order.iter().zip(right_order.iter()) {
            if left != right {
                self.union(left, right);
            }
        }
    }

    #[inline]
    fn find(&mut self, element: usize) -> usize {
        let mut root = element;
        while self.parents[root] != root {
            root = self.parents[root];
        }

        let mut current = element;
        while self.parents[current] != current {
            let parent = self.parents[current];
            self.parents[current] = root;
            current = parent;
        }

        root
    }

    #[inline]
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
        self.minima[left_root] = self.minima[left_root].min(self.minima[right_root]);
        if self.ranks[left_root] == self.ranks[right_root] {
            self.ranks[left_root] = self.ranks[left_root].saturating_add(1);
        }
    }

    fn ingest_known_orbits(&mut self, other: &mut Self) {
        let mut representatives = vec![usize::MAX; other.parents.len()];
        for element in 0..other.parents.len() {
            let root = other.find(element);
            let representative = &mut representatives[root];
            if *representative == usize::MAX {
                *representative = element;
            } else {
                self.union(*representative, element);
            }
        }
    }

    fn ingest_automorphism(&mut self, automorphism: &[usize]) {
        debug_assert_eq!(self.parents.len(), automorphism.len());
        for (vertex, &image) in automorphism.iter().enumerate() {
            if vertex != image {
                self.union(vertex, image);
            }
        }
    }
}

struct StoredSearchPath<EdgeLabel> {
    order: Rc<[usize]>,
    leaf_signature: Option<Rc<UnlabeledLeafSignature>>,
    path_invariants: Option<Rc<[Rc<RefinementTrace<EdgeLabel>>]>>,
    packed_path: Option<Rc<[u32]>>,
    choice_path: Rc<[usize]>,
    path_info: Option<Rc<[SearchPathInfo]>>,
}

struct SearchWorkspace {
    neighbour_counts: Vec<usize>,
    touched_neighbour_cells: Vec<usize>,
    component_seen: Vec<bool>,
    component_cells: Vec<PartitionCellId>,
}

impl SearchWorkspace {
    fn new(order: usize) -> Self {
        Self {
            neighbour_counts: vec![0; order],
            touched_neighbour_cells: Vec::new(),
            component_seen: vec![false; order],
            component_cells: Vec::new(),
        }
    }
}

struct SearchState<EdgeLabel> {
    stats: CanonicalSearchStats,
    first_path: Option<StoredSearchPath<EdgeLabel>>,
    first_path_revision: usize,
    best_path: Option<StoredSearchPath<EdgeLabel>>,
    best_path_revision: usize,
    first_path_orbits_global: KnownOrbits,
    first_path_orbits_by_depth: Vec<KnownOrbits>,
    best_path_orbits: KnownOrbits,
    long_prune_records: Vec<LongPruneRecord>,
    refine_workspace: RefinementWorkspace<EdgeLabel>,
    search_workspace: SearchWorkspace,
}

struct SearchOutcome<EdgeLabel> {
    order: Rc<[usize]>,
    leaf_signature: Option<Rc<UnlabeledLeafSignature>>,
    path_invariants: Option<Rc<[Rc<RefinementTrace<EdgeLabel>>]>>,
    packed_path: Option<Rc<[u32]>>,
    path_info: Option<Rc<[SearchPathInfo]>>,
    sibling_orbits: KnownOrbits,
    choice_path: Rc<[usize]>,
}

trait SearchPathSnapshot<EdgeLabel> {
    fn order(&self) -> &[usize];
    fn leaf_signature(&self) -> Option<&UnlabeledLeafSignature>;
    fn path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]>;
    fn packed_path(&self) -> Option<&[u32]>;
    fn path_info(&self) -> Option<&[SearchPathInfo]>;
}

impl<EdgeLabel> StoredSearchPath<EdgeLabel> {
    fn path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]> {
        self.path_invariants.as_deref()
    }

    fn packed_path(&self) -> Option<&[u32]> {
        self.packed_path.as_deref()
    }

    fn choice_path(&self) -> &[usize] {
        self.choice_path.as_ref()
    }

    fn path_info(&self) -> Option<&[SearchPathInfo]> {
        self.path_info.as_deref()
    }
}

impl<EdgeLabel> SearchPathSnapshot<EdgeLabel> for StoredSearchPath<EdgeLabel> {
    fn order(&self) -> &[usize] {
        self.order.as_ref()
    }

    fn leaf_signature(&self) -> Option<&UnlabeledLeafSignature> {
        self.leaf_signature.as_ref().map(Rc::as_ref)
    }

    fn path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]> {
        StoredSearchPath::path_invariants(self)
    }

    fn packed_path(&self) -> Option<&[u32]> {
        StoredSearchPath::packed_path(self)
    }

    fn path_info(&self) -> Option<&[SearchPathInfo]> {
        StoredSearchPath::path_info(self)
    }
}

impl<EdgeLabel> SearchPathSnapshot<EdgeLabel> for SearchOutcome<EdgeLabel> {
    fn order(&self) -> &[usize] {
        self.order.as_ref()
    }

    fn leaf_signature(&self) -> Option<&UnlabeledLeafSignature> {
        self.leaf_signature.as_ref().map(Rc::as_ref)
    }

    fn path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]> {
        self.path_invariants.as_deref()
    }

    fn packed_path(&self) -> Option<&[u32]> {
        self.packed_path.as_deref()
    }

    fn path_info(&self) -> Option<&[SearchPathInfo]> {
        self.path_info.as_deref()
    }
}

impl<EdgeLabel> SearchState<EdgeLabel> {
    fn first_path_snapshot(&self) -> Option<&StoredSearchPath<EdgeLabel>> {
        self.first_path.as_ref()
    }

    fn best_path_snapshot(&self) -> Option<&StoredSearchPath<EdgeLabel>> {
        self.best_path.as_ref()
    }

    fn first_order(&self) -> Option<&[usize]> {
        self.first_path_snapshot().map(SearchPathSnapshot::order)
    }

    fn first_leaf_signature(&self) -> Option<&UnlabeledLeafSignature> {
        self.first_path_snapshot().and_then(SearchPathSnapshot::leaf_signature)
    }

    fn first_path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]> {
        self.first_path_snapshot().and_then(SearchPathSnapshot::path_invariants)
    }

    fn first_packed_path(&self) -> Option<&[u32]> {
        self.first_path_snapshot().and_then(SearchPathSnapshot::packed_path)
    }

    fn first_choice_path(&self) -> Option<&[usize]> {
        self.first_path_snapshot().map(StoredSearchPath::choice_path)
    }

    fn best_order(&self) -> Option<&[usize]> {
        self.best_path_snapshot().map(SearchPathSnapshot::order)
    }

    fn best_path_invariants(&self) -> Option<&[Rc<RefinementTrace<EdgeLabel>>]> {
        self.best_path_snapshot().and_then(SearchPathSnapshot::path_invariants)
    }

    fn best_packed_path(&self) -> Option<&[u32]> {
        self.best_path_snapshot().and_then(SearchPathSnapshot::packed_path)
    }

    fn best_choice_path(&self) -> Option<&[usize]> {
        self.best_path_snapshot().map(StoredSearchPath::choice_path)
    }

    fn first_path_info(&self) -> Option<&[SearchPathInfo]> {
        self.first_path_snapshot().and_then(SearchPathSnapshot::path_info)
    }

    fn best_path_info(&self) -> Option<&[SearchPathInfo]> {
        self.best_path_snapshot().and_then(SearchPathSnapshot::path_info)
    }
}

fn search_path_snapshot_matches_candidate_path<S, EdgeLabel>(
    snapshot: &S,
    candidate_packed_path: Option<&[u32]>,
    candidate_path_info: Option<&[SearchPathInfo]>,
    candidate_trace_path: &[Rc<RefinementTrace<EdgeLabel>>],
) -> bool
where
    S: SearchPathSnapshot<EdgeLabel>,
    EdgeLabel: Ord + Clone,
{
    path_equal_strict(
        candidate_packed_path,
        candidate_path_info,
        candidate_trace_path,
        snapshot.packed_path(),
        snapshot.path_info(),
        snapshot.path_invariants(),
    )
}

fn search_path_snapshot_matches_candidate_certificate<NodeId, S, VertexLabel, EdgeLabel, EF>(
    adjacency: &AdjacencyBitMatrix,
    nodes: &[NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    snapshot: &S,
    candidate_order: &[usize],
    candidate_leaf_signature: Option<&UnlabeledLeafSignature>,
) -> bool
where
    NodeId: Copy,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(NodeId, NodeId) -> EdgeLabel,
    S: SearchPathSnapshot<EdgeLabel>,
{
    leaf_orders_equal(
        adjacency,
        nodes,
        vertex_labels,
        edge_label,
        candidate_order,
        candidate_leaf_signature,
        snapshot.order(),
        snapshot.leaf_signature(),
    )
}

fn build_search_path_info<EdgeLabel>(
    choice_path: &[usize],
    path_invariants: &[Rc<RefinementTrace<EdgeLabel>>],
) -> Rc<[SearchPathInfo]>
where
    EdgeLabel: Ord + Clone,
{
    debug_assert_eq!(choice_path.len(), path_invariants.len().saturating_sub(1));
    let mut certificate_index = 0usize;
    Rc::<[SearchPathInfo]>::from(
        choice_path
            .iter()
            .copied()
            .zip(path_invariants.iter().skip(1))
            .map(|(splitting_element, trace)| {
                let info = SearchPathInfo {
                    splitting_element,
                    certificate_index,
                    subcertificate_length: trace.subcertificate_length,
                    eqref_hash: trace.eqref_hash,
                };
                debug_assert_eq!(info.splitting_element, splitting_element);
                debug_assert_eq!(info.eqref_hash, trace.eqref_hash);
                certificate_index = info.certificate_index + info.subcertificate_length;
                info
            })
            .collect::<Vec<_>>(),
    )
}

impl<EdgeLabel> SearchOutcome<EdgeLabel>
where
    EdgeLabel: Ord + Clone,
{
    fn into_result<NodeId, VertexLabel, EF>(
        self,
        adjacency: &AdjacencyBitMatrix,
        nodes: &[NodeId],
        vertex_labels: &[VertexLabel],
        edge_label: &mut EF,
    ) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
    where
        NodeId: Copy,
        VertexLabel: Ord + Clone,
        EF: FnMut(NodeId, NodeId) -> EdgeLabel,
    {
        let certificate =
            build_certificate(adjacency, nodes, vertex_labels, edge_label, self.order.as_ref());
        CanonicalLabelingResult {
            order: self.order.as_ref().to_vec(),
            certificate,
            stats: CanonicalSearchStats::default(),
        }
    }
}

struct SearchReturn<EdgeLabel> {
    best: Option<SearchOutcome<EdgeLabel>>,
    first_path_automorphism: Option<Vec<usize>>,
    best_path_backjump_depth: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchPathInfo {
    splitting_element: usize,
    certificate_index: usize,
    subcertificate_length: usize,
    eqref_hash: u64,
}

/// Per-node first-path / best-path relation state, mirroring the `bliss`
/// `TreeNode` certificate fields closely enough for the current recursive port.
#[derive(Clone, Copy, Debug)]
struct SearchPathState {
    fp_on: bool,
    in_best_path: bool,
    fp_cert_equal: bool,
    fp_cert_equal_revision: usize,
    cmp_to_best_path: Option<core::cmp::Ordering>,
    cmp_to_best_path_revision: usize,
}

impl SearchPathState {
    fn with_live_membership<EdgeLabel>(
        self,
        state: &SearchState<EdgeLabel>,
        choice_path: &[usize],
        first_path_known_on_entry: bool,
        best_path_known_on_entry: bool,
    ) -> Self {
        if let Some(first_path_info) = state.first_path_info() {
            debug_assert_eq!(
                first_path_info.len(),
                state.first_choice_path().map_or(0, <[usize]>::len),
            );
        }
        if let Some(best_path_info) = state.best_path_info() {
            debug_assert_eq!(
                best_path_info.len(),
                state.best_choice_path().map_or(0, <[usize]>::len),
            );
        }
        Self {
            fp_on: self.fp_on
                || (!first_path_known_on_entry
                    && choice_path_is_prefix_of(state.first_choice_path(), choice_path)),
            in_best_path: self.in_best_path
                || (!best_path_known_on_entry
                    && choice_path_is_prefix_of(state.best_choice_path(), choice_path)),
            ..self
        }
    }

    fn refresh_relations<EdgeLabel>(
        &mut self,
        state: &SearchState<EdgeLabel>,
        current_packed: Option<&[u32]>,
        current_path_info: Option<&[SearchPathInfo]>,
        current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    ) where
        EdgeLabel: Ord + Clone,
    {
        if self.fp_cert_equal_revision != state.first_path_revision {
            self.fp_cert_equal = path_prefix_equal_strict(
                current_packed,
                current_path_info,
                current_traces,
                state.first_packed_path(),
                state.first_path_info(),
                state.first_path_invariants(),
            );
            self.fp_cert_equal_revision = state.first_path_revision;
        }
        if self.cmp_to_best_path_revision != state.best_path_revision {
            self.cmp_to_best_path = (state.best_path_invariants().is_some()
                || state.best_packed_path().is_some())
            .then(|| {
                path_prefix_cmp(
                    current_packed,
                    current_path_info,
                    current_traces,
                    state.best_packed_path(),
                    state.best_path_info(),
                    state.best_path_invariants(),
                )
            });
            self.cmp_to_best_path_revision = state.best_path_revision;
        }
    }

    fn best_path_equal(self) -> bool {
        self.cmp_to_best_path == Some(core::cmp::Ordering::Equal)
    }

    fn best_path_not_worse(self) -> bool {
        self.cmp_to_best_path.is_some_and(|cmp| cmp != core::cmp::Ordering::Less)
    }

    fn is_worse_than_best_off_first(self) -> bool {
        self.cmp_to_best_path == Some(core::cmp::Ordering::Less) && !self.fp_cert_equal
    }

    fn for_component_continuation(self) -> Self {
        Self { fp_on: false, in_best_path: false, ..self }
    }
}

#[derive(Clone, Debug)]
struct LongPruneRecord {
    fixed: Vec<bool>,
    mcrs: Vec<bool>,
}

#[derive(Clone, Debug)]
struct ComponentEndpoint {
    discrete_cell_limit: usize,
    next_active_component_endpoint_len: usize,
    first_checked: bool,
    best_checked: bool,
    creation_choice_path: Vec<usize>,
}

impl ComponentEndpoint {
    fn creation_path_is_prefix_of(&self, choice_path: Option<&[usize]>) -> bool {
        choice_path.is_some_and(|choice_path| {
            self.creation_choice_path.len() <= choice_path.len()
                && self
                    .creation_choice_path
                    .iter()
                    .zip(choice_path.iter())
                    .all(|(left, right)| left == right)
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UnlabeledLeafSignature {
    vertex_label_ids: Rc<[usize]>,
    present_edge_offsets: Rc<[usize]>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeSubdivisionVertexLabel<VertexLabel, EdgeLabel> {
    Original(VertexLabel),
    Edge(EdgeLabel),
}

struct AdjacencyBitMatrix {
    words_per_row: usize,
    bits: Vec<u64>,
}

impl AdjacencyBitMatrix {
    fn from_graph<G>(graph: &G, nodes: &[G::NodeId]) -> Self
    where
        G: MonoplexMonopartiteGraph,
        G::NodeId: AsPrimitive<usize> + Copy,
    {
        let order = nodes.len();
        let words_per_row = order.div_ceil(u64::BITS as usize);
        let mut bits = vec![0u64; order * words_per_row];
        for (source, &source_node) in nodes.iter().enumerate() {
            for destination_node in graph.successors(source_node) {
                let destination = destination_node.as_();
                Self::set_bit(&mut bits, words_per_row, source, destination);
                Self::set_bit(&mut bits, words_per_row, destination, source);
            }
        }
        Self { words_per_row, bits }
    }

    fn has_edge(&self, left: usize, right: usize) -> bool {
        self.has_edge_at_row_start(self.row_start(left), right)
    }

    #[inline]
    fn row_start(&self, row: usize) -> usize {
        row * self.words_per_row
    }

    #[inline]
    fn has_edge_at_row_start(&self, row_start: usize, right: usize) -> bool {
        let word = right / (u64::BITS as usize);
        let bit = right % (u64::BITS as usize);
        ((self.bits[row_start + word] >> bit) & 1) != 0
    }

    fn set_bit(bits: &mut [u64], words_per_row: usize, row: usize, column: usize) {
        let word = column / (u64::BITS as usize);
        let bit = column % (u64::BITS as usize);
        let row_start = row * words_per_row;
        bits[row_start + word] |= 1u64 << bit;
    }
}

/// Computes a canonical labeling for a simple undirected graph with total-order
/// vertex and edge labels.
///
/// This path preserves the current `bliss`-alignment strategy by reducing the
/// edge-labeled graph to a vertex-labeled graph before search.
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
///
/// This path preserves the current `bliss`-alignment strategy by reducing the
/// edge-labeled graph to a vertex-labeled graph before search.
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
    let reduced_result = canonical_label_labeled_simple_graph_core(
        &reduced.graph,
        |node| reduced.vertex_labels[node].clone(),
        |_, _| (),
        options,
    );
    let order = reduced_result
        .order
        .into_iter()
        .filter(|&vertex| vertex < reduced.original_vertex_count)
        .collect::<Vec<_>>();
    let original_adjacency = AdjacencyBitMatrix::from_graph(graph, &reduced.original_nodes);
    let certificate = build_certificate(
        &original_adjacency,
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
    let adjacency = AdjacencyBitMatrix::from_graph(graph, &nodes);
    let vertex_label_ids_map = dense_label_ids(vertex_labels.iter().cloned());
    let vertex_label_ids =
        vertex_labels.iter().map(|label| vertex_label_ids_map[label]).collect::<Vec<_>>();
    let mut partition = BacktrackableOrderedPartition::new(order);
    if order > 1 {
        let degrees =
            nodes.iter().copied().map(|node| graph.successors(node).count()).collect::<Vec<_>>();
        let _ =
            refine_partition_according_to_unsigned_invariant_like_bliss(&mut partition, |vertex| {
                vertex_label_ids[vertex]
            });
        let _ =
            refine_partition_according_to_unsigned_invariant_like_bliss(&mut partition, |vertex| {
                degrees[vertex]
            });
    }
    let mut refine_workspace = RefinementWorkspace::new(order);
    let initial_splitters =
        partition.cells().map(super::partition::PartitionCellView::id).collect::<Vec<_>>();
    let (_, root_trace) =
        refine_partition_to_labeled_equitable_with_trace_from_splitters_in_workspace(
            graph,
            &nodes,
            &mut partition,
            &mut edge_label,
            initial_splitters,
            &mut refine_workspace,
        );
    let root_is_discrete = partition.is_discrete();
    let mut state = SearchState::<EdgeLabel> {
        stats: CanonicalSearchStats {
            leaf_nodes: usize::from(!root_is_discrete),
            ..CanonicalSearchStats::default()
        },
        first_path: None,
        first_path_revision: 0,
        best_path: None,
        best_path_revision: 0,
        first_path_orbits_global: KnownOrbits::new(order),
        first_path_orbits_by_depth: Vec::new(),
        best_path_orbits: KnownOrbits::new(order),
        long_prune_records: Vec::new(),
        refine_workspace,
        search_workspace: SearchWorkspace::new(order),
    };
    let mut path_invariants = vec![Rc::new(root_trace)];
    let mut packed_path = match &path_invariants[0].storage {
        RefinementTraceStorage::Packed(words) => Some(words.clone()),
        RefinementTraceStorage::Events(_) => None,
    };
    let mut packed_path_info = Vec::new();
    let mut choice_path = Vec::new();
    let mut component_endpoints = Vec::new();
    let search_return = search_canonical_labeling(
        graph,
        &adjacency,
        &nodes,
        &vertex_labels,
        &vertex_label_ids,
        &mut edge_label,
        &mut partition,
        &mut state,
        &mut path_invariants,
        &mut packed_path,
        &mut packed_path_info,
        &mut choice_path,
        &mut component_endpoints,
        0,
        SearchPathState {
            fp_on: true,
            in_best_path: false,
            fp_cert_equal: false,
            fp_cert_equal_revision: 0,
            cmp_to_best_path: None,
            cmp_to_best_path_revision: 0,
        },
        true,
        0,
        options.splitting_heuristic,
    );
    let outcome = search_return
        .best
        .expect("a canonizer over any finite graph must produce at least one leaf");
    let result = outcome.into_result(&adjacency, &nodes, &vertex_labels, &mut edge_label);
    finish_result(result, state.stats)
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
    for (source, &source_node) in original_nodes.iter().enumerate().take(original_vertex_count) {
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

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments, clippy::too_many_lines)]
fn search_canonical_labeling<G, VertexLabel, EdgeLabel, EF>(
    graph: &G,
    adjacency: &AdjacencyBitMatrix,
    nodes: &[G::NodeId],
    vertex_labels: &[VertexLabel],
    vertex_label_ids: &[usize],
    edge_label: &mut EF,
    partition: &mut BacktrackableOrderedPartition,
    state: &mut SearchState<EdgeLabel>,
    path_invariants: &mut Vec<Rc<RefinementTrace<EdgeLabel>>>,
    packed_path: &mut Option<Vec<u32>>,
    packed_path_info: &mut Vec<SearchPathInfo>,
    choice_path: &mut Vec<usize>,
    component_endpoints: &mut Vec<ComponentEndpoint>,
    active_component_endpoint_len: usize,
    current_path_state: SearchPathState,
    count_current_node: bool,
    depth: usize,
    splitting_heuristic: CanonSplittingHeuristic,
) -> SearchReturn<EdgeLabel>
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
        if current_path_state.is_worse_than_best_off_first() {
            state.stats.pruned_path_signatures += 1;
            return SearchReturn {
                best: None,
                first_path_automorphism: None,
                best_path_backjump_depth: None,
            };
        }
        state.stats.leaf_nodes += 1;
        let order = partition
            .cells()
            .map(|cell| {
                debug_assert!(cell.is_unit());
                cell.elements()[0]
            })
            .collect::<Vec<_>>();
        let order_snapshot = Rc::<[usize]>::from(order);
        let mut leaf_signature: Option<Rc<UnlabeledLeafSignature>> = None;
        let mut ensure_leaf_signature = || {
            if core::mem::size_of::<EdgeLabel>() != 0 {
                return None;
            }
            if leaf_signature.is_none() {
                leaf_signature = Some(Rc::new(build_unlabeled_leaf_signature(
                    graph,
                    nodes,
                    vertex_label_ids,
                    order_snapshot.as_ref(),
                )));
            }
            leaf_signature.clone()
        };
        let packed_path_snapshot = packed_path.as_ref().map(|path| Rc::<[u32]>::from(path.clone()));
        let path_invariants_snapshot = if packed_path_snapshot.is_none() {
            Some(Rc::<[Rc<RefinementTrace<EdgeLabel>>]>::from(path_invariants.clone()))
        } else {
            None
        };
        let choice_path_snapshot = Rc::<[usize]>::from(choice_path.clone());
        let path_info_snapshot = if packed_path_snapshot.is_some() {
            Some(Rc::<[SearchPathInfo]>::from(packed_path_info.clone()))
        } else {
            path_invariants_snapshot
                .as_ref()
                .map(|_| build_search_path_info(choice_path, path_invariants))
        };
        let had_first_order = state.first_path.is_some();
        let comparison_to_best = compare_candidate_to_best(
            packed_path.as_deref(),
            Some(packed_path_info.as_slice()),
            path_invariants,
            state.best_packed_path(),
            state.best_path_info(),
            state.best_path_invariants(),
        );
        let candidate_equals_first = state.first_path_snapshot().is_some_and(|first_path| {
            search_path_snapshot_matches_candidate_path(
                first_path,
                packed_path.as_deref(),
                Some(packed_path_info.as_slice()),
                path_invariants,
            ) && search_path_snapshot_matches_candidate_certificate(
                adjacency,
                nodes,
                vertex_labels,
                edge_label,
                first_path,
                order_snapshot.as_ref(),
                ensure_leaf_signature().as_deref(),
            )
        });
        if state.first_path.is_none() {
            state.first_path = Some(StoredSearchPath {
                order: order_snapshot.clone(),
                leaf_signature: ensure_leaf_signature(),
                path_invariants: path_invariants_snapshot.clone(),
                packed_path: packed_path_snapshot.clone(),
                choice_path: choice_path_snapshot.clone(),
                path_info: path_info_snapshot.clone(),
            });
            state.first_path_revision += 1;
        }
        if comparison_to_best == core::cmp::Ordering::Greater {
            state.best_path = Some(StoredSearchPath {
                order: order_snapshot.clone(),
                leaf_signature: None,
                path_invariants: path_invariants_snapshot.clone(),
                packed_path: packed_path_snapshot.clone(),
                choice_path: choice_path_snapshot.clone(),
                path_info: path_info_snapshot.clone(),
            });
            state.best_path_revision += 1;
            state.best_path_orbits = KnownOrbits::new(nodes.len());
        }
        return SearchReturn {
            best: Some(SearchOutcome {
                order: order_snapshot.clone(),
                leaf_signature,
                path_invariants: path_invariants_snapshot,
                packed_path: packed_path_snapshot,
                path_info: path_info_snapshot,
                sibling_orbits: KnownOrbits::new(nodes.len()),
                choice_path: choice_path_snapshot,
            }),
            first_path_automorphism: if had_first_order
                && candidate_equals_first
                && !current_path_state.fp_on
            {
                state
                    .first_order()
                    .map(|first_order| leaf_automorphism(first_order, order_snapshot.as_ref()))
            } else {
                None
            },
            best_path_backjump_depth: None,
        };
    }

    let node_backtrack_point = partition.set_backtrack_point();
    let previous_component_endpoint_len = component_endpoints.len();
    let mut active_component_endpoint_len = active_component_endpoint_len;
    let first_path_known_on_entry = state.first_path.is_some();
    let best_path_known_on_entry = state.best_path.is_some();
    let mut node_path_state = current_path_state.with_live_membership(
        state,
        choice_path,
        first_path_known_on_entry,
        best_path_known_on_entry,
    );
    node_path_state.refresh_relations(
        state,
        packed_path.as_deref(),
        Some(packed_path_info.as_slice()),
        path_invariants,
    );
    let target_cell = prepare_component_recursion_and_choose_target_cell(
        graph,
        nodes,
        partition,
        &mut state.search_workspace,
        splitting_heuristic,
        choice_path,
        component_endpoints,
        &mut active_component_endpoint_len,
    );
    let best_path_invariants =
        state.best_path.as_ref().and_then(|path| path.path_invariants.clone());
    let candidate_choices = candidate_split_elements(
        graph,
        edge_label,
        partition,
        target_cell,
        path_invariants.as_slice(),
        best_path_invariants.as_deref(),
        &mut state.stats,
    );
    let long_prune_redundant = if !node_path_state.fp_on && depth >= 1 {
        compute_long_prune_redundant(
            choice_path.as_slice(),
            &candidate_choices,
            &state.long_prune_records,
            nodes.len(),
        )
    } else {
        Vec::new()
    };
    let mut local_orbits = KnownOrbits::new(nodes.len());
    let mut explored_choices = Vec::new();
    let mut best: Option<SearchOutcome<EdgeLabel>> = None;
    let mut best_choice: Option<usize> = None;
    let mut node_first_path_automorphism: Option<Vec<usize>> = None;
    let target_cell_len = partition.cell_len(target_cell);
    if node_path_state.is_worse_than_best_off_first() {
        state.stats.pruned_path_signatures += 1;
        partition.goto_backtrack_point(node_backtrack_point);
        component_endpoints.truncate(previous_component_endpoint_len);
        return SearchReturn {
            best: None,
            first_path_automorphism: None,
            best_path_backjump_depth: None,
        };
    }

    for (candidate_index, element) in candidate_choices.iter().copied().enumerate() {
        node_path_state = current_path_state.with_live_membership(
            state,
            choice_path,
            first_path_known_on_entry,
            best_path_known_on_entry,
        );
        node_path_state.refresh_relations(
            state,
            packed_path.as_deref(),
            Some(packed_path_info.as_slice()),
            path_invariants,
        );
        if state.first_path.is_some()
            && node_path_state.fp_on
            && !state.first_path_orbits_global.is_minimal_representative(element)
        {
            state.stats.pruned_sibling_orbits += 1;
            if depth == 0 {
                state.stats.pruned_root_orbits += 1;
            }
            continue;
        }

        if state.best_path.is_some()
            && node_path_state.in_best_path
            && !state.best_path_orbits.is_minimal_representative(element)
        {
            state.stats.pruned_sibling_orbits += 1;
            if depth == 0 {
                state.stats.pruned_root_orbits += 1;
            }
            continue;
        }

        if long_prune_redundant.get(element).copied().unwrap_or(false) {
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
        let (_, child_trace) =
            refine_partition_to_labeled_equitable_with_trace_from_splitters_in_workspace(
                graph,
                nodes,
                partition,
                &mut *edge_label,
                [individualized],
                &mut state.refine_workspace,
            );
        let packed_backtrack_len = packed_path.as_ref().map(Vec::len);
        let packed_path_info_backtrack_len = packed_path_info.len();
        let child_trace_on_stack =
            !matches!(child_trace.storage, RefinementTraceStorage::Packed(_));
        if let (Some(current_packed_path), RefinementTraceStorage::Packed(words)) =
            (packed_path.as_mut(), &child_trace.storage)
        {
            current_packed_path.extend_from_slice(words.as_slice());
            packed_path_info.push(SearchPathInfo {
                splitting_element: element,
                certificate_index: packed_backtrack_len.unwrap_or(0),
                subcertificate_length: child_trace.subcertificate_length,
                eqref_hash: child_trace.eqref_hash,
            });
        }
        choice_path.push(element);
        let had_first_path_before_child = state.first_path.is_some();
        let had_best_before_child = state.best_path.is_some();
        let previous_best_order = state.best_path.as_ref().map(|path| path.order.clone());
        let previous_best_path_invariants =
            state.best_path.as_ref().and_then(|path| path.path_invariants.clone());
        let previous_best_packed_path =
            state.best_path.as_ref().and_then(|path| path.packed_path.clone());
        let previous_best_path_info =
            state.best_path.as_ref().and_then(|path| path.path_info.clone());
        let previous_best_choice_path =
            state.best_path.as_ref().map(|path| path.choice_path.clone());
        let child_path_matches_first_prefix =
            choice_path_is_prefix_of(state.first_choice_path(), choice_path);
        let child_is_on_first_path = node_path_state.fp_on && child_path_matches_first_prefix;
        let child_parent_prefix_equal_to_first = node_path_state.fp_cert_equal;
        let child_parent_prefix_cmp_to_best = node_path_state.cmp_to_best_path;
        let child_path_is_equal_to_first_prefix = child_path_prefix_equal_to_reference(
            child_parent_prefix_equal_to_first,
            &child_trace,
            packed_backtrack_len,
            state.first_packed_path(),
            state.first_path_info(),
            state.first_path_invariants(),
            depth + 1,
        );
        let child_path_cmp_to_best_prefix = child_parent_prefix_cmp_to_best.map(|parent_cmp| {
            child_path_prefix_cmp_to_reference(
                parent_cmp,
                &child_trace,
                packed_backtrack_len,
                state.best_packed_path(),
                state.best_path_info(),
                state.best_path_invariants(),
                depth + 1,
            )
        });
        if child_trace_on_stack {
            path_invariants.push(Rc::new(child_trace));
        }
        let child_is_on_best_path = node_path_state.in_best_path
            && choice_path_is_prefix_of(state.best_choice_path(), choice_path);
        let child_path_state = SearchPathState {
            fp_on: child_is_on_first_path,
            in_best_path: child_is_on_best_path,
            fp_cert_equal: child_path_is_equal_to_first_prefix,
            fp_cert_equal_revision: state.first_path_revision,
            cmp_to_best_path: child_path_cmp_to_best_prefix,
            cmp_to_best_path_revision: state.best_path_revision,
        };
        let local_best_matches_first_path = if let Some(current_best) = best.as_mut() {
            let current_best_path_matches_first = (state.first_path_invariants().is_some()
                || state.first_packed_path().is_some())
                && path_equal_strict(
                    current_best.packed_path.as_deref(),
                    current_best.path_info.as_deref(),
                    current_best.path_invariants.as_deref().unwrap_or(&[]),
                    state.first_packed_path(),
                    state.first_path_info(),
                    state.first_path_invariants(),
                );
            current_best_path_matches_first
                && state.first_order().is_some_and(|first_order| {
                    leaf_orders_equal(
                        adjacency,
                        nodes,
                        vertex_labels,
                        edge_label,
                        current_best.order.as_ref(),
                        current_best.leaf_signature.as_deref(),
                        first_order,
                        state.first_leaf_signature(),
                    )
                })
        } else {
            false
        };
        let mut descended_via_first_component_boundary = false;
        let mut next_active_component_endpoint_len = active_component_endpoint_len;
        let mut break_after_component_endpoint_automorphism = false;
        let mut component_endpoint_first_path_automorphism: Option<Vec<usize>> = None;
        let mut component_endpoint_best_path_backjump_depth: Option<usize> = None;
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
                        endpoint.creation_path_is_prefix_of(state.first_choice_path());
                    let endpoint_created_on_best_path =
                        endpoint.creation_path_is_prefix_of(state.best_choice_path());
                    if state.first_path.is_none() || child_path_state.fp_cert_equal {
                        if !endpoint.first_checked {
                            endpoint.first_checked = true;
                            if depth > 0 {
                                state.stats.search_nodes += 1;
                            }
                            continue_to_next_component = true;
                        } else if endpoint_created_on_first_path {
                            if let Some(first_choice_path) = state.first_choice_path() {
                                let automorphism = state.first_order().map(|first_order| {
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
                                    first_path_orbits_at_depth_mut(
                                        state,
                                        first_difference_depth,
                                        nodes.len(),
                                    )
                                    .union(current_choice, first_choice);
                                    if let Some(automorphism) = automorphism.as_deref() {
                                        debug_assert!(is_labeled_graph_automorphism(
                                            adjacency,
                                            nodes,
                                            vertex_labels,
                                            edge_label,
                                            automorphism,
                                        ));
                                        state.stats.search_nodes += 1;
                                        first_path_orbits_at_depth_mut(
                                            state,
                                            first_difference_depth,
                                            nodes.len(),
                                        )
                                        .ingest_automorphism(automorphism);
                                        state
                                            .first_path_orbits_global
                                            .ingest_automorphism(automorphism);
                                        state.long_prune_records.push(
                                            long_prune_record_from_automorphism(automorphism),
                                        );
                                        local_orbits.ingest_automorphism(automorphism);
                                        component_endpoint_first_path_automorphism =
                                            Some(automorphism.to_vec());
                                        state.stats.automorphisms_found += 1;
                                        break_after_component_endpoint_automorphism = true;
                                        return true;
                                    }
                                    if first_difference_depth == 0 {
                                        state
                                            .first_path_orbits_global
                                            .union(current_choice, first_choice);
                                    }
                                }
                            }
                            if depth > 0 {
                                state.stats.search_nodes += 1;
                            }
                            continue_to_next_component = true;
                        }
                    }
                    if child_path_state.best_path_not_worse() {
                        if !endpoint.best_checked {
                            endpoint.best_checked = true;
                            if depth > 0 && !child_path_state.best_path_equal() {
                                state.stats.search_nodes += 1;
                            }
                            continue_to_next_component = true;
                        } else if endpoint_created_on_best_path
                            && !current_path_state.in_best_path
                            && child_path_state.best_path_equal()
                        {
                            if let Some(best_order) = state.best_order() {
                                let automorphism = first_path_automorphism_for_current_partition(
                                    partition, best_order,
                                );
                                debug_assert!(is_labeled_graph_automorphism(
                                    adjacency,
                                    nodes,
                                    vertex_labels,
                                    edge_label,
                                    &automorphism,
                                ));
                                if depth > 0 {
                                    state.stats.search_nodes += 1;
                                }
                                state.stats.automorphisms_found += 1;
                                state.best_path_orbits.ingest_automorphism(&automorphism);
                                state.first_path_orbits_global.ingest_automorphism(&automorphism);
                                state
                                    .long_prune_records
                                    .push(long_prune_record_from_automorphism(&automorphism));
                                local_orbits.ingest_automorphism(&automorphism);

                                let current_choice_path = &choice_path[..choice_path.len() - 1];
                                let gca_with_first = common_prefix_len(
                                    current_choice_path,
                                    state.first_choice_path(),
                                );
                                let gca_with_best = common_prefix_len(
                                    current_choice_path,
                                    state.best_choice_path(),
                                );
                                component_endpoint_best_path_backjump_depth = if gca_with_first
                                    < current_choice_path.len()
                                    && !state.first_path_orbits_global.is_minimal_representative(
                                        current_choice_path[gca_with_first],
                                    ) {
                                    Some(gca_with_first)
                                } else if gca_with_best < current_choice_path.len()
                                    && !state.best_path_orbits.is_minimal_representative(
                                        current_choice_path[gca_with_best],
                                    )
                                {
                                    Some(gca_with_best)
                                } else {
                                    Some(depth.saturating_sub(1))
                                };
                            }
                            return true;
                        }
                    }
                    if continue_to_next_component {
                        descended_via_first_component_boundary = true;
                        next_active_component_endpoint_len =
                            endpoint.next_active_component_endpoint_len;
                    }
                    false
                });
        if reached_component_endpoint {
            choice_path.pop();
            if child_trace_on_stack {
                path_invariants.pop();
            }
            truncate_packed_path(packed_path, packed_backtrack_len);
            packed_path_info.truncate(packed_path_info_backtrack_len);
            partition.goto_backtrack_point(backtrack_point);
            if break_after_component_endpoint_automorphism {
                if node_path_state.fp_on {
                    continue;
                }
                node_first_path_automorphism = component_endpoint_first_path_automorphism;
                break;
            }
            if let Some(backjump_depth) = component_endpoint_best_path_backjump_depth {
                if backjump_depth < depth {
                    partition.goto_backtrack_point(node_backtrack_point);
                    component_endpoints.truncate(previous_component_endpoint_len);
                    return SearchReturn {
                        best: best
                            .map(|best| SearchOutcome { sibling_orbits: local_orbits, ..best }),
                        first_path_automorphism: if node_path_state.fp_on {
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
        }
        let early_component_first_path_automorphism = !node_path_state.fp_on
            && active_component_endpoint(component_endpoints, active_component_endpoint_len)
                .is_some_and(|endpoint| endpoint.first_checked && local_best_matches_first_path)
            && !partition.is_discrete()
            && child_path_state.fp_cert_equal
            && partition.non_singleton_cells().count() == 1
            && target_cell_len == 2;
        let child_active_component_endpoint_len = if descended_via_first_component_boundary {
            next_active_component_endpoint_len
        } else {
            active_component_endpoint_len
        };
        if early_component_first_path_automorphism {
            if let Some(first_choice_path) =
                state.first_path.as_ref().map(|path| path.choice_path.clone())
            {
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
            if child_trace_on_stack {
                path_invariants.pop();
            }
            truncate_packed_path(packed_path, packed_backtrack_len);
            packed_path_info.truncate(packed_path_info_backtrack_len);
            partition.goto_backtrack_point(backtrack_point);
            break;
        }
        let child_return = search_canonical_labeling(
            graph,
            adjacency,
            nodes,
            vertex_labels,
            vertex_label_ids,
            edge_label,
            partition,
            state,
            path_invariants,
            packed_path,
            packed_path_info,
            choice_path,
            component_endpoints,
            child_active_component_endpoint_len,
            if descended_via_first_component_boundary {
                child_path_state.for_component_continuation()
            } else {
                child_path_state
            },
            depth == 0 || !descended_via_first_component_boundary,
            depth + 1,
            splitting_heuristic,
        );
        let child_first_path_automorphism = child_return.first_path_automorphism.clone();
        let child_best_path_backjump_depth = child_return.best_path_backjump_depth;
        choice_path.pop();
        if child_trace_on_stack {
            path_invariants.pop();
        }
        truncate_packed_path(packed_path, packed_backtrack_len);
        packed_path_info.truncate(packed_path_info_backtrack_len);
        partition.goto_backtrack_point(backtrack_point);
        let Some(candidate) = child_return.best else {
            if let Some(automorphism) = child_first_path_automorphism.as_deref() {
                first_path_orbits_at_depth_mut(state, depth, nodes.len())
                    .ingest_automorphism(automorphism);
                state.first_path_orbits_global.ingest_automorphism(automorphism);
                state.long_prune_records.push(long_prune_record_from_automorphism(automorphism));
                local_orbits.ingest_automorphism(automorphism);
                if !node_path_state.fp_on {
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
                        first_path_automorphism: if node_path_state.fp_on {
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
                    first_path_automorphism: if node_path_state.fp_on {
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

        let candidate_equals_first_path = state.first_path_snapshot().is_some_and(|first_path| {
            search_path_snapshot_matches_candidate_path(
                first_path,
                candidate.packed_path.as_deref(),
                candidate.path_info.as_deref(),
                candidate.path_invariants.as_deref().unwrap_or(&[]),
            ) && search_path_snapshot_matches_candidate_certificate(
                adjacency,
                nodes,
                vertex_labels,
                edge_label,
                first_path,
                candidate.order.as_ref(),
                candidate.leaf_signature.as_deref(),
            )
        });
        let candidate_matches_first_path_automorphism = candidate_equals_first_path
            && had_first_path_before_child
            && !child_is_on_first_path
            && child_path_state.fp_cert_equal;
        if candidate_matches_first_path_automorphism {
            if let Some(first_choice_path) =
                state.first_path.as_ref().map(|path| path.choice_path.clone())
            {
                state.stats.automorphisms_found += 1;
                if let Some(&first_choice) = first_choice_path.get(depth) {
                    let first_order = state.first_path.as_ref().map(|path| path.order.clone());
                    {
                        let first_path_orbits =
                            first_path_orbits_at_depth_mut(state, depth, nodes.len());
                        first_path_orbits.union(element, first_choice);
                        if let Some(first_order) = first_order.as_deref() {
                            first_path_orbits
                                .ingest_leaf_automorphism(first_order, candidate.order.as_ref());
                            local_orbits
                                .ingest_leaf_automorphism(first_order, candidate.order.as_ref());
                        }
                    }
                    state.first_path_orbits_global.union(element, first_choice);
                    if let Some(first_order) = first_order.as_deref() {
                        state
                            .first_path_orbits_global
                            .ingest_leaf_automorphism(first_order, candidate.order.as_ref());
                    }
                }
                if component_endpoints.is_empty() {}
            }
        }
        let candidate_first_path_automorphism = child_first_path_automorphism.or_else(|| {
            if candidate_matches_first_path_automorphism {
                state
                    .first_order()
                    .map(|first_order| leaf_automorphism(first_order, candidate.order.as_ref()))
            } else {
                None
            }
        });
        let prune_remaining_siblings_after_first_path_match = if component_endpoints.is_empty() {
            candidate_equals_first_path
                && child_path_state.fp_cert_equal
                && candidate_choices[(candidate_index + 1)..].iter().copied().all(|remaining| {
                    explored_choices
                        .iter()
                        .any(|&previous| local_orbits.same_set(previous, remaining))
                        || (state.first_path.is_some()
                            && node_path_state.fp_on
                            && !state.first_path_orbits_global.is_minimal_representative(remaining))
                })
        } else {
            candidate_equals_first_path && child_path_state.fp_cert_equal && !node_path_state.fp_on
        };
        let comparison_to_local_best = compare_candidate_to_best(
            candidate.packed_path.as_deref(),
            candidate.path_info.as_deref(),
            candidate.path_invariants.as_deref().unwrap_or(&[]),
            best.as_ref().and_then(|current_best| current_best.packed_path.as_deref()),
            best.as_ref().and_then(|current_best| current_best.path_info.as_deref()),
            best.as_ref().and_then(|current_best| current_best.path_invariants.as_deref()),
        );
        let candidate_matches_previous_best_path = (previous_best_path_invariants.is_some()
            || previous_best_packed_path.is_some())
            && path_equal_strict(
                candidate.packed_path.as_deref(),
                candidate.path_info.as_deref(),
                candidate.path_invariants.as_deref().unwrap_or(&[]),
                previous_best_packed_path.as_deref(),
                previous_best_path_info.as_deref(),
                previous_best_path_invariants.as_deref(),
            );
        let candidate_matches_previous_best_automorphism = if had_best_before_child
            && !child_is_on_best_path
            && !candidate_matches_previous_best_path
        {
            previous_best_order
                .as_ref()
                .map(|best_order| leaf_automorphism(best_order, candidate.order.as_ref()))
                .filter(|automorphism| {
                    is_labeled_graph_automorphism(
                        adjacency,
                        nodes,
                        vertex_labels,
                        edge_label,
                        automorphism,
                    )
                })
        } else {
            None
        };
        let candidate_equals_local_best_certificate = comparison_to_local_best
            == core::cmp::Ordering::Equal
            && best.as_ref().is_some_and(|current_best| {
                search_path_snapshot_matches_candidate_certificate(
                    adjacency,
                    nodes,
                    vertex_labels,
                    edge_label,
                    current_best,
                    candidate.order.as_ref(),
                    candidate.leaf_signature.as_deref(),
                )
            });

        if candidate_equals_local_best_certificate {
            let current_best =
                best.as_ref().expect("equal local-best certificates require an existing best");
            state.stats.automorphisms_found += 1;
            local_orbits
                .union(best_choice.expect("equal sibling branches require a best choice"), element);
            local_orbits
                .ingest_leaf_automorphism(current_best.order.as_ref(), candidate.order.as_ref());
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
                    .unwrap_or_else(|| leaf_automorphism(best_order, candidate.order.as_ref()));
                state.best_path_orbits.ingest_automorphism(&automorphism);
                state.first_path_orbits_global.ingest_automorphism(&automorphism);
                state.long_prune_records.push(long_prune_record_from_automorphism(&automorphism));
                local_orbits.ingest_automorphism(&automorphism);

                let gca_with_first =
                    common_prefix_len(candidate.choice_path.as_ref(), state.first_choice_path());
                let gca_with_best = common_prefix_len(
                    candidate.choice_path.as_ref(),
                    Some(best_choice_path.as_ref()),
                );
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
                            first_path_automorphism: if node_path_state.fp_on {
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

        let candidate_is_new_local_best = matches!(
            (&best, comparison_to_local_best),
            (_, core::cmp::Ordering::Greater) | (None, _)
        );
        if candidate_is_new_local_best {
            best = Some(candidate);
            best_choice = Some(element);
        }

        if let Some(automorphism) = candidate_first_path_automorphism.as_deref() {
            if !candidate_matches_first_path_automorphism {
                first_path_orbits_at_depth_mut(state, depth, nodes.len())
                    .ingest_automorphism(automorphism);
                state.first_path_orbits_global.ingest_automorphism(automorphism);
                local_orbits.ingest_automorphism(automorphism);
            }
            if !node_path_state.fp_on {
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
        first_path_automorphism: if node_path_state.fp_on {
            None
        } else {
            node_first_path_automorphism
        },
        best_path_backjump_depth: None,
    }
}

fn first_path_orbits_at_depth_mut<EdgeLabel>(
    state: &mut SearchState<EdgeLabel>,
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

#[allow(clippy::too_many_arguments)]
fn prepare_component_recursion_and_choose_target_cell<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &mut BacktrackableOrderedPartition,
    search_workspace: &mut SearchWorkspace,
    splitting_heuristic: CanonSplittingHeuristic,
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
        let current_discrete_limit =
            active_component_endpoint(component_endpoints, *active_component_endpoint_len)
                .map_or(partition.order(), |endpoint| endpoint.discrete_cell_limit);
        let Some((component_cells, component_elements, preferred_cell)) =
            find_first_component_at_level_in_workspace(
                graph,
                nodes,
                partition,
                search_workspace,
                active_level,
                splitting_heuristic,
            )
        else {
            return choose_target_cell_at_level_in_workspace(
                graph,
                nodes,
                partition,
                search_workspace,
                splitting_heuristic,
                active_level,
            );
        };
        if partition.number_of_discrete_cells() + component_elements < current_discrete_limit {
            let _ = partition.promote_cells_to_new_component_level(&component_cells);
            component_endpoints.push(ComponentEndpoint {
                discrete_cell_limit: partition.number_of_discrete_cells() + component_elements,
                next_active_component_endpoint_len: *active_component_endpoint_len,
                first_checked: false,
                best_checked: false,
                creation_choice_path: choice_path.to_vec(),
            });
            *active_component_endpoint_len = component_endpoints.len();
            continue;
        }
        return preferred_cell;
    }
}

#[cfg(test)]
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
    let mut workspace = SearchWorkspace::new(partition.order());
    find_first_component_at_level_in_workspace(
        graph,
        nodes,
        partition,
        &mut workspace,
        component_level,
        splitting_heuristic,
    )
}

fn find_first_component_at_level_in_workspace<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    workspace: &mut SearchWorkspace,
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
        .map(super::partition::PartitionCellView::id)?;
    workspace.component_cells.clear();
    workspace.component_cells.push(seed_cell);
    workspace.component_seen.fill(false);
    workspace.component_seen[seed_cell.index()] = true;
    workspace.touched_neighbour_cells.clear();
    let mut preferred_cell = seed_cell;
    let mut preferred_first = partition.cell_first(seed_cell);
    let mut preferred_size = partition.cell_len(seed_cell);
    let mut preferred_nuconn = 0usize;
    let mut cursor = 0usize;

    while cursor < workspace.component_cells.len() {
        let cell_id = workspace.component_cells[cursor];
        cursor += 1;
        let nuconn = 1 + nontrivial_neighbour_cell_count_with_scratch(
            graph,
            nodes,
            partition,
            cell_id,
            &mut workspace.neighbour_counts,
            &mut workspace.touched_neighbour_cells,
        );

        for &neighbour_cell in &workspace.touched_neighbour_cells {
            let neighbour_cell_id = PartitionCellId::from_index(neighbour_cell);
            if workspace.neighbour_counts[neighbour_cell] == partition.cell_len(neighbour_cell_id) {
                continue;
            }
            if !workspace.component_seen[neighbour_cell] {
                workspace.component_seen[neighbour_cell] = true;
                workspace.component_cells.push(neighbour_cell_id);
            }
        }
        clear_neighbour_counts(&mut workspace.neighbour_counts, &workspace.touched_neighbour_cells);
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
        workspace.component_cells.iter().map(|&cell| partition.cell_len(cell)).sum::<usize>();
    Some((workspace.component_cells.clone(), component_elements, preferred_cell))
}

fn choose_target_cell_at_level_in_workspace<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    search_workspace: &mut SearchWorkspace,
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
        .map(super::partition::PartitionCellView::id)
        .collect::<Vec<_>>();
    choose_target_cell_among_in_workspace(
        graph,
        nodes,
        partition,
        search_workspace,
        splitting_heuristic,
        &candidates,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
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
    let mut workspace = SearchWorkspace::new(partition.order());
    choose_target_cell_among_in_workspace(
        graph,
        nodes,
        partition,
        &mut workspace,
        splitting_heuristic,
        candidates,
    )
}

#[allow(clippy::too_many_lines)]
fn choose_target_cell_among_in_workspace<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    workspace: &mut SearchWorkspace,
    splitting_heuristic: CanonSplittingHeuristic,
    candidates: &[PartitionCellId],
) -> PartitionCellId
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    workspace.touched_neighbour_cells.clear();
    match splitting_heuristic {
        CanonSplittingHeuristic::First => {
            candidates
                .iter()
                .copied()
                .min_by_key(|&cell| partition.cell_first(cell))
                .expect("a non-discrete partition must contain at least one non-singleton cell")
        }
        CanonSplittingHeuristic::FirstSmallest => {
            candidates
                .iter()
                .copied()
                .min_by(|&left, &right| {
                    partition
                        .cell_len(left)
                        .cmp(&partition.cell_len(right))
                        .then(partition.cell_first(left).cmp(&partition.cell_first(right)))
                })
                .expect("a non-discrete partition must contain at least one non-singleton cell")
        }
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
            let mut best_value = nontrivial_neighbour_cell_count_with_scratch(
                graph,
                nodes,
                partition,
                best,
                &mut workspace.neighbour_counts,
                &mut workspace.touched_neighbour_cells,
            );
            clear_neighbour_counts(
                &mut workspace.neighbour_counts,
                &workspace.touched_neighbour_cells,
            );
            workspace.touched_neighbour_cells.clear();
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count_with_scratch(
                    graph,
                    nodes,
                    partition,
                    cell,
                    &mut workspace.neighbour_counts,
                    &mut workspace.touched_neighbour_cells,
                );
                clear_neighbour_counts(
                    &mut workspace.neighbour_counts,
                    &workspace.touched_neighbour_cells,
                );
                workspace.touched_neighbour_cells.clear();
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
            let mut best_value = nontrivial_neighbour_cell_count_with_scratch(
                graph,
                nodes,
                partition,
                best,
                &mut workspace.neighbour_counts,
                &mut workspace.touched_neighbour_cells,
            );
            clear_neighbour_counts(
                &mut workspace.neighbour_counts,
                &workspace.touched_neighbour_cells,
            );
            workspace.touched_neighbour_cells.clear();
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count_with_scratch(
                    graph,
                    nodes,
                    partition,
                    cell,
                    &mut workspace.neighbour_counts,
                    &mut workspace.touched_neighbour_cells,
                );
                clear_neighbour_counts(
                    &mut workspace.neighbour_counts,
                    &workspace.touched_neighbour_cells,
                );
                workspace.touched_neighbour_cells.clear();
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
            let mut best_value = nontrivial_neighbour_cell_count_with_scratch(
                graph,
                nodes,
                partition,
                best,
                &mut workspace.neighbour_counts,
                &mut workspace.touched_neighbour_cells,
            );
            clear_neighbour_counts(
                &mut workspace.neighbour_counts,
                &workspace.touched_neighbour_cells,
            );
            workspace.touched_neighbour_cells.clear();
            for &cell in &candidates[1..] {
                let value = nontrivial_neighbour_cell_count_with_scratch(
                    graph,
                    nodes,
                    partition,
                    cell,
                    &mut workspace.neighbour_counts,
                    &mut workspace.touched_neighbour_cells,
                );
                clear_neighbour_counts(
                    &mut workspace.neighbour_counts,
                    &workspace.touched_neighbour_cells,
                );
                workspace.touched_neighbour_cells.clear();
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

#[inline]
fn nontrivial_neighbour_cell_count_with_scratch<G>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &BacktrackableOrderedPartition,
    cell_id: PartitionCellId,
    neighbour_counts: &mut [usize],
    touched_neighbour_cells: &mut Vec<usize>,
) -> usize
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    touched_neighbour_cells.clear();
    let representative = nodes[partition.cell_elements(cell_id)[0]];
    let mut fully_touched_cells = 0usize;

    for neighbour in graph.successors(representative) {
        let neighbour_cell = partition.cell_of(neighbour.as_());
        let neighbour_cell_len = partition.cell_len(neighbour_cell);
        if neighbour_cell_len <= 1 {
            continue;
        }
        let neighbour_cell_index = neighbour_cell.index();
        let previous = neighbour_counts[neighbour_cell_index];
        if previous == 0 {
            touched_neighbour_cells.push(neighbour_cell_index);
        }
        let updated = previous + 1;
        neighbour_counts[neighbour_cell_index] = updated;
        if updated == neighbour_cell_len {
            fully_touched_cells += 1;
        }
    }

    touched_neighbour_cells.len().saturating_sub(fully_touched_cells)
}

#[inline]
fn clear_neighbour_counts(neighbour_counts: &mut [usize], touched_neighbour_cells: &[usize]) {
    for &neighbour_cell_index in touched_neighbour_cells {
        neighbour_counts[neighbour_cell_index] = 0;
    }
}

fn candidate_split_elements<G, EdgeLabel, EF>(
    _graph: &G,
    _edge_label: &mut EF,
    partition: &mut BacktrackableOrderedPartition,
    target_cell: PartitionCellId,
    _current_path_invariants: &[Rc<RefinementTrace<EdgeLabel>>],
    _best_path_invariants: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
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
    order: usize,
) -> Vec<bool> {
    let mut redundant = vec![false; order];

    for record in records {
        if choice_path.iter().all(|&choice| record.fixed.get(choice).copied().unwrap_or(false)) {
            for &candidate in candidates {
                if !record.mcrs.get(candidate).copied().unwrap_or(true) {
                    redundant[candidate] = true;
                }
            }
        }
    }

    redundant
}

fn build_certificate<NodeId, VertexLabel, EdgeLabel, EF>(
    adjacency: &AdjacencyBitMatrix,
    nodes: &[NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    order: &[usize],
) -> LabeledSimpleGraphCertificate<VertexLabel, EdgeLabel>
where
    NodeId: Copy,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(NodeId, NodeId) -> EdgeLabel,
{
    let mut ordered_vertex_labels = Vec::with_capacity(order.len());
    for &vertex in order {
        ordered_vertex_labels.push(vertex_labels[vertex].clone());
    }
    let mut upper_triangle_edge_labels =
        Vec::with_capacity(order.len().saturating_mul(order.len().saturating_sub(1)) / 2);

    for (left_index, &left_vertex) in order.iter().enumerate() {
        let left_row_start = adjacency.row_start(left_vertex);
        let left = nodes[left_vertex];
        for &right_vertex in order.iter().skip(left_index + 1) {
            let right = nodes[right_vertex];
            if adjacency.has_edge_at_row_start(left_row_start, right_vertex) {
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

#[inline]
fn upper_triangle_offset(order_len: usize, left: usize, right: usize) -> usize {
    debug_assert!(left < right);
    left * (2 * order_len - left - 1) / 2 + (right - left - 1)
}

fn build_unlabeled_leaf_signature<G>(
    graph: &G,
    nodes: &[G::NodeId],
    vertex_label_ids: &[usize],
    order: &[usize],
) -> UnlabeledLeafSignature
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    let ordered_vertex_labels =
        order.iter().map(|&vertex| vertex_label_ids[vertex]).collect::<Vec<_>>().into();
    let mut rank_by_vertex = vec![0usize; order.len()];
    for (rank, &vertex) in order.iter().enumerate() {
        rank_by_vertex[vertex] = rank;
    }

    let mut present_edge_offsets = Vec::with_capacity(order.len().saturating_mul(4));
    for source in 0..order.len() {
        let source_rank = rank_by_vertex[source];
        for destination in graph.successors(nodes[source]) {
            let destination = destination.as_();
            if destination >= order.len() {
                continue;
            }
            let destination_rank = rank_by_vertex[destination];
            if destination_rank <= source_rank {
                continue;
            }
            present_edge_offsets.push(upper_triangle_offset(
                order.len(),
                source_rank,
                destination_rank,
            ));
        }
    }
    present_edge_offsets.sort_unstable();

    UnlabeledLeafSignature {
        vertex_label_ids: ordered_vertex_labels,
        present_edge_offsets: present_edge_offsets.into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn leaf_orders_equal<NodeId, VertexLabel, EdgeLabel, EF>(
    adjacency: &AdjacencyBitMatrix,
    nodes: &[NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    left_order: &[usize],
    left_signature: Option<&UnlabeledLeafSignature>,
    right_order: &[usize],
    right_signature: Option<&UnlabeledLeafSignature>,
) -> bool
where
    NodeId: Copy,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(NodeId, NodeId) -> EdgeLabel,
{
    if let (Some(left_signature), Some(right_signature)) = (left_signature, right_signature) {
        return left_signature == right_signature;
    }

    compare_leaf_orders(adjacency, nodes, vertex_labels, edge_label, left_order, right_order)
        == core::cmp::Ordering::Equal
}

fn compare_leaf_orders<NodeId, VertexLabel, EdgeLabel, EF>(
    adjacency: &AdjacencyBitMatrix,
    nodes: &[NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    left_order: &[usize],
    right_order: &[usize],
) -> core::cmp::Ordering
where
    NodeId: Copy,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(NodeId, NodeId) -> EdgeLabel,
{
    for (&left, &right) in left_order.iter().zip(right_order.iter()) {
        let label_cmp = vertex_labels[left].cmp(&vertex_labels[right]);
        if label_cmp != core::cmp::Ordering::Equal {
            return label_cmp;
        }
    }

    if core::mem::size_of::<EdgeLabel>() == 0 {
        for left_index in 0..left_order.len() {
            let left_row_start = adjacency.row_start(left_order[left_index]);
            let right_row_start = adjacency.row_start(right_order[left_index]);
            for right_index in (left_index + 1)..left_order.len() {
                let left_has_edge =
                    adjacency.has_edge_at_row_start(left_row_start, left_order[right_index]);
                let right_has_edge =
                    adjacency.has_edge_at_row_start(right_row_start, right_order[right_index]);
                let edge_cmp = left_has_edge.cmp(&right_has_edge);
                if edge_cmp != core::cmp::Ordering::Equal {
                    return edge_cmp;
                }
            }
        }
        return core::cmp::Ordering::Equal;
    }

    for left_index in 0..left_order.len() {
        let left_vertex = left_order[left_index];
        let left_row_start = adjacency.row_start(left_vertex);
        let mapped_left_vertex = right_order[left_index];
        let right_row_start = adjacency.row_start(mapped_left_vertex);
        for right_index in (left_index + 1)..left_order.len() {
            let right_vertex = left_order[right_index];
            let mapped_right_vertex = right_order[right_index];
            let left_has_edge = adjacency.has_edge_at_row_start(left_row_start, right_vertex);
            let right_has_edge =
                adjacency.has_edge_at_row_start(right_row_start, mapped_right_vertex);
            let edge_presence_cmp = left_has_edge.cmp(&right_has_edge);
            if edge_presence_cmp != core::cmp::Ordering::Equal {
                return edge_presence_cmp;
            }
            if left_has_edge {
                let label_cmp = edge_label(nodes[left_vertex], nodes[right_vertex])
                    .cmp(&edge_label(nodes[mapped_left_vertex], nodes[mapped_right_vertex]));
                if label_cmp != core::cmp::Ordering::Equal {
                    return label_cmp;
                }
            }
        }
    }

    core::cmp::Ordering::Equal
}

fn is_labeled_graph_automorphism<NodeId, VertexLabel, EdgeLabel, EF>(
    adjacency: &AdjacencyBitMatrix,
    nodes: &[NodeId],
    vertex_labels: &[VertexLabel],
    edge_label: &mut EF,
    automorphism: &[usize],
) -> bool
where
    NodeId: Copy,
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
    EF: FnMut(NodeId, NodeId) -> EdgeLabel,
{
    if automorphism.len() != vertex_labels.len() {
        return false;
    }

    for (vertex, &image) in automorphism.iter().enumerate() {
        if vertex_labels[vertex] != vertex_labels[image] {
            return false;
        }
    }

    for left in 0..vertex_labels.len() {
        for right in (left + 1)..vertex_labels.len() {
            let mapped_left = automorphism[left];
            let mapped_right = automorphism[right];
            let left_node = nodes[left];
            let right_node = nodes[right];
            let mapped_left_node = nodes[mapped_left];
            let mapped_right_node = nodes[mapped_right];
            let has_edge = adjacency.has_edge(left, right);
            let has_mapped_edge = adjacency.has_edge(mapped_left, mapped_right);
            if has_edge != has_mapped_edge {
                return false;
            }
            if has_edge
                && core::mem::size_of::<EdgeLabel>() != 0
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
    let cells = partition
        .non_singleton_cells()
        .map(super::partition::PartitionCellView::id)
        .collect::<Vec<_>>();
    let mut refined = false;

    for cell in cells {
        if partition.cell_elements(cell).len() <= 1 {
            continue;
        }
        let produced =
            partition.split_cell_by_unsigned_invariant_like_bliss(cell, &mut invariant_of);
        refined |= produced.len() > 1;
    }

    refined
}

fn truncate_packed_path(packed_path: &mut Option<Vec<u32>>, length: Option<usize>) {
    if let (Some(packed_path), Some(length)) = (packed_path.as_mut(), length) {
        packed_path.truncate(length);
    }
}

fn packed_child_reference_segment<'a>(
    reference: &'a [u32],
    reference_path_info: Option<&'a [SearchPathInfo]>,
    packed_offset: Option<usize>,
    child_depth: usize,
    child_segment_len: usize,
) -> Option<(&'a [u32], Option<&'a SearchPathInfo>)> {
    let reference_info =
        reference_path_info.and_then(|path_info| path_info.get(child_depth.saturating_sub(1)));
    let start = reference_info.map_or_else(
        || packed_offset.expect("packed child trace requires packed path offset"),
        |info| info.certificate_index,
    );
    let end = start + child_segment_len;
    reference.get(start..end).map(|segment| (segment, reference_info))
}

fn trace_path_prefix_cmp<EdgeLabel>(
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    reference_traces: &[Rc<RefinementTrace<EdgeLabel>>],
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    let limit = current_traces.len().min(reference_traces.len());
    for index in 0..limit {
        let cmp = compare_refinement_trace(
            current_traces[index].as_ref(),
            reference_traces[index].as_ref(),
        );
        if cmp != core::cmp::Ordering::Equal {
            return cmp;
        }
    }
    core::cmp::Ordering::Equal
}

fn trace_path_prefix_equal<EdgeLabel>(
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    reference_traces: &[Rc<RefinementTrace<EdgeLabel>>],
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    current_traces.len() <= reference_traces.len()
        && current_traces
            .iter()
            .zip(reference_traces.iter())
            .all(|(left, right)| refinement_trace_equal(left.as_ref(), right.as_ref()))
}

fn trace_path_equal<EdgeLabel>(
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    reference_traces: &[Rc<RefinementTrace<EdgeLabel>>],
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    current_traces.len() == reference_traces.len()
        && current_traces
            .iter()
            .zip(reference_traces.iter())
            .all(|(left, right)| refinement_trace_equal(left.as_ref(), right.as_ref()))
}

fn packed_path_prefix_cmp(current: &[u32], reference: &[u32]) -> core::cmp::Ordering {
    let limit = current.len().min(reference.len());
    current[..limit].cmp(&reference[..limit])
}

fn packed_path_compare(
    current: &[u32],
    current_info: Option<&[SearchPathInfo]>,
    reference: &[u32],
    reference_info: Option<&[SearchPathInfo]>,
    prefix_only: bool,
) -> core::cmp::Ordering {
    if let (Some(current_info), Some(reference_info)) = (current_info, reference_info) {
        if !current_info.is_empty() && !reference_info.is_empty() {
            let current_root_end = current_info[0].certificate_index;
            let reference_root_end = reference_info[0].certificate_index;
            if current_root_end == reference_root_end {
                debug_assert_eq!(current[..current_root_end], reference[..reference_root_end]);
                let current_suffix = &current[current_root_end..];
                let reference_suffix = &reference[reference_root_end..];
                return if prefix_only {
                    let limit = current_suffix.len().min(reference_suffix.len());
                    current_suffix[..limit].cmp(&reference_suffix[..limit])
                } else {
                    current_suffix.cmp(reference_suffix)
                };
            }
        }
    }

    if prefix_only { packed_path_prefix_cmp(current, reference) } else { current.cmp(reference) }
}

fn packed_path_equal(
    current: &[u32],
    current_info: Option<&[SearchPathInfo]>,
    reference: &[u32],
    reference_info: Option<&[SearchPathInfo]>,
    prefix_only: bool,
) -> bool {
    if let (Some(current_info), Some(reference_info)) = (current_info, reference_info) {
        if !current_info.is_empty() && !reference_info.is_empty() {
            let current_root_end = current_info[0].certificate_index;
            let reference_root_end = reference_info[0].certificate_index;
            if current_root_end != reference_root_end {
                return false;
            }
            let current_suffix = &current[current_root_end..];
            let reference_suffix = &reference[reference_root_end..];
            return if prefix_only {
                current_suffix.len() <= reference_suffix.len()
                    && current_suffix == &reference_suffix[..current_suffix.len()]
            } else {
                current_suffix == reference_suffix
            };
        }
    }

    if prefix_only {
        current.len() <= reference.len() && current == &reference[..current.len()]
    } else {
        current == reference
    }
}

#[cfg(test)]
fn packed_path_suffix_cmp(
    current: &[u32],
    current_info: &[SearchPathInfo],
    reference: &[u32],
    reference_info: &[SearchPathInfo],
    prefix_only: bool,
) -> core::cmp::Ordering {
    packed_path_compare(current, Some(current_info), reference, Some(reference_info), prefix_only)
}

#[cfg(test)]
fn packed_path_lex_cmp(current: &[u32], reference: &[u32]) -> core::cmp::Ordering {
    packed_path_compare(current, None, reference, None, false)
}

#[cfg(test)]
fn packed_path_prefix_equal_strict(current: &[u32], reference: &[u32]) -> bool {
    packed_path_equal(current, None, reference, None, true)
}

fn child_path_prefix_equal_to_reference<EdgeLabel>(
    parent_prefix_equal: bool,
    child_trace: &RefinementTrace<EdgeLabel>,
    packed_offset: Option<usize>,
    reference_packed: Option<&[u32]>,
    reference_path_info: Option<&[SearchPathInfo]>,
    reference_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
    child_depth: usize,
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    if !parent_prefix_equal {
        return false;
    }

    match (&child_trace.storage, reference_packed, reference_path_info, reference_traces) {
        (RefinementTraceStorage::Packed(words), Some(reference), reference_path_info, None) => {
            packed_child_reference_segment(
                reference,
                reference_path_info,
                packed_offset,
                child_depth,
                words.len(),
            )
            .is_some_and(|(segment, reference_info)| {
                reference_info.is_none_or(|info| {
                    info.subcertificate_length == child_trace.subcertificate_length
                        && info.eqref_hash == child_trace.eqref_hash
                }) && segment == words.as_slice()
            })
        }
        (RefinementTraceStorage::Events(_), None, _, Some(reference)) => {
            reference
                .get(child_depth)
                .is_some_and(|expected| refinement_trace_equal(child_trace, expected.as_ref()))
        }
        (RefinementTraceStorage::Packed(_) | RefinementTraceStorage::Events(_), None, _, None) => {
            false
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn child_path_prefix_cmp_to_reference<EdgeLabel>(
    parent_prefix_cmp: core::cmp::Ordering,
    child_trace: &RefinementTrace<EdgeLabel>,
    packed_offset: Option<usize>,
    reference_packed: Option<&[u32]>,
    reference_path_info: Option<&[SearchPathInfo]>,
    reference_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
    child_depth: usize,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    if parent_prefix_cmp != core::cmp::Ordering::Equal {
        return parent_prefix_cmp;
    }

    match (&child_trace.storage, reference_packed, reference_path_info, reference_traces) {
        (RefinementTraceStorage::Packed(words), Some(reference), reference_path_info, None) => {
            if let Some((segment, reference_info)) = packed_child_reference_segment(
                reference,
                reference_path_info,
                packed_offset,
                child_depth,
                words.len(),
            ) {
                if reference_info.is_some() {
                    words.as_slice().cmp(segment)
                } else {
                    let available = words.len().min(segment.len());
                    words[..available].cmp(&segment[..available])
                }
            } else if reference_path_info
                .and_then(|path_info| path_info.get(child_depth.saturating_sub(1)))
                .is_some()
            {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        }
        (RefinementTraceStorage::Events(_), None, _, Some(reference)) => {
            reference.get(child_depth).map_or(core::cmp::Ordering::Equal, |expected| {
                compare_refinement_trace(child_trace, expected.as_ref())
            })
        }
        (RefinementTraceStorage::Packed(_) | RefinementTraceStorage::Events(_), None, _, None) => {
            core::cmp::Ordering::Equal
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn path_prefix_cmp<EdgeLabel>(
    current_packed: Option<&[u32]>,
    current_path_info: Option<&[SearchPathInfo]>,
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    best_packed: Option<&[u32]>,
    best_path_info: Option<&[SearchPathInfo]>,
    best_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    if best_packed.is_none() && best_traces.is_none() {
        return core::cmp::Ordering::Greater;
    }

    match (current_packed, best_packed) {
        (Some(current), Some(best)) => {
            packed_path_compare(current, current_path_info, best, best_path_info, true)
        }
        (None, None) => {
            let best = best_traces.expect("trace path comparison requires trace snapshots");
            trace_path_prefix_cmp(current_traces, best)
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn path_lex_cmp<EdgeLabel>(
    current_packed: Option<&[u32]>,
    current_path_info: Option<&[SearchPathInfo]>,
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    best_packed: Option<&[u32]>,
    best_path_info: Option<&[SearchPathInfo]>,
    best_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    if best_packed.is_none() && best_traces.is_none() {
        return core::cmp::Ordering::Greater;
    }

    match (current_packed, best_packed) {
        (Some(current), Some(best)) => {
            packed_path_compare(current, current_path_info, best, best_path_info, false)
        }
        (None, None) => {
            let best = best_traces.expect("trace path comparison requires trace snapshots");
            let prefix_cmp = trace_path_prefix_cmp(current_traces, best);
            if prefix_cmp != core::cmp::Ordering::Equal {
                return prefix_cmp;
            }
            current_traces.len().cmp(&best.len())
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn path_prefix_equal_strict<EdgeLabel>(
    current_packed: Option<&[u32]>,
    current_path_info: Option<&[SearchPathInfo]>,
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    reference_packed: Option<&[u32]>,
    reference_path_info: Option<&[SearchPathInfo]>,
    reference_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    if reference_packed.is_none() && reference_traces.is_none() {
        return false;
    }

    match (current_packed, reference_packed) {
        (Some(current), Some(reference)) => {
            packed_path_equal(current, current_path_info, reference, reference_path_info, true)
        }
        (None, None) => {
            let reference = reference_traces.expect("trace path equality requires trace snapshots");
            trace_path_prefix_equal(current_traces, reference)
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn path_equal_strict<EdgeLabel>(
    current_packed: Option<&[u32]>,
    current_path_info: Option<&[SearchPathInfo]>,
    current_traces: &[Rc<RefinementTrace<EdgeLabel>>],
    reference_packed: Option<&[u32]>,
    reference_path_info: Option<&[SearchPathInfo]>,
    reference_traces: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    if reference_packed.is_none() && reference_traces.is_none() {
        return false;
    }

    match (current_packed, reference_packed) {
        (Some(current), Some(reference)) => {
            packed_path_equal(current, current_path_info, reference, reference_path_info, false)
        }
        (None, None) => {
            let reference = reference_traces.expect("trace path equality requires trace snapshots");
            trace_path_equal(current_traces, reference)
        }
        _ => unreachable!("packed and trace path modes must not mix within one search"),
    }
}

fn compare_refinement_trace<EdgeLabel>(
    current: &RefinementTrace<EdgeLabel>,
    best: &RefinementTrace<EdgeLabel>,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    let storage_cmp = match (&current.storage, &best.storage) {
        (
            RefinementTraceStorage::Packed(current_words),
            RefinementTraceStorage::Packed(best_words),
        ) => current_words.cmp(best_words),
        (
            RefinementTraceStorage::Events(current_events),
            RefinementTraceStorage::Events(best_events),
        ) => current_events.cmp(best_events),
        _ => unreachable!("refinement traces of the same search must use the same storage form"),
    };
    if storage_cmp != core::cmp::Ordering::Equal {
        return storage_cmp;
    }

    let length_cmp = current.subcertificate_length.cmp(&best.subcertificate_length);
    if length_cmp != core::cmp::Ordering::Equal {
        return length_cmp;
    }

    current.eqref_hash.cmp(&best.eqref_hash)
}

fn refinement_trace_equal<EdgeLabel>(
    current: &RefinementTrace<EdgeLabel>,
    reference: &RefinementTrace<EdgeLabel>,
) -> bool
where
    EdgeLabel: Ord + Clone,
{
    current.subcertificate_length == reference.subcertificate_length
        && current.eqref_hash == reference.eqref_hash
        && match (&current.storage, &reference.storage) {
            (
                RefinementTraceStorage::Packed(current_words),
                RefinementTraceStorage::Packed(reference_words),
            ) => current_words == reference_words,
            (
                RefinementTraceStorage::Events(current_events),
                RefinementTraceStorage::Events(reference_events),
            ) => current_events == reference_events,
            _ => false,
        }
}

fn compare_candidate_to_best<EdgeLabel>(
    candidate_packed_path: Option<&[u32]>,
    candidate_path_info: Option<&[SearchPathInfo]>,
    candidate_trace_path: &[Rc<RefinementTrace<EdgeLabel>>],
    best_packed_path: Option<&[u32]>,
    best_path_info: Option<&[SearchPathInfo]>,
    best_trace_path: Option<&[Rc<RefinementTrace<EdgeLabel>>]>,
) -> core::cmp::Ordering
where
    EdgeLabel: Ord + Clone,
{
    match (best_packed_path, best_trace_path) {
        (None, None) => core::cmp::Ordering::Greater,
        _ => {
            path_lex_cmp(
                candidate_packed_path,
                candidate_path_info,
                candidate_trace_path,
                best_packed_path,
                best_path_info,
                best_trace_path,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, vec};

    use super::{super::refine::RefinementTraceEvent, *};
    use crate::{
        impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
        naive_structs::{GenericGraph, GenericVocabularyBuilder},
        traits::{
            Edges, MonopartiteGraph, MonoplexGraph, PartitionCellView, SparseValuedMatrix2D,
            VocabularyBuilder,
        },
    };

    type TestEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
    type TestGraph = GenericGraph<SortedVec<usize>, TestEdges>;

    fn packed_trace(words: &[u32], hash: u64) -> Rc<RefinementTrace<()>> {
        Rc::new(RefinementTrace {
            storage: RefinementTraceStorage::Packed(words.to_vec()),
            subcertificate_length: words.len(),
            eqref_hash: hash,
        })
    }

    fn event_trace(events: Vec<RefinementTraceEvent<u8>>, hash: u64) -> Rc<RefinementTrace<u8>> {
        Rc::new(RefinementTrace {
            subcertificate_length: events.len(),
            storage: RefinementTraceStorage::Events(events),
            eqref_hash: hash,
        })
    }

    fn build_test_graph(number_of_nodes: usize, edges: &[(usize, usize, u8)]) -> TestGraph {
        let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(number_of_nodes)
            .symbols((0..number_of_nodes).enumerate())
            .build()
            .unwrap();
        let mut upper_edges = edges
            .iter()
            .map(|&(source, destination, label)| {
                if source <= destination {
                    (source, destination, label)
                } else {
                    (destination, source, label)
                }
            })
            .collect::<Vec<_>>();
        upper_edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        upper_edges.dedup();
        let edges =
            SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges)
                .unwrap();

        GenericGraph::from((nodes, edges))
    }

    fn build_partition_from_groups(groups: &[usize]) -> BacktrackableOrderedPartition {
        let mut partition = BacktrackableOrderedPartition::new(groups.len());
        let _ = refine_partition_according_to_unsigned_invariant_like_bliss(
            &mut partition,
            |element| groups[element],
        );
        partition
    }

    #[test]
    fn test_build_search_path_info_and_search_state_accessors() {
        let root = packed_trace(&[], 1);
        let first = packed_trace(&[10, 11], 2);
        let second = packed_trace(&[20, 21, 22], 3);
        let traces = vec![root.clone(), first.clone(), second.clone()];
        let choice_path = vec![4, 9];
        let path_info = build_search_path_info(&choice_path, &traces);

        assert_eq!(path_info.len(), 2);
        assert_eq!(path_info[0].splitting_element, 4);
        assert_eq!(path_info[0].certificate_index, 0);
        assert_eq!(path_info[0].subcertificate_length, 2);
        assert_eq!(path_info[1].splitting_element, 9);
        assert_eq!(path_info[1].certificate_index, 2);
        assert_eq!(path_info[1].subcertificate_length, 3);

        let first_path = StoredSearchPath {
            order: Rc::from(vec![2_usize, 0, 1]),
            leaf_signature: Some(Rc::new(UnlabeledLeafSignature {
                vertex_label_ids: Rc::from(vec![1_usize, 2, 3]),
                present_edge_offsets: Rc::from(vec![0_usize, 2]),
            })),
            path_invariants: Some(Rc::from(traces.clone())),
            packed_path: Some(Rc::from(vec![10_u32, 11, 20, 21, 22])),
            choice_path: Rc::from(choice_path.clone()),
            path_info: Some(path_info.clone()),
        };
        let best_path = StoredSearchPath {
            order: Rc::from(vec![1_usize, 0, 2]),
            leaf_signature: None,
            path_invariants: None,
            packed_path: Some(Rc::from(vec![30_u32, 31])),
            choice_path: Rc::from(vec![7_usize]),
            path_info: Some(Rc::from(vec![SearchPathInfo {
                splitting_element: 7,
                certificate_index: 0,
                subcertificate_length: 2,
                eqref_hash: 5,
            }])),
        };
        let state = SearchState {
            stats: CanonicalSearchStats::default(),
            first_path: Some(first_path),
            first_path_revision: 1,
            best_path: Some(best_path),
            best_path_revision: 2,
            first_path_orbits_global: KnownOrbits::new(3),
            first_path_orbits_by_depth: vec![],
            best_path_orbits: KnownOrbits::new(3),
            long_prune_records: vec![],
            refine_workspace: RefinementWorkspace::new(3),
            search_workspace: SearchWorkspace::new(3),
        };

        assert_eq!(state.first_order(), Some(&[2, 0, 1][..]));
        assert_eq!(state.first_choice_path(), Some(&[4, 9][..]));
        assert_eq!(state.first_packed_path(), Some(&[10, 11, 20, 21, 22][..]));
        assert_eq!(state.best_order(), Some(&[1, 0, 2][..]));
        assert_eq!(state.best_choice_path(), Some(&[7][..]));
        assert_eq!(state.best_packed_path(), Some(&[30, 31][..]));
        assert_eq!(state.first_path_info().map(<[SearchPathInfo]>::len), Some(2));
        assert_eq!(state.best_path_info().map(<[SearchPathInfo]>::len), Some(1));
        assert_eq!(
            state.first_leaf_signature().unwrap().present_edge_offsets.as_ref(),
            &[0_usize, 2]
        );
    }

    #[test]
    fn test_search_path_state_refresh_and_component_continuation() {
        let root = packed_trace(&[], 1);
        let first = packed_trace(&[10, 11], 2);
        let traces = vec![root.clone(), first.clone()];
        let path_info = build_search_path_info(&[4], &traces);
        let state = SearchState {
            stats: CanonicalSearchStats::default(),
            first_path: Some(StoredSearchPath {
                order: Rc::from(vec![0_usize, 1]),
                leaf_signature: None,
                path_invariants: Some(Rc::from(traces.clone())),
                packed_path: Some(Rc::from(vec![10_u32, 11])),
                choice_path: Rc::from(vec![4_usize]),
                path_info: Some(path_info.clone()),
            }),
            first_path_revision: 3,
            best_path: Some(StoredSearchPath {
                order: Rc::from(vec![0_usize, 1]),
                leaf_signature: None,
                path_invariants: Some(Rc::from(traces.clone())),
                packed_path: Some(Rc::from(vec![10_u32, 11])),
                choice_path: Rc::from(vec![4_usize]),
                path_info: Some(path_info.clone()),
            }),
            best_path_revision: 5,
            first_path_orbits_global: KnownOrbits::new(2),
            first_path_orbits_by_depth: vec![],
            best_path_orbits: KnownOrbits::new(2),
            long_prune_records: vec![],
            refine_workspace: RefinementWorkspace::new(2),
            search_workspace: SearchWorkspace::new(2),
        };
        let mut path_state = SearchPathState {
            fp_on: false,
            in_best_path: false,
            fp_cert_equal: false,
            fp_cert_equal_revision: 0,
            cmp_to_best_path: None,
            cmp_to_best_path_revision: 0,
        }
        .with_live_membership(&state, &[4], false, false);

        assert!(path_state.fp_on);
        assert!(path_state.in_best_path);

        path_state.refresh_relations(&state, Some(&[10, 11]), Some(&path_info), &traces);
        assert!(path_state.fp_cert_equal);
        assert!(path_state.best_path_equal());
        assert!(path_state.best_path_not_worse());
        assert!(!path_state.is_worse_than_best_off_first());

        let continued = path_state.for_component_continuation();
        assert!(!continued.fp_on);
        assert!(!continued.in_best_path);
        assert!(continued.fp_cert_equal);
    }

    #[test]
    fn test_child_prefix_helpers_cover_packed_mode() {
        let packed_reference = [10_u32, 11, 20, 21, 22];
        let packed_info = [
            SearchPathInfo {
                splitting_element: 4,
                certificate_index: 0,
                subcertificate_length: 2,
                eqref_hash: 2,
            },
            SearchPathInfo {
                splitting_element: 9,
                certificate_index: 2,
                subcertificate_length: 3,
                eqref_hash: 3,
            },
        ];
        let packed_child = RefinementTrace {
            storage: RefinementTraceStorage::<()>::Packed(vec![20_u32, 21, 22]),
            subcertificate_length: 3,
            eqref_hash: 3,
        };
        let packed_other = RefinementTrace {
            storage: RefinementTraceStorage::<()>::Packed(vec![20_u32, 21, 23]),
            subcertificate_length: 3,
            eqref_hash: 4,
        };

        assert!(child_path_prefix_equal_to_reference(
            true,
            &packed_child,
            Some(2),
            Some(&packed_reference),
            Some(&packed_info),
            None,
            2,
        ));
        assert!(!child_path_prefix_equal_to_reference(
            false,
            &packed_child,
            Some(2),
            Some(&packed_reference),
            Some(&packed_info),
            None,
            2,
        ));
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                &packed_other,
                Some(2),
                Some(&packed_reference),
                Some(&packed_info),
                None,
                2,
            ),
            core::cmp::Ordering::Greater,
        );
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Less,
                &packed_child,
                Some(2),
                Some(&packed_reference),
                Some(&packed_info),
                None,
                2,
            ),
            core::cmp::Ordering::Less,
        );
    }

    #[test]
    fn test_child_prefix_helpers_cover_event_mode() {
        let event_child = event_trace(vec![RefinementTraceEvent::Split { first: 3, length: 2 }], 7);
        let event_reference = vec![
            event_trace(vec![RefinementTraceEvent::Split { first: 0, length: 1 }], 1),
            event_child.clone(),
        ];
        assert!(child_path_prefix_equal_to_reference(
            true,
            event_child.as_ref(),
            None,
            None,
            None,
            Some(&event_reference),
            1,
        ));
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                event_child.as_ref(),
                None,
                None,
                None,
                Some(&event_reference),
                1,
            ),
            core::cmp::Ordering::Equal,
        );
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                event_child.as_ref(),
                None,
                None,
                None,
                None,
                1,
            ),
            core::cmp::Ordering::Equal,
        );
    }

    #[test]
    fn test_path_comparison_helpers_cover_packed_mode() {
        let current = [10_u32, 11, 20, 21];
        let best = [10_u32, 11, 20, 22];
        let current_info = [
            SearchPathInfo {
                splitting_element: 4,
                certificate_index: 0,
                subcertificate_length: 2,
                eqref_hash: 2,
            },
            SearchPathInfo {
                splitting_element: 9,
                certificate_index: 2,
                subcertificate_length: 2,
                eqref_hash: 3,
            },
        ];
        let best_info = current_info;
        let empty_traces: Vec<Rc<RefinementTrace<()>>> = vec![];

        assert_eq!(packed_path_prefix_cmp(&current, &best), core::cmp::Ordering::Less);
        assert_eq!(
            packed_path_suffix_cmp(&current, &current_info, &best, &best_info, true),
            core::cmp::Ordering::Less,
        );
        assert_eq!(
            packed_path_suffix_cmp(&current, &current_info, &best, &best_info, false),
            core::cmp::Ordering::Less,
        );
        assert_eq!(packed_path_lex_cmp(&current, &best), core::cmp::Ordering::Less);
        assert!(packed_path_prefix_equal_strict(&current[..2], &best));
        assert_eq!(
            path_prefix_cmp(
                Some(&current),
                Some(&current_info),
                &empty_traces,
                Some(&best),
                Some(&best_info),
                None,
            ),
            core::cmp::Ordering::Less,
        );
        assert_eq!(
            path_lex_cmp(
                Some(&current),
                Some(&current_info),
                &empty_traces,
                Some(&best),
                Some(&best_info),
                None,
            ),
            core::cmp::Ordering::Less,
        );
        assert!(path_prefix_equal_strict(
            Some(&current[..2]),
            None,
            &empty_traces,
            Some(&best),
            None,
            None,
        ));
        assert!(!path_equal_strict(
            Some(&current),
            Some(&current_info),
            &empty_traces,
            Some(&best),
            Some(&best_info),
            None,
        ));
        assert_eq!(
            compare_candidate_to_best(
                Some(&current),
                Some(&current_info),
                &empty_traces,
                None,
                None,
                None,
            ),
            core::cmp::Ordering::Greater,
        );
    }

    #[test]
    fn test_path_comparison_helpers_cover_event_mode() {
        let event_a = event_trace(vec![RefinementTraceEvent::Split { first: 0, length: 2 }], 1);
        let event_b = event_trace(vec![RefinementTraceEvent::Split { first: 1, length: 2 }], 2);
        let current_traces = vec![event_a.clone()];
        let best_traces = vec![event_b.clone()];
        assert_eq!(
            path_prefix_cmp::<u8>(None, None, &current_traces, None, None, Some(&best_traces)),
            core::cmp::Ordering::Less,
        );
        assert_eq!(
            path_lex_cmp::<u8>(None, None, &current_traces, None, None, Some(&best_traces)),
            core::cmp::Ordering::Less,
        );
        assert!(path_prefix_equal_strict::<u8>(
            None,
            None,
            &current_traces,
            None,
            None,
            Some(&current_traces),
        ));
        assert!(path_equal_strict::<u8>(
            None,
            None,
            &current_traces,
            None,
            None,
            Some(&current_traces),
        ));
        assert_eq!(
            compare_refinement_trace(event_a.as_ref(), event_b.as_ref()),
            core::cmp::Ordering::Less,
        );
        assert!(refinement_trace_equal(event_a.as_ref(), event_a.as_ref()));
    }

    #[test]
    fn test_choose_target_cell_among_covers_bliss_heuristics() {
        let graph = build_test_graph(8, &[(0, 5, 1), (1, 2, 1), (2, 5, 1)]);
        let nodes = graph.node_ids().collect::<Vec<_>>();
        let partition = build_partition_from_groups(&[0, 0, 1, 1, 1, 2, 2, 3]);
        let candidates =
            partition.non_singleton_cells().map(PartitionCellView::id).take(2).collect::<Vec<_>>();

        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::First,
                &candidates,
            ),
            candidates[0]
        );
        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::FirstSmallest,
                &candidates,
            ),
            candidates[0]
        );
        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::FirstLargest,
                &candidates,
            ),
            candidates[1]
        );
        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::FirstMaxNeighbours,
                &candidates,
            ),
            candidates[1]
        );
        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
                &candidates,
            ),
            candidates[1]
        );
        assert_eq!(
            choose_target_cell_among(
                &graph,
                &nodes,
                &partition,
                CanonSplittingHeuristic::FirstLargestMaxNeighbours,
                &candidates,
            ),
            candidates[1]
        );
    }

    #[test]
    fn test_choose_target_cell_among_covers_tie_break_paths() {
        let tie_graph = build_test_graph(8, &[(0, 5, 1), (2, 5, 1)]);
        let tie_nodes = tie_graph.node_ids().collect::<Vec<_>>();
        let tie_partition = build_partition_from_groups(&[0, 0, 1, 1, 1, 2, 2, 3]);
        let tie_candidates = tie_partition
            .non_singleton_cells()
            .map(PartitionCellView::id)
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(
            choose_target_cell_among(
                &tie_graph,
                &tie_nodes,
                &tie_partition,
                CanonSplittingHeuristic::FirstMaxNeighbours,
                &tie_candidates,
            ),
            tie_candidates[0]
        );
        assert_eq!(
            choose_target_cell_among(
                &tie_graph,
                &tie_nodes,
                &tie_partition,
                CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
                &tie_candidates,
            ),
            tie_candidates[0]
        );
        assert_eq!(
            choose_target_cell_among(
                &tie_graph,
                &tie_nodes,
                &tie_partition,
                CanonSplittingHeuristic::FirstLargestMaxNeighbours,
                &tie_candidates,
            ),
            tie_candidates[1]
        );
    }

    #[test]
    fn test_component_helpers_cover_partial_full_and_promoted_components() {
        let graph = build_test_graph(8, &[(0, 5, 1), (1, 2, 1), (2, 5, 1)]);
        let nodes = graph.node_ids().collect::<Vec<_>>();
        let partition = build_partition_from_groups(&[0, 0, 1, 1, 1, 2, 2, 3]);
        let candidates =
            partition.non_singleton_cells().map(PartitionCellView::id).take(2).collect::<Vec<_>>();
        let (component_cells, component_elements, preferred_cell) = find_first_component_at_level(
            &graph,
            &nodes,
            &partition,
            0,
            CanonSplittingHeuristic::FirstLargestMaxNeighbours,
        )
        .expect("partial component should be found");
        assert_eq!(component_cells.len(), 3);
        assert_eq!(component_elements, 7);
        assert_eq!(preferred_cell, candidates[1]);

        let full_touch_graph = build_test_graph(8, &[(0, 5, 1), (0, 6, 1)]);
        let full_touch_nodes = full_touch_graph.node_ids().collect::<Vec<_>>();
        let full_touch_partition = build_partition_from_groups(&[0, 0, 1, 1, 1, 2, 2, 3]);
        let (single_component_cells, single_component_elements, single_preferred) =
            find_first_component_at_level(
                &full_touch_graph,
                &full_touch_nodes,
                &full_touch_partition,
                0,
                CanonSplittingHeuristic::First,
            )
            .expect("seed cell should still form a component");
        assert_eq!(single_component_cells.len(), 1);
        assert_eq!(single_component_elements, 2);
        assert_eq!(partition.cell_first(single_preferred), 0);

        let promoted_graph = build_test_graph(9, &[(0, 5, 1), (1, 2, 1), (2, 5, 1)]);
        let promoted_nodes = promoted_graph.node_ids().collect::<Vec<_>>();
        let mut promoted_partition = build_partition_from_groups(&[0, 0, 1, 1, 1, 2, 2, 3, 3]);
        let mut component_endpoints = Vec::new();
        let mut active_component_endpoint_len = 0;
        let mut search_workspace = SearchWorkspace::new(promoted_partition.order());
        let selected = prepare_component_recursion_and_choose_target_cell(
            &promoted_graph,
            &promoted_nodes,
            &mut promoted_partition,
            &mut search_workspace,
            CanonSplittingHeuristic::FirstLargestMaxNeighbours,
            &[],
            &mut component_endpoints,
            &mut active_component_endpoint_len,
        );
        assert_eq!(promoted_partition.cell_first(selected), 2);
        assert_eq!(component_endpoints.len(), 1);
        assert_eq!(active_component_endpoint_len, 1);
        assert_eq!(component_endpoints[0].discrete_cell_limit, 7);
    }

    #[test]
    fn test_leaf_order_and_automorphism_helpers_cover_fallback_branches() {
        let unlabeled_graph = build_test_graph(3, &[(0, 1, 1), (1, 2, 1)]);
        let unlabeled_nodes = unlabeled_graph.node_ids().collect::<Vec<_>>();
        let unlabeled_adjacency =
            AdjacencyBitMatrix::from_graph(&unlabeled_graph, &unlabeled_nodes);
        let uniform_vertex_labels = vec![0_u8, 0, 0];
        let mut unit_edge_label = |_: usize, _: usize| ();

        assert_eq!(
            compare_leaf_orders(
                &unlabeled_adjacency,
                &unlabeled_nodes,
                &uniform_vertex_labels,
                &mut unit_edge_label,
                &[0, 1, 2],
                &[1, 0, 2],
            ),
            core::cmp::Ordering::Less,
        );
        assert!(leaf_orders_equal(
            &unlabeled_adjacency,
            &unlabeled_nodes,
            &uniform_vertex_labels,
            &mut unit_edge_label,
            &[0, 1, 2],
            None,
            &[2, 1, 0],
            None,
        ));

        let labeled_graph = build_test_graph(3, &[(0, 1, 1), (1, 2, 2)]);
        let labeled_nodes = labeled_graph.node_ids().collect::<Vec<_>>();
        let labeled_adjacency = AdjacencyBitMatrix::from_graph(&labeled_graph, &labeled_nodes);
        let labeled_matrix = Edges::matrix(labeled_graph.edges());
        let mut labeled_edge_label =
            |left: usize, right: usize| labeled_matrix.sparse_value_at(left, right).unwrap();
        assert_eq!(
            compare_leaf_orders(
                &labeled_adjacency,
                &labeled_nodes,
                &uniform_vertex_labels,
                &mut labeled_edge_label,
                &[0, 1, 2],
                &[2, 1, 0],
            ),
            core::cmp::Ordering::Less,
        );
        assert!(!is_labeled_graph_automorphism(
            &labeled_adjacency,
            &labeled_nodes,
            &uniform_vertex_labels,
            &mut labeled_edge_label,
            &[0, 1],
        ));
        assert!(!is_labeled_graph_automorphism(
            &labeled_adjacency,
            &labeled_nodes,
            &[0_u8, 1, 0],
            &mut labeled_edge_label,
            &[1, 0, 2],
        ));
        assert!(!is_labeled_graph_automorphism(
            &labeled_adjacency,
            &labeled_nodes,
            &uniform_vertex_labels,
            &mut labeled_edge_label,
            &[0, 2, 1],
        ));
        assert!(!is_labeled_graph_automorphism(
            &labeled_adjacency,
            &labeled_nodes,
            &uniform_vertex_labels,
            &mut labeled_edge_label,
            &[2, 1, 0],
        ));

        let equal_label_graph = build_test_graph(3, &[(0, 1, 1), (1, 2, 1)]);
        let equal_label_nodes = equal_label_graph.node_ids().collect::<Vec<_>>();
        let equal_label_adjacency =
            AdjacencyBitMatrix::from_graph(&equal_label_graph, &equal_label_nodes);
        let equal_label_matrix = Edges::matrix(equal_label_graph.edges());
        let mut equal_edge_label =
            |left: usize, right: usize| equal_label_matrix.sparse_value_at(left, right).unwrap();
        assert!(is_labeled_graph_automorphism(
            &equal_label_adjacency,
            &equal_label_nodes,
            &uniform_vertex_labels,
            &mut equal_edge_label,
            &[2, 1, 0],
        ));
    }

    #[test]
    fn test_child_prefix_helpers_cover_missing_reference_cases() {
        let packed_child = RefinementTrace {
            storage: RefinementTraceStorage::<()>::Packed(vec![20_u32, 21]),
            subcertificate_length: 2,
            eqref_hash: 2,
        };
        assert!(child_path_prefix_equal_to_reference(
            true,
            &packed_child,
            Some(2),
            Some(&[10_u32, 11, 20, 21]),
            None,
            None,
            1,
        ));
        assert!(!child_path_prefix_equal_to_reference(
            true,
            &packed_child,
            Some(3),
            Some(&[10_u32, 11, 20]),
            None,
            None,
            1,
        ));
        assert!(!child_path_prefix_equal_to_reference(
            true,
            &packed_child,
            None,
            None,
            None,
            None,
            1,
        ));

        let short_reference_info = [SearchPathInfo {
            splitting_element: 4,
            certificate_index: 3,
            subcertificate_length: 2,
            eqref_hash: 2,
        }];
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                &packed_child,
                None,
                Some(&[10_u32, 11, 12, 13]),
                Some(&short_reference_info),
                None,
                1,
            ),
            core::cmp::Ordering::Greater,
        );
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                &packed_child,
                Some(8),
                Some(&[10_u32, 11, 12, 13]),
                None,
                None,
                1,
            ),
            core::cmp::Ordering::Equal,
        );
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                &packed_child,
                None,
                None,
                None,
                None,
                1,
            ),
            core::cmp::Ordering::Equal,
        );
    }

    #[test]
    fn test_path_comparison_helpers_cover_root_and_missing_best_fallbacks() {
        let current = [10_u32, 11, 20, 21];
        let best = [10_u32, 11, 20, 22];
        let empty_info: [SearchPathInfo; 0] = [];
        let mismatched_root_current = [SearchPathInfo {
            splitting_element: 4,
            certificate_index: 1,
            subcertificate_length: 3,
            eqref_hash: 2,
        }];
        let mismatched_root_best = [SearchPathInfo {
            splitting_element: 4,
            certificate_index: 0,
            subcertificate_length: 4,
            eqref_hash: 2,
        }];
        let empty_traces: Vec<Rc<RefinementTrace<()>>> = vec![];

        assert_eq!(
            packed_path_suffix_cmp(&current, &empty_info, &best, &empty_info, true),
            core::cmp::Ordering::Less,
        );
        assert_eq!(
            packed_path_suffix_cmp(
                &current,
                &mismatched_root_current,
                &best,
                &mismatched_root_best,
                false,
            ),
            current.cmp(&best),
        );
        assert_eq!(
            path_prefix_cmp::<()>(Some(&current), None, &empty_traces, None, None, None),
            core::cmp::Ordering::Greater,
        );
        assert_eq!(
            path_lex_cmp::<()>(Some(&current), None, &empty_traces, None, None, None),
            core::cmp::Ordering::Greater,
        );
        assert!(!path_prefix_equal_strict::<()>(
            Some(&current),
            None,
            &empty_traces,
            None,
            None,
            None,
        ));
        assert!(path_prefix_equal_strict(
            Some(&current[..2]),
            Some(&empty_info),
            &empty_traces,
            Some(&best),
            Some(&empty_info),
            None,
        ));
        assert!(!path_prefix_equal_strict(
            Some(&current),
            Some(&mismatched_root_current),
            &empty_traces,
            Some(&best),
            Some(&mismatched_root_best),
            None,
        ));
        assert!(!path_equal_strict::<()>(Some(&current), None, &empty_traces, None, None, None,));
        assert!(path_equal_strict(
            Some(&current),
            Some(&empty_info),
            &empty_traces,
            Some(&current),
            Some(&empty_info),
            None,
        ));
        assert!(!path_equal_strict(
            Some(&current),
            Some(&mismatched_root_current),
            &empty_traces,
            Some(&current),
            Some(&mismatched_root_best),
            None,
        ));
    }

    #[test]
    fn test_misc_search_helper_fallbacks_cover_trace_mismatch_and_prefix_utilities() {
        let packed_short = RefinementTrace {
            storage: RefinementTraceStorage::<u8>::Packed(vec![1_u32, 2]),
            subcertificate_length: 2,
            eqref_hash: 9,
        };
        let packed_long = RefinementTrace {
            storage: RefinementTraceStorage::<u8>::Packed(vec![1_u32, 2]),
            subcertificate_length: 3,
            eqref_hash: 9,
        };
        let event_empty = RefinementTrace {
            storage: RefinementTraceStorage::Events(Vec::<RefinementTraceEvent<u8>>::new()),
            subcertificate_length: 0,
            eqref_hash: 0,
        };
        assert_eq!(
            compare_refinement_trace(&packed_short, &packed_long),
            core::cmp::Ordering::Less,
        );
        assert!(!refinement_trace_equal(&packed_short, &event_empty));

        let mut partition = build_partition_from_groups(&[0, 0, 1, 2]);
        assert!(!refine_partition_according_to_unsigned_invariant_like_bliss(
            &mut partition,
            |element| usize::from(element < 2),
        ));
        let mut edge_label = |_: usize, _: usize| ();
        let target_cell = partition.cells().next().unwrap().id();
        assert_eq!(
            candidate_split_elements(
                &build_test_graph(4, &[(0, 1, 1)]),
                &mut edge_label,
                &mut partition,
                target_cell,
                &[],
                None,
                &mut CanonicalSearchStats::default(),
            ),
            vec![0, 1]
        );
        assert!(choice_path_is_prefix_of(Some(&[1, 2, 3]), &[1, 2]));
        assert!(!choice_path_is_prefix_of(Some(&[1, 3, 2]), &[1, 2]));
        assert_eq!(common_prefix_len(&[1, 2], None), 0);
    }

    #[test]
    fn test_unsigned_invariant_refinement_and_truncation_helpers() {
        let mut partition = BacktrackableOrderedPartition::new(6);
        assert!(refine_partition_according_to_unsigned_invariant_like_bliss(
            &mut partition,
            |element| element / 2,
        ));
        let cells = partition.cells().map(|cell| cell.elements().to_vec()).collect::<Vec<_>>();
        assert_eq!(cells, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);

        let mut packed_path = Some(vec![1_u32, 2, 3, 4]);
        truncate_packed_path(&mut packed_path, Some(2));
        assert_eq!(packed_path, Some(vec![1, 2]));
        truncate_packed_path(&mut packed_path, None);
        assert_eq!(packed_path, Some(vec![1, 2]));
    }

    #[test]
    fn test_search_outcome_snapshot_accessors_cover_outcome_storage() {
        let path_invariants = Rc::<[Rc<RefinementTrace<()>>]>::from(vec![packed_trace(&[7, 8], 2)]);
        let path_info = Rc::<[SearchPathInfo]>::from(vec![SearchPathInfo {
            splitting_element: 4,
            certificate_index: 0,
            subcertificate_length: 2,
            eqref_hash: 2,
        }]);
        let outcome = SearchOutcome {
            order: Rc::from(vec![2_usize, 0, 1]),
            leaf_signature: Some(Rc::new(UnlabeledLeafSignature {
                vertex_label_ids: Rc::from(vec![1_usize, 2, 3]),
                present_edge_offsets: Rc::from(vec![0_usize, 2]),
            })),
            path_invariants: Some(path_invariants.clone()),
            packed_path: Some(Rc::from(vec![7_u32, 8])),
            path_info: Some(path_info.clone()),
            sibling_orbits: KnownOrbits::new(3),
            choice_path: Rc::from(vec![4_usize]),
        };

        assert_eq!(SearchPathSnapshot::order(&outcome), &[2, 0, 1]);
        assert_eq!(
            SearchPathSnapshot::leaf_signature(&outcome)
                .expect("leaf signature should exist")
                .present_edge_offsets
                .as_ref(),
            &[0_usize, 2]
        );
        assert_eq!(
            SearchPathSnapshot::path_invariants(&outcome)
                .expect("path invariants should exist")
                .len(),
            1
        );
        assert_eq!(
            SearchPathSnapshot::packed_path(&outcome).expect("packed path should exist"),
            &[7_u32, 8]
        );
        assert_eq!(
            SearchPathSnapshot::path_info(&outcome).expect("path info should exist")[0]
                .splitting_element,
            4
        );
    }

    #[test]
    fn test_choose_target_cell_at_level_in_workspace_covers_wrapper_branch() {
        let graph = build_test_graph(6, &[]);
        let nodes = graph.node_ids().collect::<Vec<_>>();
        let partition = build_partition_from_groups(&[0, 0, 1, 1, 2, 3]);
        let candidates =
            partition.non_singleton_cells().map(PartitionCellView::id).collect::<Vec<_>>();

        let mut first_largest_workspace = SearchWorkspace::new(partition.order());
        let first_largest = choose_target_cell_at_level_in_workspace(
            &graph,
            &nodes,
            &partition,
            &mut first_largest_workspace,
            CanonSplittingHeuristic::FirstLargest,
            0,
        );
        let mut max_neighbour_workspace = SearchWorkspace::new(partition.order());
        let first_largest_max_neighbours = choose_target_cell_at_level_in_workspace(
            &graph,
            &nodes,
            &partition,
            &mut max_neighbour_workspace,
            CanonSplittingHeuristic::FirstLargestMaxNeighbours,
            0,
        );

        assert_eq!(first_largest, candidates[0]);
        assert_eq!(first_largest_max_neighbours, candidates[0]);
    }

    #[test]
    fn test_packed_helper_fallbacks_cover_missing_info_paths() {
        let packed_equal = RefinementTrace {
            storage: RefinementTraceStorage::<u8>::Packed(vec![20_u32, 21]),
            subcertificate_length: 2,
            eqref_hash: 2,
        };
        let same_packed = RefinementTrace {
            storage: RefinementTraceStorage::<u8>::Packed(vec![20_u32, 21]),
            subcertificate_length: 2,
            eqref_hash: 2,
        };

        assert!(refinement_trace_equal(&packed_equal, &same_packed));
        assert_eq!(
            child_path_prefix_cmp_to_reference(
                core::cmp::Ordering::Equal,
                &packed_equal,
                Some(2),
                Some(&[10_u32, 11, 20, 21]),
                None,
                None,
                1,
            ),
            core::cmp::Ordering::Equal,
        );
    }
}
