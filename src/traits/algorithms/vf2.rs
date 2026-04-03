//! VF2 builder and generic matching hooks.
//!
//! This module provides a generic VF2 public surface with structural and
//! semantic hooks.
//!
//! The current implementation keeps the public API small while moving the core
//! search closer to classical VF2:
//! incremental terminal-frontier state, prepared adjacency, and safe
//! future-neighbor pruning, without introducing domain-specific semantics.
//!
//! References:
//!
//! - Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, and Mario Vento (2001).
//!   *An improved algorithm for matching large graphs*. 3rd IAPR-TC15 Workshop
//!   on Graph-based Representations in Pattern Recognition.
//! - Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, and Mario Vento (2004).
//!   *A (sub)graph isomorphism algorithm for matching large graphs*. IEEE
//!   Transactions on Pattern Analysis and Machine Intelligence, 26(10),
//!   1367-1372.
//!
//! # Examples
//!
//! ```
//! use geometric_traits::{
//!     impls::{CSR2D, SortedVec, SymmetricCSR2D},
//!     prelude::*,
//!     traits::{EdgesBuilder, VocabularyBuilder},
//! };
//!
//! fn build_graph(node_count: usize, mut edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
//!     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
//!         .expected_number_of_symbols(node_count)
//!         .symbols((0..node_count).enumerate())
//!         .build()
//!         .unwrap();
//!     edges.sort_unstable();
//!     let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
//!         .expected_number_of_edges(edges.len())
//!         .expected_shape(node_count)
//!         .edges(edges.into_iter())
//!         .build()
//!         .unwrap();
//!     UndiGraph::from((nodes, edges))
//! }
//!
//! let query = build_graph(3, vec![(0, 1), (1, 2)]);
//! let target = build_graph(3, vec![(0, 1), (1, 2), (0, 2)]);
//!
//! assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
//! ```

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

/// Matching mode for the VF2 search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vf2Mode {
    /// Require a full graph isomorphism between query and target.
    Isomorphism,
    /// Require an induced subgraph isomorphism from query into target.
    InducedSubgraphIsomorphism,
    /// Require a non-induced subgraph isomorphism from query into target.
    ///
    /// In this crate's current simple-graph setting, this preserves query
    /// edges injectively but allows extra target edges among matched nodes.
    SubgraphIsomorphism,
    /// Require a graph monomorphism from query into target.
    ///
    /// In this crate's current simple-graph setting, this has the same
    /// feasibility semantics as [`Self::SubgraphIsomorphism`].
    Monomorphism,
}

/// A single VF2 node mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vf2Match<QueryNodeId, TargetNodeId> {
    pairs: Vec<(QueryNodeId, TargetNodeId)>,
}

impl<QueryNodeId, TargetNodeId> Vf2Match<QueryNodeId, TargetNodeId> {
    /// Creates a new owned VF2 match.
    #[inline]
    #[must_use]
    pub fn new(pairs: Vec<(QueryNodeId, TargetNodeId)>) -> Self {
        Self { pairs }
    }

    /// Returns the matched node pairs.
    #[inline]
    #[must_use]
    pub fn pairs(&self) -> &[(QueryNodeId, TargetNodeId)] {
        &self.pairs
    }

    /// Consumes the match and returns the inner node pairs.
    #[inline]
    #[must_use]
    pub fn into_pairs(self) -> Vec<(QueryNodeId, TargetNodeId)> {
        self.pairs
    }

    /// Returns the number of matched node pairs.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns whether the match contains no node pairs.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Predicate over candidate query/target node pairs.
///
/// Implementations should be pure and deterministic. The search may evaluate
/// the same candidate pair more than once while pruning.
pub trait Vf2NodeMatcher<QueryNodeId, TargetNodeId> {
    /// Returns whether the query node may match the target node.
    fn matches(&self, query_node: QueryNodeId, target_node: TargetNodeId) -> bool;
}

/// Default node matcher that accepts every node pair.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptAllNodeMatcher;

impl<QueryNodeId, TargetNodeId> Vf2NodeMatcher<QueryNodeId, TargetNodeId> for AcceptAllNodeMatcher {
    #[inline]
    fn matches(&self, _query_node: QueryNodeId, _target_node: TargetNodeId) -> bool {
        true
    }
}

/// Wrapper for custom node matcher closures.
pub struct CustomNodeMatcher<F>(pub F);

impl<QueryNodeId, TargetNodeId, F> Vf2NodeMatcher<QueryNodeId, TargetNodeId>
    for CustomNodeMatcher<F>
where
    F: Fn(QueryNodeId, TargetNodeId) -> bool,
{
    #[inline]
    fn matches(&self, query_node: QueryNodeId, target_node: TargetNodeId) -> bool {
        (self.0)(query_node, target_node)
    }
}

/// Predicate over candidate query/target edge pairs.
///
/// Implementations should be pure and deterministic. The search may evaluate
/// the same logical edge more than once while pruning; for undirected graphs
/// both orientations may be checked.
pub trait Vf2EdgeMatcher<QueryNodeId, TargetNodeId> {
    /// Returns whether the query edge may match the target edge.
    fn matches(
        &self,
        query_source: QueryNodeId,
        query_destination: QueryNodeId,
        target_source: TargetNodeId,
        target_destination: TargetNodeId,
    ) -> bool;
}

/// Default edge matcher that accepts every edge pair.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptAllEdgeMatcher;

impl<QueryNodeId, TargetNodeId> Vf2EdgeMatcher<QueryNodeId, TargetNodeId> for AcceptAllEdgeMatcher {
    #[inline]
    fn matches(
        &self,
        _query_source: QueryNodeId,
        _query_destination: QueryNodeId,
        _target_source: TargetNodeId,
        _target_destination: TargetNodeId,
    ) -> bool {
        true
    }
}

/// Wrapper for custom edge matcher closures.
pub struct CustomEdgeMatcher<F>(pub F);

impl<QueryNodeId, TargetNodeId, F> Vf2EdgeMatcher<QueryNodeId, TargetNodeId>
    for CustomEdgeMatcher<F>
where
    F: Fn(QueryNodeId, QueryNodeId, TargetNodeId, TargetNodeId) -> bool,
{
    #[inline]
    fn matches(
        &self,
        query_source: QueryNodeId,
        query_destination: QueryNodeId,
        target_source: TargetNodeId,
        target_destination: TargetNodeId,
    ) -> bool {
        (self.0)(query_source, query_destination, target_source, target_destination)
    }
}

/// Predicate over a complete VF2 mapping.
pub trait Vf2FinalMatcher<QueryNodeId, TargetNodeId> {
    /// Returns whether the full mapping is accepted.
    fn matches(&self, mapping: &[(QueryNodeId, TargetNodeId)]) -> bool;
}

/// Default final matcher that accepts every mapping.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptAllFinalMatcher;

impl<QueryNodeId, TargetNodeId> Vf2FinalMatcher<QueryNodeId, TargetNodeId>
    for AcceptAllFinalMatcher
{
    #[inline]
    fn matches(&self, _mapping: &[(QueryNodeId, TargetNodeId)]) -> bool {
        true
    }
}

/// Wrapper for custom final matcher closures.
pub struct CustomFinalMatcher<F>(pub F);

impl<QueryNodeId, TargetNodeId, F> Vf2FinalMatcher<QueryNodeId, TargetNodeId>
    for CustomFinalMatcher<F>
where
    F: Fn(&[(QueryNodeId, TargetNodeId)]) -> bool,
{
    #[inline]
    fn matches(&self, mapping: &[(QueryNodeId, TargetNodeId)]) -> bool {
        (self.0)(mapping)
    }
}

/// VF2 search builder.
pub struct Vf2Builder<
    'g,
    Query,
    Target,
    NodeMatch = AcceptAllNodeMatcher,
    EdgeMatch = AcceptAllEdgeMatcher,
    FinalMatch = AcceptAllFinalMatcher,
> {
    query: &'g Query,
    target: &'g Target,
    mode: Vf2Mode,
    node_match: NodeMatch,
    edge_match: EdgeMatch,
    final_match: FinalMatch,
}

/// Reusable graph-local VF2 preprocessing.
///
/// This holds data derived from a single graph only. It can be reused across
/// multiple top-level VF2 runs against different counterpart graphs or
/// different semantic predicates.
pub struct PreparedVf2Graph<NodeId> {
    prepared: PreparedGraph<NodeId>,
    number_of_edges: usize,
    number_of_self_loops: usize,
}

impl<NodeId> PreparedVf2Graph<NodeId>
where
    NodeId: Copy + AsPrimitive<usize>,
{
    fn new<Graph>(graph: &Graph) -> Self
    where
        Graph: MonoplexMonopartiteGraph<NodeId = NodeId>,
    {
        Self {
            prepared: PreparedGraph::new(graph),
            number_of_edges: graph.number_of_edges().as_(),
            number_of_self_loops: graph.number_of_self_loops().as_(),
        }
    }

    #[inline]
    fn prepared(&self) -> &PreparedGraph<NodeId> {
        &self.prepared
    }

    /// Returns the number of nodes in the prepared graph.
    #[inline]
    #[must_use]
    pub fn number_of_nodes(&self) -> usize {
        self.prepared.node_ids.len()
    }

    /// Returns the number of edges in the prepared graph.
    #[inline]
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.number_of_edges
    }

    /// Returns the number of self-loops in the prepared graph.
    #[inline]
    #[must_use]
    pub fn number_of_self_loops(&self) -> usize {
        self.number_of_self_loops
    }

    /// Returns a VF2 builder using `self` as the prepared query graph.
    ///
    /// Use this when the same prepared graph will participate in multiple
    /// top-level VF2 runs.
    #[inline]
    #[must_use]
    pub fn vf2<'g, TargetNodeId>(
        &'g self,
        target: &'g PreparedVf2Graph<TargetNodeId>,
    ) -> PreparedVf2Builder<'g, NodeId, TargetNodeId>
    where
        TargetNodeId: Copy + AsPrimitive<usize>,
    {
        PreparedVf2Builder::new(self, target)
    }
}

/// Trait providing a reusable graph-local VF2 preparation step.
pub trait PrepareVf2: MonoplexMonopartiteGraph {
    /// Precomputes reusable graph-local VF2 data.
    ///
    /// This is useful when the same graph will be queried repeatedly, for
    /// example when checking `has_match()` first and then enumerating matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// fn build_graph(node_count: usize, mut edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(node_count)
    ///         .symbols((0..node_count).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     edges.sort_unstable();
    ///     let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape(node_count)
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, edges))
    /// }
    ///
    /// let query = build_graph(3, vec![(0, 1), (1, 2)]);
    /// let target = build_graph(3, vec![(0, 1), (1, 2), (0, 2)]);
    ///
    /// let prepared_query = query.prepare_vf2();
    /// let prepared_target = target.prepare_vf2();
    /// let matcher = prepared_query.vf2(&prepared_target).with_mode(Vf2Mode::SubgraphIsomorphism);
    ///
    /// assert!(matcher.has_match());
    /// assert!(matcher.first_match().is_some());
    /// ```
    #[inline]
    #[must_use]
    fn prepare_vf2(&self) -> PreparedVf2Graph<Self::NodeId>
    where
        Self: Sized,
        Self::NodeId: Copy + AsPrimitive<usize>,
    {
        PreparedVf2Graph::new(self)
    }
}

impl<G> PrepareVf2 for G where G: MonoplexMonopartiteGraph {}

/// VF2 search builder over preprocessed graphs.
pub struct PreparedVf2Builder<
    'g,
    QueryNodeId,
    TargetNodeId,
    NodeMatch = AcceptAllNodeMatcher,
    EdgeMatch = AcceptAllEdgeMatcher,
    FinalMatch = AcceptAllFinalMatcher,
> {
    query: &'g PreparedVf2Graph<QueryNodeId>,
    target: &'g PreparedVf2Graph<TargetNodeId>,
    mode: Vf2Mode,
    node_match: NodeMatch,
    edge_match: EdgeMatch,
    final_match: FinalMatch,
}

impl<'g, QueryNodeId, TargetNodeId> PreparedVf2Builder<'g, QueryNodeId, TargetNodeId>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    /// Creates a new prepared VF2 builder with permissive default hooks.
    #[inline]
    #[must_use]
    pub fn new(
        query: &'g PreparedVf2Graph<QueryNodeId>,
        target: &'g PreparedVf2Graph<TargetNodeId>,
    ) -> Self {
        Self {
            query,
            target,
            mode: Vf2Mode::Isomorphism,
            node_match: AcceptAllNodeMatcher,
            edge_match: AcceptAllEdgeMatcher,
            final_match: AcceptAllFinalMatcher,
        }
    }
}

impl<'g, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
    PreparedVf2Builder<'g, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    /// Sets the VF2 search mode.
    #[inline]
    #[must_use]
    pub fn with_mode(mut self, mode: Vf2Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Replaces the node matcher.
    ///
    /// The matcher is treated as a pure predicate. The search may evaluate the
    /// same candidate pair more than once while pruning, and call count is not
    /// part of the API contract.
    #[inline]
    #[must_use]
    pub fn with_node_match<F>(
        self,
        node_match: F,
    ) -> PreparedVf2Builder<
        'g,
        QueryNodeId,
        TargetNodeId,
        CustomNodeMatcher<F>,
        EdgeMatch,
        FinalMatch,
    >
    where
        F: Fn(QueryNodeId, TargetNodeId) -> bool,
    {
        PreparedVf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: CustomNodeMatcher(node_match),
            edge_match: self.edge_match,
            final_match: self.final_match,
        }
    }

    /// Replaces the edge matcher.
    ///
    /// The matcher is treated as a pure predicate over oriented edge
    /// candidates. The search may evaluate the same logical edge more than
    /// once while pruning, and undirected graphs may observe both
    /// orientations.
    #[inline]
    #[must_use]
    pub fn with_edge_match<F>(
        self,
        edge_match: F,
    ) -> PreparedVf2Builder<
        'g,
        QueryNodeId,
        TargetNodeId,
        NodeMatch,
        CustomEdgeMatcher<F>,
        FinalMatch,
    >
    where
        F: Fn(QueryNodeId, QueryNodeId, TargetNodeId, TargetNodeId) -> bool,
    {
        PreparedVf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: self.node_match,
            edge_match: CustomEdgeMatcher(edge_match),
            final_match: self.final_match,
        }
    }

    /// Replaces the final-match predicate.
    #[inline]
    #[must_use]
    pub fn with_final_match<F>(
        self,
        final_match: F,
    ) -> PreparedVf2Builder<
        'g,
        QueryNodeId,
        TargetNodeId,
        NodeMatch,
        EdgeMatch,
        CustomFinalMatcher<F>,
    >
    where
        F: Fn(&[(QueryNodeId, TargetNodeId)]) -> bool,
    {
        PreparedVf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: self.node_match,
            edge_match: self.edge_match,
            final_match: CustomFinalMatcher(final_match),
        }
    }

    /// Returns the configured prepared query graph.
    #[inline]
    #[must_use]
    pub fn query(&self) -> &'g PreparedVf2Graph<QueryNodeId> {
        self.query
    }

    /// Returns the configured prepared target graph.
    #[inline]
    #[must_use]
    pub fn target(&self) -> &'g PreparedVf2Graph<TargetNodeId> {
        self.target
    }

    /// Returns the configured search mode.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> Vf2Mode {
        self.mode
    }
}

impl<'g, Query, Target> Vf2Builder<'g, Query, Target> {
    /// Creates a new VF2 builder with permissive default hooks.
    #[inline]
    #[must_use]
    pub fn new(query: &'g Query, target: &'g Target) -> Self {
        Self {
            query,
            target,
            mode: Vf2Mode::Isomorphism,
            node_match: AcceptAllNodeMatcher,
            edge_match: AcceptAllEdgeMatcher,
            final_match: AcceptAllFinalMatcher,
        }
    }
}

impl<'g, Query, Target, NodeMatch, EdgeMatch, FinalMatch>
    Vf2Builder<'g, Query, Target, NodeMatch, EdgeMatch, FinalMatch>
{
    /// Sets the VF2 search mode.
    #[inline]
    #[must_use]
    pub fn with_mode(mut self, mode: Vf2Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Replaces the node matcher.
    ///
    /// The matcher is treated as a pure predicate. The search may evaluate the
    /// same candidate pair more than once while pruning, and call count is not
    /// part of the API contract.
    #[inline]
    #[must_use]
    pub fn with_node_match<F>(
        self,
        node_match: F,
    ) -> Vf2Builder<'g, Query, Target, CustomNodeMatcher<F>, EdgeMatch, FinalMatch>
    where
        F: Fn(
            <Query as crate::traits::MonopartiteGraph>::NodeId,
            <Target as crate::traits::MonopartiteGraph>::NodeId,
        ) -> bool,
        Query: MonoplexMonopartiteGraph,
        Target: MonoplexMonopartiteGraph,
    {
        Vf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: CustomNodeMatcher(node_match),
            edge_match: self.edge_match,
            final_match: self.final_match,
        }
    }

    /// Replaces the edge matcher.
    ///
    /// The matcher is treated as a pure predicate over oriented edge
    /// candidates. The search may evaluate the same logical edge more than
    /// once while pruning, and undirected graphs may observe both
    /// orientations.
    #[inline]
    #[must_use]
    pub fn with_edge_match<F>(
        self,
        edge_match: F,
    ) -> Vf2Builder<'g, Query, Target, NodeMatch, CustomEdgeMatcher<F>, FinalMatch>
    where
        F: Fn(
            <Query as crate::traits::MonopartiteGraph>::NodeId,
            <Query as crate::traits::MonopartiteGraph>::NodeId,
            <Target as crate::traits::MonopartiteGraph>::NodeId,
            <Target as crate::traits::MonopartiteGraph>::NodeId,
        ) -> bool,
        Query: MonoplexMonopartiteGraph,
        Target: MonoplexMonopartiteGraph,
    {
        Vf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: self.node_match,
            edge_match: CustomEdgeMatcher(edge_match),
            final_match: self.final_match,
        }
    }

    /// Replaces the final-match predicate.
    #[inline]
    #[must_use]
    pub fn with_final_match<F>(
        self,
        final_match: F,
    ) -> Vf2Builder<'g, Query, Target, NodeMatch, EdgeMatch, CustomFinalMatcher<F>>
    where
        F: Fn(
            &[(
                <Query as crate::traits::MonopartiteGraph>::NodeId,
                <Target as crate::traits::MonopartiteGraph>::NodeId,
            )],
        ) -> bool,
        Query: MonoplexMonopartiteGraph,
        Target: MonoplexMonopartiteGraph,
    {
        Vf2Builder {
            query: self.query,
            target: self.target,
            mode: self.mode,
            node_match: self.node_match,
            edge_match: self.edge_match,
            final_match: CustomFinalMatcher(final_match),
        }
    }

    /// Returns the configured query graph.
    #[inline]
    #[must_use]
    pub fn query(&self) -> &'g Query {
        self.query
    }

    /// Returns the configured target graph.
    #[inline]
    #[must_use]
    pub fn target(&self) -> &'g Target {
        self.target
    }

    /// Returns the configured search mode.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> Vf2Mode {
        self.mode
    }
}

#[inline]
fn passes_global_precheck<Query, Target>(query: &Query, target: &Target, mode: Vf2Mode) -> bool
where
    Query: MonoplexMonopartiteGraph,
    Target: MonoplexMonopartiteGraph,
{
    passes_global_precheck_counts(
        query.number_of_nodes().as_(),
        target.number_of_nodes().as_(),
        query.number_of_edges().as_(),
        target.number_of_edges().as_(),
        query.number_of_self_loops().as_(),
        target.number_of_self_loops().as_(),
        mode,
    )
}

#[inline]
fn passes_global_precheck_counts(
    query_nodes: usize,
    target_nodes: usize,
    query_edges: usize,
    target_edges: usize,
    query_loops: usize,
    target_loops: usize,
    mode: Vf2Mode,
) -> bool {
    match mode {
        Vf2Mode::Isomorphism => {
            query_nodes == target_nodes
                && query_edges == target_edges
                && query_loops == target_loops
        }
        Vf2Mode::InducedSubgraphIsomorphism
        | Vf2Mode::SubgraphIsomorphism
        | Vf2Mode::Monomorphism => {
            query_nodes <= target_nodes
                && query_edges <= target_edges
                && query_loops <= target_loops
        }
    }
}

struct PreparedGraph<NodeId> {
    node_ids: Vec<NodeId>,
    selection_order: Vec<NodeId>,
    successors: Vec<Vec<NodeId>>,
    predecessors: Vec<Vec<NodeId>>,
    out_degrees: Vec<usize>,
    in_degrees: Vec<usize>,
    self_loops: Vec<bool>,
}

impl<NodeId> PreparedGraph<NodeId>
where
    NodeId: Copy + AsPrimitive<usize>,
{
    fn new<Graph>(graph: &Graph) -> Self
    where
        Graph: MonoplexMonopartiteGraph<NodeId = NodeId>,
    {
        let node_ids: Vec<NodeId> = graph.node_ids().collect();
        let node_count = node_ids.len();
        let mut successors = vec![Vec::new(); node_count];
        let mut predecessors = vec![Vec::new(); node_count];
        let mut self_loops = vec![false; node_count];

        for node in node_ids.iter().copied() {
            let node_index = node.as_();
            for successor in graph.successors(node) {
                successors[node_index].push(successor);
                predecessors[successor.as_()].push(node);
                if successor.as_() == node_index {
                    self_loops[node_index] = true;
                }
            }
        }

        for neighbors in &mut successors {
            neighbors.sort_unstable_by_key(|node_id| node_id.as_());
        }
        for neighbors in &mut predecessors {
            neighbors.sort_unstable_by_key(|node_id| node_id.as_());
        }

        let out_degrees: Vec<usize> = successors.iter().map(Vec::len).collect();
        let in_degrees: Vec<usize> = predecessors.iter().map(Vec::len).collect();
        let mut selection_order = node_ids.clone();
        selection_order.sort_unstable_by_key(|&node_id| {
            (core::cmp::Reverse(out_degrees[node_id.as_()]), node_id.as_())
        });

        Self {
            node_ids,
            selection_order,
            successors,
            predecessors,
            out_degrees,
            in_degrees,
            self_loops,
        }
    }

    #[inline]
    fn out_degree(&self, node: NodeId) -> usize {
        self.out_degrees[node.as_()]
    }

    #[inline]
    fn in_degree(&self, node: NodeId) -> usize {
        self.in_degrees[node.as_()]
    }

    #[inline]
    fn has_self_loop(&self, node: NodeId) -> bool {
        self.self_loops[node.as_()]
    }

    #[inline]
    fn has_successor(&self, source: NodeId, target: NodeId) -> bool {
        self.successors[source.as_()]
            .binary_search_by_key(&target.as_(), |node_id| node_id.as_())
            .is_ok()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FrontierSnapshot {
    query_unmatched_head: usize,
    query_in_depth_touched_len: usize,
    query_out_depth_touched_len: usize,
    target_in_depth_touched_len: usize,
    target_out_depth_touched_len: usize,
}

struct Vf2State<QueryNodeId, TargetNodeId> {
    query_core: Vec<Option<TargetNodeId>>,
    target_core: Vec<Option<QueryNodeId>>,
    target_unmapped_nodes: Vec<TargetNodeId>,
    target_unmapped_positions: Vec<usize>,
    target_unmapped_len: usize,
    query_unmatched_head: usize,
    query_in_depth: Vec<usize>,
    query_out_depth: Vec<usize>,
    target_in_depth: Vec<usize>,
    target_out_depth: Vec<usize>,
    query_in_frontier: Vec<QueryNodeId>,
    query_in_frontier_positions: Vec<usize>,
    query_out_frontier: Vec<QueryNodeId>,
    query_out_frontier_positions: Vec<usize>,
    target_in_frontier: Vec<TargetNodeId>,
    target_in_frontier_positions: Vec<usize>,
    target_out_frontier: Vec<TargetNodeId>,
    target_out_frontier_positions: Vec<usize>,
    query_in_depth_touched: Vec<usize>,
    query_out_depth_touched: Vec<usize>,
    target_in_depth_touched: Vec<usize>,
    target_out_depth_touched: Vec<usize>,
    frontier_snapshots: Vec<FrontierSnapshot>,
    query_unmapped_in_count: usize,
    query_unmapped_out_count: usize,
    target_unmapped_in_count: usize,
    target_unmapped_out_count: usize,
    query_mapped_neighbor_counts: Vec<MappedNeighborCounts>,
    target_mapped_neighbor_counts: Vec<MappedNeighborCounts>,
    query_mapped_predecessors: Vec<Vec<QueryNodeId>>,
    query_mapped_successors: Vec<Vec<QueryNodeId>>,
    query_future_counts: Vec<FutureNeighborCounts>,
    target_future_counts: Vec<FutureNeighborCounts>,
    candidate_scratch: Vec<Vec<TargetNodeId>>,
    mapping: Vec<(QueryNodeId, TargetNodeId)>,
}

impl<QueryNodeId, TargetNodeId> Vf2State<QueryNodeId, TargetNodeId>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    #[inline]
    fn new(query: &PreparedGraph<QueryNodeId>, target: &PreparedGraph<TargetNodeId>) -> Self {
        let query_node_ids = &query.node_ids;
        let target_node_ids = &target.node_ids;
        let query_nodes = query_node_ids.len();
        let target_nodes = target_node_ids.len();
        let mut target_unmapped_positions = vec![0; target_nodes];
        let mut candidate_scratch = Vec::with_capacity(query_nodes + 1);
        let query_core = vec![None; query_nodes];
        let target_core = vec![None; target_nodes];
        let query_in_depth = vec![0; query_nodes];
        let query_out_depth = vec![0; query_nodes];
        let target_in_depth = vec![0; target_nodes];
        let target_out_depth = vec![0; target_nodes];
        let query_mapped_neighbor_counts = vec![MappedNeighborCounts::default(); query_nodes];
        let target_mapped_neighbor_counts = vec![MappedNeighborCounts::default(); target_nodes];
        let query_mapped_predecessors = vec![Vec::new(); query_nodes];
        let query_mapped_successors = vec![Vec::new(); query_nodes];
        let mut query_future_counts = vec![FutureNeighborCounts::default(); query_nodes];
        let mut target_future_counts = vec![FutureNeighborCounts::default(); target_nodes];

        for (position, &node_id) in target_node_ids.iter().enumerate() {
            target_unmapped_positions[node_id.as_()] = position;
        }
        for _ in 0..=query_nodes {
            candidate_scratch.push(Vec::new());
        }
        for query_node_index in 0..query_nodes {
            adjust_unmapped_node_future_contribution(
                query,
                &query_core,
                &query_in_depth,
                &query_out_depth,
                &mut query_future_counts,
                query_node_index,
                ContributionDelta::Add,
            );
        }
        for target_node_index in 0..target_nodes {
            adjust_unmapped_node_future_contribution(
                target,
                &target_core,
                &target_in_depth,
                &target_out_depth,
                &mut target_future_counts,
                target_node_index,
                ContributionDelta::Add,
            );
        }

        Self {
            query_core,
            target_core,
            target_unmapped_nodes: target_node_ids.clone(),
            target_unmapped_positions,
            target_unmapped_len: target_nodes,
            query_unmatched_head: 0,
            query_in_depth,
            query_out_depth,
            target_in_depth,
            target_out_depth,
            query_in_frontier: Vec::with_capacity(query_nodes),
            query_in_frontier_positions: vec![usize::MAX; query_nodes],
            query_out_frontier: Vec::with_capacity(query_nodes),
            query_out_frontier_positions: vec![usize::MAX; query_nodes],
            target_in_frontier: Vec::with_capacity(target_nodes),
            target_in_frontier_positions: vec![usize::MAX; target_nodes],
            target_out_frontier: Vec::with_capacity(target_nodes),
            target_out_frontier_positions: vec![usize::MAX; target_nodes],
            query_in_depth_touched: Vec::with_capacity(query_nodes),
            query_out_depth_touched: Vec::with_capacity(query_nodes),
            target_in_depth_touched: Vec::with_capacity(target_nodes),
            target_out_depth_touched: Vec::with_capacity(target_nodes),
            frontier_snapshots: Vec::with_capacity(query_nodes),
            query_unmapped_in_count: 0,
            query_unmapped_out_count: 0,
            target_unmapped_in_count: 0,
            target_unmapped_out_count: 0,
            query_mapped_neighbor_counts,
            target_mapped_neighbor_counts,
            query_mapped_predecessors,
            query_mapped_successors,
            query_future_counts,
            target_future_counts,
            candidate_scratch,
            mapping: Vec::with_capacity(query_nodes),
        }
    }

    #[inline]
    fn remove_target_unmapped_node(&mut self, target_node: TargetNodeId) {
        let target_node_index = target_node.as_();
        let node_position = self.target_unmapped_positions[target_node_index];
        debug_assert!(node_position < self.target_unmapped_len);
        let last_position = self.target_unmapped_len - 1;
        let swapped_node = self.target_unmapped_nodes[last_position];

        self.target_unmapped_nodes.swap(node_position, last_position);
        self.target_unmapped_positions[swapped_node.as_()] = node_position;
        self.target_unmapped_positions[target_node_index] = last_position;
        self.target_unmapped_len -= 1;
    }

    #[inline]
    fn add_query_in_frontier(&mut self, query_node: QueryNodeId) {
        add_frontier_node(
            &mut self.query_in_frontier,
            &mut self.query_in_frontier_positions,
            query_node,
        );
    }

    #[inline]
    fn add_query_out_frontier(&mut self, query_node: QueryNodeId) {
        add_frontier_node(
            &mut self.query_out_frontier,
            &mut self.query_out_frontier_positions,
            query_node,
        );
    }

    #[inline]
    fn add_target_in_frontier(&mut self, target_node: TargetNodeId) {
        add_frontier_node(
            &mut self.target_in_frontier,
            &mut self.target_in_frontier_positions,
            target_node,
        );
    }

    #[inline]
    fn add_target_out_frontier(&mut self, target_node: TargetNodeId) {
        add_frontier_node(
            &mut self.target_out_frontier,
            &mut self.target_out_frontier_positions,
            target_node,
        );
    }

    #[inline]
    fn remove_query_in_frontier(&mut self, query_node: QueryNodeId) {
        remove_frontier_node(
            &mut self.query_in_frontier,
            &mut self.query_in_frontier_positions,
            query_node,
        );
    }

    #[inline]
    fn remove_query_out_frontier(&mut self, query_node: QueryNodeId) {
        remove_frontier_node(
            &mut self.query_out_frontier,
            &mut self.query_out_frontier_positions,
            query_node,
        );
    }

    #[inline]
    fn remove_target_in_frontier(&mut self, target_node: TargetNodeId) {
        remove_frontier_node(
            &mut self.target_in_frontier,
            &mut self.target_in_frontier_positions,
            target_node,
        );
    }

    #[inline]
    fn remove_target_out_frontier(&mut self, target_node: TargetNodeId) {
        remove_frontier_node(
            &mut self.target_out_frontier,
            &mut self.target_out_frontier_positions,
            target_node,
        );
    }

    #[inline]
    fn advance_query_unmatched_head(&mut self, selection_order: &[QueryNodeId]) {
        while let Some(&query_node) = selection_order.get(self.query_unmatched_head) {
            if !self.is_query_mapped(query_node) {
                break;
            }
            self.query_unmatched_head += 1;
        }
    }

    #[inline]
    fn is_query_mapped(&self, query_node: QueryNodeId) -> bool {
        self.query_core[query_node.as_()].is_some()
    }

    #[cfg(test)]
    #[inline]
    fn is_query_in_terminal(&self, query_node: QueryNodeId) -> bool {
        !self.is_query_mapped(query_node) && self.query_in_depth[query_node.as_()] != 0
    }

    #[cfg(test)]
    #[inline]
    fn is_query_out_terminal(&self, query_node: QueryNodeId) -> bool {
        !self.is_query_mapped(query_node) && self.query_out_depth[query_node.as_()] != 0
    }

    #[cfg(test)]
    #[inline]
    fn is_query_terminal(&self, query_node: QueryNodeId) -> bool {
        self.is_query_in_terminal(query_node) || self.is_query_out_terminal(query_node)
    }

    #[inline]
    fn has_unmapped_query_in(&self) -> bool {
        self.query_unmapped_in_count != 0
    }

    #[inline]
    fn has_unmapped_query_out(&self) -> bool {
        self.query_unmapped_out_count != 0
    }

    #[inline]
    fn has_unmapped_target_in(&self) -> bool {
        self.target_unmapped_in_count != 0
    }

    #[inline]
    fn has_unmapped_target_out(&self) -> bool {
        self.target_unmapped_out_count != 0
    }

    #[inline]
    fn begin_query_mapping(&mut self, query: &PreparedGraph<QueryNodeId>, query_node: QueryNodeId) {
        let query_node_index = query_node.as_();
        adjust_unmapped_node_future_contribution(
            query,
            &self.query_core,
            &self.query_in_depth,
            &self.query_out_depth,
            &mut self.query_future_counts,
            query_node_index,
            ContributionDelta::Remove,
        );
        if self.query_in_depth[query_node_index] != 0 {
            debug_assert!(self.query_unmapped_in_count != 0);
            self.query_unmapped_in_count -= 1;
        }
        if self.query_out_depth[query_node_index] != 0 {
            debug_assert!(self.query_unmapped_out_count != 0);
            self.query_unmapped_out_count -= 1;
        }
        self.remove_query_in_frontier(query_node);
        self.remove_query_out_frontier(query_node);
    }

    #[inline]
    fn begin_target_mapping(
        &mut self,
        target: &PreparedGraph<TargetNodeId>,
        target_node: TargetNodeId,
    ) {
        let target_node_index = target_node.as_();
        adjust_unmapped_node_future_contribution(
            target,
            &self.target_core,
            &self.target_in_depth,
            &self.target_out_depth,
            &mut self.target_future_counts,
            target_node_index,
            ContributionDelta::Remove,
        );
        if self.target_in_depth[target_node_index] != 0 {
            debug_assert!(self.target_unmapped_in_count != 0);
            self.target_unmapped_in_count -= 1;
        }
        if self.target_out_depth[target_node_index] != 0 {
            debug_assert!(self.target_unmapped_out_count != 0);
            self.target_unmapped_out_count -= 1;
        }
        self.remove_target_in_frontier(target_node);
        self.remove_target_out_frontier(target_node);
        self.remove_target_unmapped_node(target_node);
    }

    #[inline]
    fn finish_query_unmapping(&mut self, query_node: QueryNodeId) {
        let query_node_index = query_node.as_();
        if self.query_in_depth[query_node_index] != 0 {
            self.query_unmapped_in_count += 1;
            self.add_query_in_frontier(query_node);
        }
        if self.query_out_depth[query_node_index] != 0 {
            self.query_unmapped_out_count += 1;
            self.add_query_out_frontier(query_node);
        }
    }

    #[inline]
    fn finish_target_unmapping(&mut self, target_node: TargetNodeId) {
        let target_node_index = target_node.as_();
        debug_assert_eq!(
            self.target_unmapped_positions[target_node_index],
            self.target_unmapped_len
        );
        self.target_unmapped_len += 1;
        if self.target_in_depth[target_node_index] != 0 {
            self.target_unmapped_in_count += 1;
            self.add_target_in_frontier(target_node);
        }
        if self.target_out_depth[target_node_index] != 0 {
            self.target_unmapped_out_count += 1;
            self.add_target_out_frontier(target_node);
        }
    }
}

#[inline]
fn add_frontier_node<NodeId>(
    frontier: &mut Vec<NodeId>,
    frontier_positions: &mut [usize],
    node: NodeId,
) where
    NodeId: Copy + AsPrimitive<usize>,
{
    let node_index = node.as_();
    if frontier_positions[node_index] != usize::MAX {
        return;
    }
    frontier_positions[node_index] = frontier.len();
    frontier.push(node);
}

#[inline]
fn remove_frontier_node<NodeId>(
    frontier: &mut Vec<NodeId>,
    frontier_positions: &mut [usize],
    node: NodeId,
) where
    NodeId: Copy + AsPrimitive<usize>,
{
    remove_frontier_node_at_index(frontier, frontier_positions, node.as_());
}

#[inline]
fn remove_frontier_node_at_index<NodeId>(
    frontier: &mut Vec<NodeId>,
    frontier_positions: &mut [usize],
    node_index: usize,
) where
    NodeId: Copy + AsPrimitive<usize>,
{
    let node_position = frontier_positions[node_index];
    if node_position == usize::MAX {
        return;
    }

    let _ = frontier.swap_remove(node_position);
    if node_position < frontier.len() {
        frontier_positions[frontier[node_position].as_()] = node_position;
    }
    frontier_positions[node_index] = usize::MAX;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateClass {
    Out,
    In,
    Unmatched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchControl {
    Continue,
    Stop,
}

trait Vf2SearchMethods<QueryNodeId: Copy, TargetNodeId: Copy> {
    fn search_with_mapping<F>(&self, visitor: F) -> SearchControl
    where
        F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool;

    fn has_match_impl(&self) -> bool {
        let mut found_match = false;
        let _ = self.search_with_mapping(|_| {
            found_match = true;
            false
        });
        found_match
    }

    fn first_match_impl(&self) -> Option<Vf2Match<QueryNodeId, TargetNodeId>> {
        let mut first_match = None;
        let _ = self.search_with_mapping(|mapping| {
            first_match = Some(Vf2Match::new(mapping.to_vec()));
            false
        });
        first_match
    }

    fn for_each_mapping_impl<F>(&self, visitor: F) -> bool
    where
        F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
    {
        matches!(self.search_with_mapping(visitor), SearchControl::Continue)
    }

    fn for_each_match_impl<F>(&self, mut visitor: F) -> bool
    where
        F: FnMut(&Vf2Match<QueryNodeId, TargetNodeId>) -> bool,
    {
        self.for_each_mapping_impl(|mapping| visitor(&Vf2Match::new(mapping.to_vec())))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FutureNeighborCounts {
    predecessor_in: usize,
    successor_in: usize,
    predecessor_out: usize,
    successor_out: usize,
    predecessor_new: usize,
    successor_new: usize,
    predecessor_total: usize,
    successor_total: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MappedNeighborCounts {
    predecessors: usize,
    successors: usize,
}

impl FutureNeighborCounts {
    #[inline]
    fn matches(self, other: Self, mode: Vf2Mode) -> bool {
        match mode {
            Vf2Mode::Isomorphism => self == other,
            Vf2Mode::InducedSubgraphIsomorphism => {
                self.predecessor_in <= other.predecessor_in
                    && self.successor_in <= other.successor_in
                    && self.predecessor_out <= other.predecessor_out
                    && self.successor_out <= other.successor_out
                    && self.predecessor_new <= other.predecessor_new
                    && self.successor_new <= other.successor_new
                    && self.predecessor_total <= other.predecessor_total
                    && self.successor_total <= other.successor_total
            }
            Vf2Mode::SubgraphIsomorphism | Vf2Mode::Monomorphism => {
                self.predecessor_in <= other.predecessor_in
                    && self.successor_in <= other.successor_in
                    && self.predecessor_out <= other.predecessor_out
                    && self.successor_out <= other.successor_out
                    && self.predecessor_total <= other.predecessor_total
                    && self.successor_total <= other.successor_total
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContributionDelta {
    Add,
    Remove,
}

struct TerminalSideMut<'a, NodeId> {
    frontier: &'a mut Vec<NodeId>,
    frontier_positions: &'a mut [usize],
    depth: &'a mut [usize],
    depth_touched: &'a mut Vec<usize>,
    unmatched_count: &'a mut usize,
}

#[derive(Clone, Copy)]
enum TerminalDepthKind {
    In,
    Out,
}

struct SearchContext<'a, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch> {
    prepared_query: &'a PreparedGraph<QueryNodeId>,
    prepared_target: &'a PreparedGraph<TargetNodeId>,
    mode: Vf2Mode,
    node_match: &'a NodeMatch,
    edge_match: &'a EdgeMatch,
    final_match: &'a FinalMatch,
}

struct TerminalTransitionContext<'a, GraphNodeId, OtherNodeId> {
    graph: &'a PreparedGraph<GraphNodeId>,
    core: &'a [Option<OtherNodeId>],
    other_depth: &'a [usize],
    future_counts: &'a mut [FutureNeighborCounts],
    depth_kind: TerminalDepthKind,
}

#[inline]
fn adjust_count(
    count: &mut usize,
    delta: ContributionDelta,
    label: &str,
    node_index: usize,
    neighbor_index: usize,
) {
    match delta {
        ContributionDelta::Add => *count += 1,
        ContributionDelta::Remove => {
            debug_assert!(
                *count != 0,
                "removing {label} contribution of node {node_index} via neighbor {neighbor_index}"
            );
            *count -= 1;
        }
    }
}

#[inline]
fn adjust_mapped_node_neighbor_contribution<GraphNodeId>(
    graph: &PreparedGraph<GraphNodeId>,
    mapped_neighbor_counts: &mut [MappedNeighborCounts],
    node_index: usize,
    delta: ContributionDelta,
) where
    GraphNodeId: Copy + AsPrimitive<usize>,
{
    for &neighbor in &graph.predecessors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index {
            continue;
        }

        adjust_count(
            &mut mapped_neighbor_counts[neighbor_index].successors,
            delta,
            "mapped_successors",
            node_index,
            neighbor_index,
        );
    }

    for &neighbor in &graph.successors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index {
            continue;
        }

        adjust_count(
            &mut mapped_neighbor_counts[neighbor_index].predecessors,
            delta,
            "mapped_predecessors",
            node_index,
            neighbor_index,
        );
    }
}

#[inline]
fn adjust_mapped_query_neighbor_lists<QueryNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    mapped_predecessors: &mut [Vec<QueryNodeId>],
    mapped_successors: &mut [Vec<QueryNodeId>],
    query_node: QueryNodeId,
    delta: ContributionDelta,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
{
    let query_node_index = query_node.as_();

    for &neighbor in &query.predecessors[query_node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == query_node_index {
            continue;
        }

        let successors = &mut mapped_successors[neighbor_index];
        match delta {
            ContributionDelta::Add => successors.push(query_node),
            ContributionDelta::Remove => {
                let popped = successors.pop();
                debug_assert!(matches!(popped, Some(node) if node.as_() == query_node_index));
            }
        }
    }

    for &neighbor in &query.successors[query_node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == query_node_index {
            continue;
        }

        let predecessors = &mut mapped_predecessors[neighbor_index];
        match delta {
            ContributionDelta::Add => predecessors.push(query_node),
            ContributionDelta::Remove => {
                let popped = predecessors.pop();
                debug_assert!(matches!(popped, Some(node) if node.as_() == query_node_index));
            }
        }
    }
}

#[inline]
fn adjust_unmapped_node_future_contribution<GraphNodeId, OtherNodeId>(
    graph: &PreparedGraph<GraphNodeId>,
    core: &[Option<OtherNodeId>],
    in_depth: &[usize],
    out_depth: &[usize],
    future_counts: &mut [FutureNeighborCounts],
    node_index: usize,
    delta: ContributionDelta,
) where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    if core[node_index].is_some() {
        return;
    }

    let is_in = in_depth[node_index] != 0;
    let is_out = out_depth[node_index] != 0;
    let is_new = !is_in && !is_out;

    for &neighbor in &graph.predecessors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index {
            continue;
        }

        let counts = &mut future_counts[neighbor_index];
        adjust_count(
            &mut counts.successor_total,
            delta,
            "successor_total",
            node_index,
            neighbor_index,
        );
        if is_in {
            adjust_count(
                &mut counts.successor_in,
                delta,
                "successor_in",
                node_index,
                neighbor_index,
            );
        }
        if is_out {
            adjust_count(
                &mut counts.successor_out,
                delta,
                "successor_out",
                node_index,
                neighbor_index,
            );
        }
        if is_new {
            adjust_count(
                &mut counts.successor_new,
                delta,
                "successor_new",
                node_index,
                neighbor_index,
            );
        }
    }

    for &neighbor in &graph.successors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index {
            continue;
        }

        let counts = &mut future_counts[neighbor_index];
        adjust_count(
            &mut counts.predecessor_total,
            delta,
            "predecessor_total",
            node_index,
            neighbor_index,
        );
        if is_in {
            adjust_count(
                &mut counts.predecessor_in,
                delta,
                "predecessor_in",
                node_index,
                neighbor_index,
            );
        }
        if is_out {
            adjust_count(
                &mut counts.predecessor_out,
                delta,
                "predecessor_out",
                node_index,
                neighbor_index,
            );
        }
        if is_new {
            adjust_count(
                &mut counts.predecessor_new,
                delta,
                "predecessor_new",
                node_index,
                neighbor_index,
            );
        }
    }
}

impl<GraphNodeId, OtherNodeId> TerminalTransitionContext<'_, GraphNodeId, OtherNodeId>
where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    #[inline]
    fn adjust(&mut self, depth: &[usize], node_index: usize, delta: ContributionDelta) {
        match self.depth_kind {
            TerminalDepthKind::In => {
                adjust_unmapped_node_future_contribution(
                    self.graph,
                    self.core,
                    depth,
                    self.other_depth,
                    self.future_counts,
                    node_index,
                    delta,
                );
            }
            TerminalDepthKind::Out => {
                adjust_unmapped_node_future_contribution(
                    self.graph,
                    self.core,
                    self.other_depth,
                    depth,
                    self.future_counts,
                    node_index,
                    delta,
                );
            }
        }
    }
}

#[inline]
fn select_best_unmapped_query_node_from_frontier<QueryNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    frontier: &[QueryNodeId],
) -> Option<QueryNodeId>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
{
    let mut best = None;
    let mut best_degree = 0usize;

    for query_node in frontier.iter().copied() {
        let degree = query.out_degree(query_node);
        match best {
            None => {
                best = Some(query_node);
                best_degree = degree;
            }
            Some(current) => {
                if degree > best_degree
                    || (degree == best_degree && query_node.as_() < current.as_())
                {
                    best = Some(query_node);
                    best_degree = degree;
                }
            }
        }
    }

    best
}

#[inline]
fn next_query_node<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    _mode: Vf2Mode,
    state: &Vf2State<QueryNodeId, TargetNodeId>,
) -> Option<(QueryNodeId, CandidateClass)>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    if state.has_unmapped_query_out() && state.has_unmapped_target_out() {
        return select_best_unmapped_query_node_from_frontier(query, &state.query_out_frontier)
            .map(|query_node| (query_node, CandidateClass::Out));
    }

    if state.has_unmapped_query_in() && state.has_unmapped_target_in() {
        return select_best_unmapped_query_node_from_frontier(query, &state.query_in_frontier)
            .map(|query_node| (query_node, CandidateClass::In));
    }

    let next_query_node = query.selection_order.get(state.query_unmatched_head).copied();
    debug_assert!(match next_query_node {
        Some(query_node) => !state.is_query_mapped(query_node),
        None => true,
    });
    next_query_node.map(|query_node| (query_node, CandidateClass::Unmatched))
}

#[inline]
fn frontier_requirements_are_satisfiable<QueryNodeId, TargetNodeId>(
    state: &Vf2State<QueryNodeId, TargetNodeId>,
) -> bool
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    (!state.has_unmapped_query_out() || state.has_unmapped_target_out())
        && (!state.has_unmapped_query_in() || state.has_unmapped_target_in())
}

#[inline]
fn candidate_targets_into<QueryNodeId, TargetNodeId, NodeMatch>(
    query: &PreparedGraph<QueryNodeId>,
    target: &PreparedGraph<TargetNodeId>,
    state: &Vf2State<QueryNodeId, TargetNodeId>,
    candidate_class: CandidateClass,
    query_node: QueryNodeId,
    node_match: &NodeMatch,
    candidates: &mut Vec<TargetNodeId>,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
{
    candidates.clear();

    let query_degree = query.out_degree(query_node);
    let query_in_degree = query.in_degree(query_node);
    let query_has_loop = query.has_self_loop(query_node);
    let mut push_matching_targets = |targets: &[TargetNodeId]| {
        for target_node in targets.iter().copied() {
            if target.out_degree(target_node) < query_degree
                || target.in_degree(target_node) < query_in_degree
                || (query_has_loop && !target.has_self_loop(target_node))
                || !node_match.matches(query_node, target_node)
            {
                continue;
            }

            candidates.push(target_node);
        }
    };

    match candidate_class {
        CandidateClass::Out => push_matching_targets(&state.target_out_frontier),
        CandidateClass::In => push_matching_targets(&state.target_in_frontier),
        CandidateClass::Unmatched => {
            for target_node in
                state.target_unmapped_nodes[..state.target_unmapped_len].iter().copied()
            {
                if target.out_degree(target_node) < query_degree
                    || target.in_degree(target_node) < query_in_degree
                    || (query_has_loop && !target.has_self_loop(target_node))
                    || !node_match.matches(query_node, target_node)
                {
                    continue;
                }

                candidates.push(target_node);
            }
        }
    }
}

#[inline]
fn mark_terminal_neighbors<GraphNodeId, OtherNodeId>(
    neighbors: &[GraphNodeId],
    mut context: TerminalTransitionContext<'_, GraphNodeId, OtherNodeId>,
    side: TerminalSideMut<'_, GraphNodeId>,
    depth_value: usize,
) where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    let TerminalSideMut { frontier, frontier_positions, depth, depth_touched, unmatched_count } =
        side;

    for &neighbor in neighbors {
        let neighbor_index = neighbor.as_();
        if context.core[neighbor_index].is_some() || depth[neighbor_index] != 0 {
            continue;
        }
        context.adjust(depth, neighbor_index, ContributionDelta::Remove);
        depth[neighbor_index] = depth_value;
        depth_touched.push(neighbor_index);
        context.adjust(depth, neighbor_index, ContributionDelta::Add);
        add_frontier_node(frontier, frontier_positions, neighbor);
        *unmatched_count += 1;
    }
}

fn clear_depth_since_snapshot<GraphNodeId, OtherNodeId>(
    mut context: TerminalTransitionContext<'_, GraphNodeId, OtherNodeId>,
    side: TerminalSideMut<'_, GraphNodeId>,
    touched_len: usize,
    excluded_node_index: Option<usize>,
) where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    let TerminalSideMut { frontier, frontier_positions, depth, depth_touched, unmatched_count } =
        side;

    while depth_touched.len() > touched_len {
        let node_index = depth_touched.pop().unwrap();
        let should_remove_old_future_contribution =
            context.core[node_index].is_none() && excluded_node_index != Some(node_index);
        if should_remove_old_future_contribution {
            context.adjust(depth, node_index, ContributionDelta::Remove);
        }
        if context.core[node_index].is_none() {
            remove_frontier_node_at_index(frontier, frontier_positions, node_index);
            if excluded_node_index != Some(node_index) {
                debug_assert!(*unmatched_count != 0);
                *unmatched_count -= 1;
            }
        }
        depth[node_index] = 0;
        if should_remove_old_future_contribution {
            context.adjust(depth, node_index, ContributionDelta::Add);
        }
    }
}

#[cfg(test)]
fn future_neighbor_counts<GraphNodeId, OtherNodeId>(
    graph: &PreparedGraph<GraphNodeId>,
    core: &[Option<OtherNodeId>],
    in_depth: &[usize],
    out_depth: &[usize],
    node: GraphNodeId,
) -> FutureNeighborCounts
where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    let mut counts = FutureNeighborCounts::default();
    let node_index = node.as_();

    for &neighbor in &graph.predecessors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index || core[neighbor_index].is_some() {
            continue;
        }

        counts.predecessor_total += 1;
        let is_in = in_depth[neighbor_index] != 0;
        let is_out = out_depth[neighbor_index] != 0;
        let is_new = !is_in && !is_out;

        if is_in {
            counts.predecessor_in += 1;
        }
        if is_out {
            counts.predecessor_out += 1;
        }
        if is_new {
            counts.predecessor_new += 1;
        }
    }

    for &neighbor in &graph.successors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index == node_index || core[neighbor_index].is_some() {
            continue;
        }

        counts.successor_total += 1;
        let is_in = in_depth[neighbor_index] != 0;
        let is_out = out_depth[neighbor_index] != 0;
        let is_new = !is_in && !is_out;

        if is_in {
            counts.successor_in += 1;
        }
        if is_out {
            counts.successor_out += 1;
        }
        if is_new {
            counts.successor_new += 1;
        }
    }

    counts
}

#[cfg(test)]
fn mapped_neighbor_counts<GraphNodeId, OtherNodeId>(
    graph: &PreparedGraph<GraphNodeId>,
    core: &[Option<OtherNodeId>],
    node: GraphNodeId,
) -> MappedNeighborCounts
where
    GraphNodeId: Copy + AsPrimitive<usize>,
    OtherNodeId: Copy + AsPrimitive<usize>,
{
    let mut counts = MappedNeighborCounts::default();
    let node_index = node.as_();

    for &neighbor in &graph.predecessors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index != node_index && core[neighbor_index].is_some() {
            counts.predecessors += 1;
        }
    }

    for &neighbor in &graph.successors[node_index] {
        let neighbor_index = neighbor.as_();
        if neighbor_index != node_index && core[neighbor_index].is_some() {
            counts.successors += 1;
        }
    }

    counts
}

#[cfg(test)]
fn mapped_query_neighbors<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    core: &[Option<TargetNodeId>],
    query_node: QueryNodeId,
) -> (Vec<QueryNodeId>, Vec<QueryNodeId>)
where
    QueryNodeId: Copy + AsPrimitive<usize> + Ord,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    let query_node_index = query_node.as_();
    let mut predecessors = query.predecessors[query_node_index]
        .iter()
        .copied()
        .filter(|&neighbor| neighbor.as_() != query_node_index && core[neighbor.as_()].is_some())
        .collect::<Vec<_>>();
    let mut successors = query.successors[query_node_index]
        .iter()
        .copied()
        .filter(|&neighbor| neighbor.as_() != query_node_index && core[neighbor.as_()].is_some())
        .collect::<Vec<_>>();
    predecessors.sort_unstable();
    successors.sort_unstable();
    (predecessors, successors)
}

#[inline]
fn future_neighbor_counts_are_compatible<QueryNodeId, TargetNodeId>(
    mode: Vf2Mode,
    state: &Vf2State<QueryNodeId, TargetNodeId>,
    query_counts: FutureNeighborCounts,
    target_node: TargetNodeId,
) -> bool
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    let target_counts = state.target_future_counts[target_node.as_()];

    query_counts.matches(target_counts, mode)
}

fn is_structurally_feasible<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>(
    context: &SearchContext<'_, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>,
    state: &Vf2State<QueryNodeId, TargetNodeId>,
    query_future_counts: Option<FutureNeighborCounts>,
    query_node: QueryNodeId,
    target_node: TargetNodeId,
) -> bool
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
{
    let query_mapped_neighbor_counts = state.query_mapped_neighbor_counts[query_node.as_()];
    let target_mapped_neighbor_counts = state.target_mapped_neighbor_counts[target_node.as_()];
    let query_mapped_predecessors = &state.query_mapped_predecessors[query_node.as_()];
    let query_mapped_successors = &state.query_mapped_successors[query_node.as_()];
    match context.mode {
        Vf2Mode::Isomorphism | Vf2Mode::InducedSubgraphIsomorphism => {
            if query_mapped_neighbor_counts != target_mapped_neighbor_counts {
                return false;
            }
        }
        Vf2Mode::SubgraphIsomorphism | Vf2Mode::Monomorphism => {
            if query_mapped_neighbor_counts.predecessors
                > target_mapped_neighbor_counts.predecessors
                || query_mapped_neighbor_counts.successors
                    > target_mapped_neighbor_counts.successors
            {
                return false;
            }
        }
    }

    let query_self_loop = context.prepared_query.has_self_loop(query_node);
    let target_self_loop = context.prepared_target.has_self_loop(target_node);
    if query_self_loop {
        if !target_self_loop
            || !context.edge_match.matches(query_node, query_node, target_node, target_node)
        {
            return false;
        }
    } else if matches!(context.mode, Vf2Mode::Isomorphism | Vf2Mode::InducedSubgraphIsomorphism)
        && target_self_loop
    {
        return false;
    }

    if !query_mapped_successors.is_empty() {
        for &mapped_query_node in query_mapped_successors {
            let mapped_target_node = state.query_core[mapped_query_node.as_()]
                .expect("mapped successor cache referenced an unmapped query node");
            if !context.prepared_target.has_successor(target_node, mapped_target_node)
                || !context.edge_match.matches(
                    query_node,
                    mapped_query_node,
                    target_node,
                    mapped_target_node,
                )
            {
                return false;
            }
        }
    }

    if !query_mapped_predecessors.is_empty() {
        for &mapped_query_node in query_mapped_predecessors {
            let mapped_target_node = state.query_core[mapped_query_node.as_()]
                .expect("mapped predecessor cache referenced an unmapped query node");
            if !context.prepared_target.has_successor(mapped_target_node, target_node)
                || !context.edge_match.matches(
                    mapped_query_node,
                    query_node,
                    mapped_target_node,
                    target_node,
                )
            {
                return false;
            }
        }
    }

    if let Some(query_future_counts) = query_future_counts {
        if !future_neighbor_counts_are_compatible(
            context.mode,
            state,
            query_future_counts,
            target_node,
        ) {
            return false;
        }
    }

    true
}

fn mark_mapped_node_depth<QueryNodeId, TargetNodeId>(
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    query_node: QueryNodeId,
    target_node: TargetNodeId,
    depth: usize,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    let query_node_index = query_node.as_();
    let target_node_index = target_node.as_();

    if state.query_in_depth[query_node_index] == 0 {
        state.query_in_depth[query_node_index] = depth;
        state.query_in_depth_touched.push(query_node_index);
    }
    if state.query_out_depth[query_node_index] == 0 {
        state.query_out_depth[query_node_index] = depth;
        state.query_out_depth_touched.push(query_node_index);
    }
    if state.target_in_depth[target_node_index] == 0 {
        state.target_in_depth[target_node_index] = depth;
        state.target_in_depth_touched.push(target_node_index);
    }
    if state.target_out_depth[target_node_index] == 0 {
        state.target_out_depth[target_node_index] = depth;
        state.target_out_depth_touched.push(target_node_index);
    }
}

fn mark_new_terminal_neighbors<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    target: &PreparedGraph<TargetNodeId>,
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    query_node_index: usize,
    target_node_index: usize,
    depth: usize,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    mark_terminal_neighbors(
        &query.predecessors[query_node_index],
        TerminalTransitionContext {
            graph: query,
            core: &state.query_core,
            other_depth: &state.query_out_depth,
            future_counts: &mut state.query_future_counts,
            depth_kind: TerminalDepthKind::In,
        },
        TerminalSideMut {
            frontier: &mut state.query_in_frontier,
            frontier_positions: &mut state.query_in_frontier_positions,
            depth: &mut state.query_in_depth,
            depth_touched: &mut state.query_in_depth_touched,
            unmatched_count: &mut state.query_unmapped_in_count,
        },
        depth,
    );
    mark_terminal_neighbors(
        &query.successors[query_node_index],
        TerminalTransitionContext {
            graph: query,
            core: &state.query_core,
            other_depth: &state.query_in_depth,
            future_counts: &mut state.query_future_counts,
            depth_kind: TerminalDepthKind::Out,
        },
        TerminalSideMut {
            frontier: &mut state.query_out_frontier,
            frontier_positions: &mut state.query_out_frontier_positions,
            depth: &mut state.query_out_depth,
            depth_touched: &mut state.query_out_depth_touched,
            unmatched_count: &mut state.query_unmapped_out_count,
        },
        depth,
    );
    mark_terminal_neighbors(
        &target.predecessors[target_node_index],
        TerminalTransitionContext {
            graph: target,
            core: &state.target_core,
            other_depth: &state.target_out_depth,
            future_counts: &mut state.target_future_counts,
            depth_kind: TerminalDepthKind::In,
        },
        TerminalSideMut {
            frontier: &mut state.target_in_frontier,
            frontier_positions: &mut state.target_in_frontier_positions,
            depth: &mut state.target_in_depth,
            depth_touched: &mut state.target_in_depth_touched,
            unmatched_count: &mut state.target_unmapped_in_count,
        },
        depth,
    );
    mark_terminal_neighbors(
        &target.successors[target_node_index],
        TerminalTransitionContext {
            graph: target,
            core: &state.target_core,
            other_depth: &state.target_in_depth,
            future_counts: &mut state.target_future_counts,
            depth_kind: TerminalDepthKind::Out,
        },
        TerminalSideMut {
            frontier: &mut state.target_out_frontier,
            frontier_positions: &mut state.target_out_frontier_positions,
            depth: &mut state.target_out_depth,
            depth_touched: &mut state.target_out_depth_touched,
            unmatched_count: &mut state.target_unmapped_out_count,
        },
        depth,
    );
}

fn extend_state<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    target: &PreparedGraph<TargetNodeId>,
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    query_node: QueryNodeId,
    target_node: TargetNodeId,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    let depth = state.mapping.len() + 1;
    let query_node_index = query_node.as_();
    let target_node_index = target_node.as_();
    state.frontier_snapshots.push(FrontierSnapshot {
        query_unmatched_head: state.query_unmatched_head,
        query_in_depth_touched_len: state.query_in_depth_touched.len(),
        query_out_depth_touched_len: state.query_out_depth_touched.len(),
        target_in_depth_touched_len: state.target_in_depth_touched.len(),
        target_out_depth_touched_len: state.target_out_depth_touched.len(),
    });
    state.begin_query_mapping(query, query_node);
    state.begin_target_mapping(target, target_node);

    state.query_core[query_node.as_()] = Some(target_node);
    state.target_core[target_node.as_()] = Some(query_node);
    state.mapping.push((query_node, target_node));
    adjust_mapped_node_neighbor_contribution(
        query,
        &mut state.query_mapped_neighbor_counts,
        query_node_index,
        ContributionDelta::Add,
    );
    adjust_mapped_query_neighbor_lists(
        query,
        &mut state.query_mapped_predecessors,
        &mut state.query_mapped_successors,
        query_node,
        ContributionDelta::Add,
    );
    adjust_mapped_node_neighbor_contribution(
        target,
        &mut state.target_mapped_neighbor_counts,
        target_node_index,
        ContributionDelta::Add,
    );
    state.advance_query_unmatched_head(&query.selection_order);
    mark_mapped_node_depth(state, query_node, target_node, depth);
    mark_new_terminal_neighbors(query, target, state, query_node_index, target_node_index, depth);
}

fn clear_depth_since_snapshot_for_mapped_pair<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    target: &PreparedGraph<TargetNodeId>,
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    snapshot: FrontierSnapshot,
    query_node_index: usize,
    target_node_index: usize,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    clear_depth_since_snapshot(
        TerminalTransitionContext {
            graph: query,
            core: &state.query_core,
            other_depth: &state.query_out_depth,
            future_counts: &mut state.query_future_counts,
            depth_kind: TerminalDepthKind::In,
        },
        TerminalSideMut {
            frontier: &mut state.query_in_frontier,
            frontier_positions: &mut state.query_in_frontier_positions,
            depth: &mut state.query_in_depth,
            depth_touched: &mut state.query_in_depth_touched,
            unmatched_count: &mut state.query_unmapped_in_count,
        },
        snapshot.query_in_depth_touched_len,
        Some(query_node_index),
    );
    clear_depth_since_snapshot(
        TerminalTransitionContext {
            graph: query,
            core: &state.query_core,
            other_depth: &state.query_in_depth,
            future_counts: &mut state.query_future_counts,
            depth_kind: TerminalDepthKind::Out,
        },
        TerminalSideMut {
            frontier: &mut state.query_out_frontier,
            frontier_positions: &mut state.query_out_frontier_positions,
            depth: &mut state.query_out_depth,
            depth_touched: &mut state.query_out_depth_touched,
            unmatched_count: &mut state.query_unmapped_out_count,
        },
        snapshot.query_out_depth_touched_len,
        Some(query_node_index),
    );
    clear_depth_since_snapshot(
        TerminalTransitionContext {
            graph: target,
            core: &state.target_core,
            other_depth: &state.target_out_depth,
            future_counts: &mut state.target_future_counts,
            depth_kind: TerminalDepthKind::In,
        },
        TerminalSideMut {
            frontier: &mut state.target_in_frontier,
            frontier_positions: &mut state.target_in_frontier_positions,
            depth: &mut state.target_in_depth,
            depth_touched: &mut state.target_in_depth_touched,
            unmatched_count: &mut state.target_unmapped_in_count,
        },
        snapshot.target_in_depth_touched_len,
        Some(target_node_index),
    );
    clear_depth_since_snapshot(
        TerminalTransitionContext {
            graph: target,
            core: &state.target_core,
            other_depth: &state.target_in_depth,
            future_counts: &mut state.target_future_counts,
            depth_kind: TerminalDepthKind::Out,
        },
        TerminalSideMut {
            frontier: &mut state.target_out_frontier,
            frontier_positions: &mut state.target_out_frontier_positions,
            depth: &mut state.target_out_depth,
            depth_touched: &mut state.target_out_depth_touched,
            unmatched_count: &mut state.target_unmapped_out_count,
        },
        snapshot.target_out_depth_touched_len,
        Some(target_node_index),
    );
}

fn restore_state<QueryNodeId, TargetNodeId>(
    query: &PreparedGraph<QueryNodeId>,
    target: &PreparedGraph<TargetNodeId>,
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    query_node: QueryNodeId,
    target_node: TargetNodeId,
) where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
{
    let query_node_index = query_node.as_();
    let target_node_index = target_node.as_();

    let _ = state.mapping.pop();
    state.query_core[query_node_index] = None;
    state.target_core[target_node_index] = None;
    adjust_mapped_node_neighbor_contribution(
        query,
        &mut state.query_mapped_neighbor_counts,
        query_node_index,
        ContributionDelta::Remove,
    );
    adjust_mapped_query_neighbor_lists(
        query,
        &mut state.query_mapped_predecessors,
        &mut state.query_mapped_successors,
        query_node,
        ContributionDelta::Remove,
    );
    adjust_mapped_node_neighbor_contribution(
        target,
        &mut state.target_mapped_neighbor_counts,
        target_node_index,
        ContributionDelta::Remove,
    );
    let snapshot = state
        .frontier_snapshots
        .pop()
        .expect("restore_state called without a matching frontier snapshot");
    clear_depth_since_snapshot_for_mapped_pair(
        query,
        target,
        state,
        snapshot,
        query_node_index,
        target_node_index,
    );
    state.query_unmatched_head = snapshot.query_unmatched_head;
    state.finish_query_unmapping(query_node);
    state.finish_target_unmapping(target_node);
    adjust_unmapped_node_future_contribution(
        query,
        &state.query_core,
        &state.query_in_depth,
        &state.query_out_depth,
        &mut state.query_future_counts,
        query_node_index,
        ContributionDelta::Add,
    );
    adjust_unmapped_node_future_contribution(
        target,
        &state.target_core,
        &state.target_in_depth,
        &state.target_out_depth,
        &mut state.target_future_counts,
        target_node_index,
        ContributionDelta::Add,
    );
}

fn search<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch, Visit>(
    context: &SearchContext<'_, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>,
    state: &mut Vf2State<QueryNodeId, TargetNodeId>,
    visitor: &mut Visit,
) -> SearchControl
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
    FinalMatch: Vf2FinalMatcher<QueryNodeId, TargetNodeId>,
    Visit: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
{
    if state.mapping.len() == context.prepared_query.node_ids.len() {
        if !context.final_match.matches(&state.mapping) {
            return SearchControl::Continue;
        }
        return if visitor(&state.mapping) { SearchControl::Continue } else { SearchControl::Stop };
    }

    if !frontier_requirements_are_satisfiable(state) {
        return SearchControl::Continue;
    }

    let Some((query_node, candidate_class)) =
        next_query_node(context.prepared_query, context.mode, state)
    else {
        return SearchControl::Continue;
    };

    let depth = state.mapping.len();
    let mut candidates = core::mem::take(&mut state.candidate_scratch[depth]);
    let query_future_counts = Some(state.query_future_counts[query_node.as_()]);
    candidate_targets_into(
        context.prepared_query,
        context.prepared_target,
        state,
        candidate_class,
        query_node,
        context.node_match,
        &mut candidates,
    );

    for target_node in candidates.iter().copied() {
        if !is_structurally_feasible(context, state, query_future_counts, query_node, target_node) {
            continue;
        }

        extend_state(
            context.prepared_query,
            context.prepared_target,
            state,
            query_node,
            target_node,
        );

        let control = search(context, state, visitor);

        restore_state(
            context.prepared_query,
            context.prepared_target,
            state,
            query_node,
            target_node,
        );

        if matches!(control, SearchControl::Stop) {
            candidates.clear();
            state.candidate_scratch[depth] = candidates;
            return SearchControl::Stop;
        }
    }

    candidates.clear();
    state.candidate_scratch[depth] = candidates;
    SearchControl::Continue
}

fn run_prepared_search<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch, F>(
    query: &PreparedVf2Graph<QueryNodeId>,
    target: &PreparedVf2Graph<TargetNodeId>,
    mode: Vf2Mode,
    node_match: &NodeMatch,
    edge_match: &EdgeMatch,
    final_match: &FinalMatch,
    mut visitor: F,
) -> SearchControl
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
    FinalMatch: Vf2FinalMatcher<QueryNodeId, TargetNodeId>,
    F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
{
    let prepared_query = query.prepared();
    let prepared_target = target.prepared();
    let context = SearchContext {
        prepared_query,
        prepared_target,
        mode,
        node_match,
        edge_match,
        final_match,
    };
    let mut state = Vf2State::new(prepared_query, prepared_target);

    search(&context, &mut state, &mut visitor)
}

fn search_prepared_with_mapping<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch, F>(
    query: &PreparedVf2Graph<QueryNodeId>,
    target: &PreparedVf2Graph<TargetNodeId>,
    mode: Vf2Mode,
    node_match: &NodeMatch,
    edge_match: &EdgeMatch,
    final_match: &FinalMatch,
    visitor: F,
) -> SearchControl
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
    FinalMatch: Vf2FinalMatcher<QueryNodeId, TargetNodeId>,
    F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
{
    if !passes_global_precheck_counts(
        query.number_of_nodes(),
        target.number_of_nodes(),
        query.number_of_edges(),
        target.number_of_edges(),
        query.number_of_self_loops(),
        target.number_of_self_loops(),
        mode,
    ) {
        return SearchControl::Continue;
    }

    run_prepared_search(query, target, mode, node_match, edge_match, final_match, visitor)
}

impl<Query, Target, NodeMatch, EdgeMatch, FinalMatch>
    Vf2SearchMethods<Query::NodeId, Target::NodeId>
    for Vf2Builder<'_, Query, Target, NodeMatch, EdgeMatch, FinalMatch>
where
    Query: MonoplexMonopartiteGraph,
    Target: MonoplexMonopartiteGraph,
    Query::NodeId: Copy,
    Target::NodeId: Copy,
    NodeMatch: Vf2NodeMatcher<Query::NodeId, Target::NodeId>,
    EdgeMatch: Vf2EdgeMatcher<Query::NodeId, Target::NodeId>,
    FinalMatch: Vf2FinalMatcher<Query::NodeId, Target::NodeId>,
{
    fn search_with_mapping<F>(&self, mut visitor: F) -> SearchControl
    where
        F: FnMut(&[(Query::NodeId, Target::NodeId)]) -> bool,
    {
        if !passes_global_precheck(self.query, self.target, self.mode) {
            return SearchControl::Continue;
        }

        let prepared_query = PreparedVf2Graph::new(self.query);
        let prepared_target = PreparedVf2Graph::new(self.target);

        run_prepared_search(
            &prepared_query,
            &prepared_target,
            self.mode,
            &self.node_match,
            &self.edge_match,
            &self.final_match,
            &mut visitor,
        )
    }
}

impl<Query, Target, NodeMatch, EdgeMatch, FinalMatch>
    Vf2Builder<'_, Query, Target, NodeMatch, EdgeMatch, FinalMatch>
where
    Query: MonoplexMonopartiteGraph,
    Target: MonoplexMonopartiteGraph,
    Query::NodeId: Copy,
    Target::NodeId: Copy,
    NodeMatch: Vf2NodeMatcher<Query::NodeId, Target::NodeId>,
    EdgeMatch: Vf2EdgeMatcher<Query::NodeId, Target::NodeId>,
    FinalMatch: Vf2FinalMatcher<Query::NodeId, Target::NodeId>,
{
    /// Returns whether at least one VF2 match exists.
    #[must_use]
    pub fn has_match(&self) -> bool {
        <Self as Vf2SearchMethods<Query::NodeId, Target::NodeId>>::has_match_impl(self)
    }

    /// Returns the first VF2 match found by the search.
    ///
    /// Match order is implementation-defined and may change as the search
    /// heuristics evolve.
    #[must_use]
    pub fn first_match(&self) -> Option<Vf2Match<Query::NodeId, Target::NodeId>> {
        <Self as Vf2SearchMethods<Query::NodeId, Target::NodeId>>::first_match_impl(self)
    }

    /// Streams borrowed mappings to the provided visitor.
    ///
    /// This avoids allocating a new [`Vf2Match`] for every embedding.
    ///
    /// Returns `true` when the search exhausts all matches and `false` when the
    /// visitor stops the search early.
    pub fn for_each_mapping<F>(&self, visitor: F) -> bool
    where
        F: FnMut(&[(Query::NodeId, Target::NodeId)]) -> bool,
    {
        <Self as Vf2SearchMethods<Query::NodeId, Target::NodeId>>::for_each_mapping_impl(
            self, visitor,
        )
    }

    /// Streams matches to the provided visitor.
    ///
    /// Match order is implementation-defined and may change as the search
    /// heuristics evolve.
    ///
    /// Returns `true` when the search exhausts all matches and `false` when the
    /// visitor stops the search early.
    pub fn for_each_match<F>(&self, mut visitor: F) -> bool
    where
        F: FnMut(&Vf2Match<Query::NodeId, Target::NodeId>) -> bool,
    {
        <Self as Vf2SearchMethods<Query::NodeId, Target::NodeId>>::for_each_match_impl(
            self,
            &mut visitor,
        )
    }
}

impl<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
    Vf2SearchMethods<QueryNodeId, TargetNodeId>
    for PreparedVf2Builder<'_, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
    FinalMatch: Vf2FinalMatcher<QueryNodeId, TargetNodeId>,
{
    fn search_with_mapping<F>(&self, mut visitor: F) -> SearchControl
    where
        F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
    {
        search_prepared_with_mapping(
            self.query,
            self.target,
            self.mode,
            &self.node_match,
            &self.edge_match,
            &self.final_match,
            &mut visitor,
        )
    }
}

impl<QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
    PreparedVf2Builder<'_, QueryNodeId, TargetNodeId, NodeMatch, EdgeMatch, FinalMatch>
where
    QueryNodeId: Copy + AsPrimitive<usize>,
    TargetNodeId: Copy + AsPrimitive<usize>,
    NodeMatch: Vf2NodeMatcher<QueryNodeId, TargetNodeId>,
    EdgeMatch: Vf2EdgeMatcher<QueryNodeId, TargetNodeId>,
    FinalMatch: Vf2FinalMatcher<QueryNodeId, TargetNodeId>,
{
    /// Returns whether at least one VF2 match exists.
    #[must_use]
    pub fn has_match(&self) -> bool {
        <Self as Vf2SearchMethods<QueryNodeId, TargetNodeId>>::has_match_impl(self)
    }

    /// Returns the first VF2 match found by the search.
    ///
    /// Match order is implementation-defined and may change as the search
    /// heuristics evolve.
    #[must_use]
    pub fn first_match(&self) -> Option<Vf2Match<QueryNodeId, TargetNodeId>> {
        <Self as Vf2SearchMethods<QueryNodeId, TargetNodeId>>::first_match_impl(self)
    }

    /// Streams borrowed mappings to the provided visitor.
    ///
    /// This avoids allocating a new [`Vf2Match`] for every embedding.
    ///
    /// Returns `true` when the search exhausts all matches and `false` when the
    /// visitor stops the search early.
    pub fn for_each_mapping<F>(&self, visitor: F) -> bool
    where
        F: FnMut(&[(QueryNodeId, TargetNodeId)]) -> bool,
    {
        <Self as Vf2SearchMethods<QueryNodeId, TargetNodeId>>::for_each_mapping_impl(self, visitor)
    }

    /// Streams matches to the provided visitor.
    ///
    /// Match order is implementation-defined and may change as the search
    /// heuristics evolve.
    ///
    /// Returns `true` when the search exhausts all matches and `false` when the
    /// visitor stops the search early.
    pub fn for_each_match<F>(&self, mut visitor: F) -> bool
    where
        F: FnMut(&Vf2Match<QueryNodeId, TargetNodeId>) -> bool,
    {
        <Self as Vf2SearchMethods<QueryNodeId, TargetNodeId>>::for_each_match_impl(
            self,
            &mut visitor,
        )
    }
}

/// Trait providing the VF2 entry point for monoplex monopartite graphs.
pub trait Vf2: MonoplexMonopartiteGraph {
    /// Returns a VF2 builder using `self` as the query graph.
    ///
    /// For one-off searches this is the simplest entry point. If the same graph
    /// pair will be searched repeatedly, prefer [`PrepareVf2::prepare_vf2`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// fn build_graph(node_count: usize, mut edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(node_count)
    ///         .symbols((0..node_count).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     edges.sort_unstable();
    ///     let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape(node_count)
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, edges))
    /// }
    ///
    /// let query = build_graph(3, vec![(0, 1), (1, 2)]);
    /// let target = build_graph(3, vec![(0, 1), (1, 2), (0, 2)]);
    /// let colors = [0_u8, 1, 0];
    ///
    /// let has_match = query
    ///     .vf2(&target)
    ///     .with_mode(Vf2Mode::SubgraphIsomorphism)
    ///     .with_node_match(|query_node, target_node| colors[query_node] == colors[target_node])
    ///     .has_match();
    ///
    /// assert!(has_match);
    /// ```
    #[inline]
    #[must_use]
    fn vf2<'g, Target>(&'g self, target: &'g Target) -> Vf2Builder<'g, Self, Target>
    where
        Self: Sized,
        Target: MonoplexMonopartiteGraph,
    {
        Vf2Builder::new(self, target)
    }
}

impl<G> Vf2 for G where G: MonoplexMonopartiteGraph {}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::{
        AcceptAllEdgeMatcher, AcceptAllNodeMatcher, CandidateClass, PreparedGraph, SearchContext,
        Vf2Match, Vf2Mode, Vf2State, candidate_targets_into, extend_state,
        frontier_requirements_are_satisfiable, future_neighbor_counts_are_compatible,
        is_structurally_feasible, next_query_node, restore_state,
    };
    use crate::{
        impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
        prelude::*,
        traits::{EdgesBuilder, VocabularyBuilder},
    };

    fn build_undigraph(node_count: usize, mut edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
        let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(node_count)
            .symbols((0..node_count).enumerate())
            .build()
            .unwrap();
        edges.sort_unstable();
        let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape(node_count)
            .edges(edges.into_iter())
            .build()
            .unwrap();
        UndiGraph::from((nodes, edges))
    }

    fn build_digraph(node_count: usize, mut edges: Vec<(usize, usize)>) -> DiGraph<usize> {
        let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(node_count)
            .symbols((0..node_count).enumerate())
            .build()
            .unwrap();
        edges.sort_unstable();
        let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape(node_count)
            .edges(edges.into_iter())
            .build()
            .unwrap();
        DiGraph::from((nodes, edges))
    }

    #[test]
    fn test_vf2_match_round_trips_pairs() {
        let mapping = Vf2Match::new(vec![(0_u8, 1_u8), (2_u8, 3_u8)]);
        assert_eq!(mapping.pairs(), &[(0, 1), (2, 3)]);
        assert_eq!(mapping.clone().into_pairs(), vec![(0, 1), (2, 3)]);
        assert_eq!(mapping.len(), 2);
        assert!(!mapping.is_empty());
    }

    #[test]
    fn test_vf2_mode_variants_are_distinct() {
        assert_ne!(Vf2Mode::Isomorphism, Vf2Mode::InducedSubgraphIsomorphism);
        assert_ne!(Vf2Mode::InducedSubgraphIsomorphism, Vf2Mode::SubgraphIsomorphism);
        assert_ne!(Vf2Mode::Isomorphism, Vf2Mode::Monomorphism);
        assert_ne!(Vf2Mode::InducedSubgraphIsomorphism, Vf2Mode::Monomorphism);
        assert_ne!(Vf2Mode::SubgraphIsomorphism, Vf2Mode::Monomorphism);
    }

    #[test]
    fn test_prepared_graph_selection_order_prefers_higher_degree_then_lower_node_id() {
        let graph = build_undigraph(5, vec![(0, 1), (2, 3), (2, 4)]);
        let prepared = PreparedGraph::new(&graph);

        assert_eq!(prepared.selection_order, vec![2, 0, 1, 3, 4]);
    }

    #[test]
    fn test_extend_state_marks_nodes_adjacent_to_mapping_as_terminal() {
        let graph = build_undigraph(5, vec![(0, 1), (2, 3), (2, 4)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);

        assert!(state.is_query_terminal(1));
        assert!(!state.is_query_terminal(2));
        assert!(!state.is_query_terminal(3));
        assert!(!state.is_query_terminal(4));
    }

    #[test]
    fn test_extend_state_splits_directed_in_and_out_frontiers() {
        let graph = build_digraph(3, vec![(2, 0), (0, 1)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);

        assert!(state.is_query_in_terminal(2));
        assert!(!state.is_query_out_terminal(2));
        assert!(state.is_query_out_terminal(1));
        assert!(!state.is_query_in_terminal(1));
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.target_unmapped_in_count, 1);
        assert_eq!(state.target_unmapped_out_count, 1);
    }

    #[test]
    fn test_extend_state_tracks_unmatched_directed_frontier_counts() {
        let graph = build_digraph(3, vec![(2, 0), (0, 1)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);

        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.target_unmapped_in_count, 1);
        assert_eq!(state.target_unmapped_out_count, 1);
    }

    #[test]
    fn test_next_query_node_prefers_terminal_nodes_when_available() {
        let graph = build_undigraph(5, vec![(0, 1), (2, 3), (2, 4)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        let next = next_query_node(&prepared, Vf2Mode::Isomorphism, &state);

        assert_eq!(next, Some((1, super::CandidateClass::Out)));
    }

    #[test]
    fn test_next_query_node_unmatched_fallback_preserves_degree_order() {
        let graph = build_digraph(4, vec![(0, 1), (0, 2), (2, 3)]);
        let prepared = PreparedGraph::new(&graph);
        let state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((0, CandidateClass::Unmatched))
        );
    }

    #[test]
    fn test_next_query_node_falls_back_to_unmatched_when_only_target_has_frontier() {
        let graph = build_digraph(2, vec![(0, 1)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        state.target_out_depth[1] = 1;
        state.target_unmapped_out_count = 1;

        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((0, super::CandidateClass::Unmatched))
        );
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::InducedSubgraphIsomorphism, &state),
            Some((0, super::CandidateClass::Unmatched))
        );
    }

    #[test]
    fn test_frontier_requirements_reject_when_query_has_terminal_nodes_but_target_does_not() {
        let query = build_digraph(3, vec![(0, 1)]);
        let target = build_digraph(3, vec![]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);

        extend_state(&prepared_query, &prepared_target, &mut state, 0, 0);

        assert!(state.has_unmapped_query_out());
        assert!(!state.has_unmapped_target_out());
        assert!(!frontier_requirements_are_satisfiable(&state));
    }

    #[test]
    fn test_next_query_node_uses_in_frontier_when_out_frontier_absent() {
        let graph = build_digraph(3, vec![(1, 0), (2, 1)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);

        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::In))
        );
    }

    #[test]
    fn test_next_query_node_falls_back_to_unmatched_after_frontier_component_finishes() {
        let graph = build_digraph(3, vec![(0, 1)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        extend_state(&prepared, &prepared, &mut state, 1, 1);

        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((2, CandidateClass::Unmatched))
        );

        restore_state(&prepared, &prepared, &mut state, 1, 1);

        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::Out))
        );
    }

    #[test]
    fn test_query_unmatched_head_restores_after_frontier_mapping_backtracks() {
        let graph = build_digraph(4, vec![(0, 1), (0, 2)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        assert_eq!(state.query_unmatched_head, 1);

        extend_state(&prepared, &prepared, &mut state, 1, 1);
        assert_eq!(state.query_unmatched_head, 2);

        restore_state(&prepared, &prepared, &mut state, 1, 1);
        assert_eq!(state.query_unmatched_head, 1);
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::Out))
        );

        restore_state(&prepared, &prepared, &mut state, 0, 0);
        assert_eq!(state.query_unmatched_head, 0);
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((0, CandidateClass::Unmatched))
        );
    }

    #[test]
    fn test_query_unmatched_head_ignores_nonhead_frontier_mappings() {
        let graph = build_undigraph(5, vec![(0, 1), (2, 3), (2, 4)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        assert_eq!(prepared.selection_order[state.query_unmatched_head], 2);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        assert_eq!(prepared.selection_order[state.query_unmatched_head], 2);
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::Out))
        );

        extend_state(&prepared, &prepared, &mut state, 1, 1);
        assert_eq!(prepared.selection_order[state.query_unmatched_head], 2);
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((2, CandidateClass::Unmatched))
        );

        restore_state(&prepared, &prepared, &mut state, 1, 1);
        assert_eq!(prepared.selection_order[state.query_unmatched_head], 2);
        restore_state(&prepared, &prepared, &mut state, 0, 0);
        assert_eq!(prepared.selection_order[state.query_unmatched_head], 2);
    }

    #[test]
    fn test_future_neighbor_counts_prune_insufficient_targets() {
        let query = build_digraph(3, vec![(0, 1), (0, 2)]);
        let target = build_digraph(2, vec![(0, 1)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);
        let query_counts = super::future_neighbor_counts(
            &prepared_query,
            &state.query_core,
            &state.query_in_depth,
            &state.query_out_depth,
            0,
        );

        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::InducedSubgraphIsomorphism,
            &state,
            query_counts,
            0,
        ));
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::SubgraphIsomorphism,
            &state,
            query_counts,
            0,
        ));
    }

    #[test]
    fn test_future_neighbor_counts_distinguish_inbound_and_outbound_neighbors() {
        let query = build_digraph(2, vec![(0, 1), (1, 0)]);
        let matching_target = build_digraph(2, vec![(0, 1), (1, 0)]);
        let missing_inbound_target = build_digraph(2, vec![(0, 1)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_matching_target = PreparedGraph::new(&matching_target);
        let prepared_missing_inbound_target = PreparedGraph::new(&missing_inbound_target);
        let matching_state: Vf2State<usize, usize> =
            Vf2State::new(&prepared_query, &prepared_matching_target);
        let missing_state: Vf2State<usize, usize> =
            Vf2State::new(&prepared_query, &prepared_missing_inbound_target);
        let query_counts = super::future_neighbor_counts(
            &prepared_query,
            &matching_state.query_core,
            &matching_state.query_in_depth,
            &matching_state.query_out_depth,
            0,
        );

        assert!(future_neighbor_counts_are_compatible(
            Vf2Mode::InducedSubgraphIsomorphism,
            &matching_state,
            query_counts,
            0,
        ));
        assert!(future_neighbor_counts_are_compatible(
            Vf2Mode::SubgraphIsomorphism,
            &matching_state,
            query_counts,
            0,
        ));
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::InducedSubgraphIsomorphism,
            &missing_state,
            query_counts,
            0,
        ));
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::SubgraphIsomorphism,
            &missing_state,
            query_counts,
            0,
        ));
    }

    #[test]
    fn test_future_neighbor_counts_subgraph_can_use_terminal_capacity_for_query_new_neighbors() {
        let query = build_digraph(4, vec![(0, 1), (2, 3)]);
        let target = build_digraph(4, vec![(0, 1), (0, 3), (2, 3)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);

        extend_state(&prepared_query, &prepared_target, &mut state, 0, 0);

        let query_counts = super::future_neighbor_counts(
            &prepared_query,
            &state.query_core,
            &state.query_in_depth,
            &state.query_out_depth,
            2,
        );

        assert_eq!(query_counts.successor_new, 1);
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::InducedSubgraphIsomorphism,
            &state,
            query_counts,
            2,
        ));
        assert!(future_neighbor_counts_are_compatible(
            Vf2Mode::SubgraphIsomorphism,
            &state,
            query_counts,
            2,
        ));
    }

    #[test]
    fn test_future_neighbor_counts_subgraph_still_requires_terminal_capacity_for_query_terminal_neighbors()
     {
        let query = build_digraph(4, vec![(0, 2), (1, 3), (2, 3)]);
        let target = build_digraph(5, vec![(0, 2), (1, 3), (2, 4)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);

        extend_state(&prepared_query, &prepared_target, &mut state, 0, 0);
        extend_state(&prepared_query, &prepared_target, &mut state, 1, 1);

        let query_counts = super::future_neighbor_counts(
            &prepared_query,
            &state.query_core,
            &state.query_in_depth,
            &state.query_out_depth,
            2,
        );

        assert_eq!(query_counts.successor_out, 1);
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::InducedSubgraphIsomorphism,
            &state,
            query_counts,
            2,
        ));
        assert!(!future_neighbor_counts_are_compatible(
            Vf2Mode::SubgraphIsomorphism,
            &state,
            query_counts,
            2,
        ));
    }

    #[test]
    fn test_future_neighbor_count_caches_match_rescan_across_extend_and_restore() {
        let graph = build_digraph(4, vec![(2, 0), (0, 1), (1, 3)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        let assert_cached_counts_match_rescan = |state: &Vf2State<usize, usize>, stage: &str| {
            for &node in &prepared.node_ids {
                let (expected_predecessors, expected_successors) =
                    super::mapped_query_neighbors(&prepared, &state.query_core, node);
                let mut cached_predecessors = state.query_mapped_predecessors[node].clone();
                let mut cached_successors = state.query_mapped_successors[node].clone();
                cached_predecessors.sort_unstable();
                cached_successors.sort_unstable();
                assert_eq!(
                    cached_predecessors, expected_predecessors,
                    "query mapped-predecessor list mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    cached_successors, expected_successors,
                    "query mapped-successor list mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.query_mapped_neighbor_counts[node],
                    super::mapped_neighbor_counts(&prepared, &state.query_core, node),
                    "query mapped-neighbor cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.target_mapped_neighbor_counts[node],
                    super::mapped_neighbor_counts(&prepared, &state.target_core, node),
                    "target mapped-neighbor cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.query_future_counts[node],
                    super::future_neighbor_counts(
                        &prepared,
                        &state.query_core,
                        &state.query_in_depth,
                        &state.query_out_depth,
                        node,
                    ),
                    "query cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.target_future_counts[node],
                    super::future_neighbor_counts(
                        &prepared,
                        &state.target_core,
                        &state.target_in_depth,
                        &state.target_out_depth,
                        node,
                    ),
                    "target cache mismatch at stage {stage} for node {node}",
                );
            }
        };

        assert_cached_counts_match_rescan(&state, "initial");

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        assert_cached_counts_match_rescan(&state, "after mapping 0");

        extend_state(&prepared, &prepared, &mut state, 1, 1);
        assert_cached_counts_match_rescan(&state, "after mapping 1");

        restore_state(&prepared, &prepared, &mut state, 1, 1);
        assert_cached_counts_match_rescan(&state, "after restoring 1");

        restore_state(&prepared, &prepared, &mut state, 0, 0);
        assert_cached_counts_match_rescan(&state, "after restoring 0");
    }

    #[test]
    fn test_future_neighbor_count_caches_match_rescan_for_dual_terminal_cycle() {
        let graph = build_digraph(3, vec![(0, 1), (1, 2), (2, 0)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        let assert_cached_counts_match_rescan = |state: &Vf2State<usize, usize>, stage: &str| {
            for &node in &prepared.node_ids {
                let (expected_predecessors, expected_successors) =
                    super::mapped_query_neighbors(&prepared, &state.query_core, node);
                let mut cached_predecessors = state.query_mapped_predecessors[node].clone();
                let mut cached_successors = state.query_mapped_successors[node].clone();
                cached_predecessors.sort_unstable();
                cached_successors.sort_unstable();
                assert_eq!(
                    cached_predecessors, expected_predecessors,
                    "query mapped-predecessor list mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    cached_successors, expected_successors,
                    "query mapped-successor list mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.query_mapped_neighbor_counts[node],
                    super::mapped_neighbor_counts(&prepared, &state.query_core, node),
                    "query mapped-neighbor cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.target_mapped_neighbor_counts[node],
                    super::mapped_neighbor_counts(&prepared, &state.target_core, node),
                    "target mapped-neighbor cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.query_future_counts[node],
                    super::future_neighbor_counts(
                        &prepared,
                        &state.query_core,
                        &state.query_in_depth,
                        &state.query_out_depth,
                        node,
                    ),
                    "query cache mismatch at stage {stage} for node {node}",
                );
                assert_eq!(
                    state.target_future_counts[node],
                    super::future_neighbor_counts(
                        &prepared,
                        &state.target_core,
                        &state.target_in_depth,
                        &state.target_out_depth,
                        node,
                    ),
                    "target cache mismatch at stage {stage} for node {node}",
                );
            }
        };

        assert_cached_counts_match_rescan(&state, "initial");

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        assert_cached_counts_match_rescan(&state, "after mapping 0");

        extend_state(&prepared, &prepared, &mut state, 1, 1);
        assert_cached_counts_match_rescan(&state, "after mapping 1");

        restore_state(&prepared, &prepared, &mut state, 1, 1);
        assert_cached_counts_match_rescan(&state, "after restoring 1");

        restore_state(&prepared, &prepared, &mut state, 0, 0);
        assert_cached_counts_match_rescan(&state, "after restoring 0");
    }

    #[test]
    fn test_restore_state_returns_dual_terminal_cycle_to_initial_state() {
        let graph = build_digraph(3, vec![(0, 1), (1, 2), (2, 0)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        extend_state(&prepared, &prepared, &mut state, 1, 1);
        restore_state(&prepared, &prepared, &mut state, 1, 1);
        restore_state(&prepared, &prepared, &mut state, 0, 0);

        assert!(state.mapping.is_empty());
        assert_eq!(state.query_in_frontier, Vec::<usize>::new());
        assert_eq!(state.query_out_frontier, Vec::<usize>::new());
        assert_eq!(state.target_in_frontier, Vec::<usize>::new());
        assert_eq!(state.target_out_frontier, Vec::<usize>::new());
        assert_eq!(state.query_unmapped_in_count, 0);
        assert_eq!(state.query_unmapped_out_count, 0);
        assert_eq!(state.target_unmapped_in_count, 0);
        assert_eq!(state.target_unmapped_out_count, 0);
        assert_eq!(state.query_in_depth, vec![0, 0, 0]);
        assert_eq!(state.query_out_depth, vec![0, 0, 0]);
        assert_eq!(state.target_in_depth, vec![0, 0, 0]);
        assert_eq!(state.target_out_depth, vec![0, 0, 0]);
    }

    #[test]
    fn test_subgraph_isolated_nodes_keep_unmatched_candidate_fallback() {
        let query = build_undigraph(2, Vec::new());
        let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);
        let context = SearchContext {
            prepared_query: &prepared_query,
            prepared_target: &prepared_target,
            mode: Vf2Mode::SubgraphIsomorphism,
            node_match: &AcceptAllNodeMatcher,
            edge_match: &AcceptAllEdgeMatcher,
            final_match: &super::AcceptAllFinalMatcher,
        };

        extend_state(&prepared_query, &prepared_target, &mut state, 0, 0);

        assert_eq!(
            next_query_node(&prepared_query, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::Unmatched))
        );
        let mut candidates = vec![99];
        candidate_targets_into(
            &prepared_query,
            &prepared_target,
            &state,
            CandidateClass::Unmatched,
            1,
            &AcceptAllNodeMatcher,
            &mut candidates,
        );
        candidates.sort_unstable();
        assert_eq!(candidates, vec![1, 2]);
        assert!(is_structurally_feasible(&context, &state, None, 1, 1));
        assert!(is_structurally_feasible(&context, &state, None, 1, 2));
    }

    #[test]
    fn test_candidate_targets_into_clears_reused_buffer() {
        let query = build_digraph(2, vec![(0, 1)]);
        let target = build_digraph(5, vec![(0, 1), (2, 3)]);
        let prepared_query = PreparedGraph::new(&query);
        let prepared_target = PreparedGraph::new(&target);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared_query, &prepared_target);
        let mut candidates = vec![99, 100];

        candidate_targets_into(
            &prepared_query,
            &prepared_target,
            &state,
            CandidateClass::Unmatched,
            0,
            &AcceptAllNodeMatcher,
            &mut candidates,
        );
        assert_eq!(candidates, vec![0, 2]);

        extend_state(&prepared_query, &prepared_target, &mut state, 0, 0);
        candidate_targets_into(
            &prepared_query,
            &prepared_target,
            &state,
            CandidateClass::Out,
            1,
            &AcceptAllNodeMatcher,
            &mut candidates,
        );
        assert_eq!(candidates, vec![1]);
    }

    #[test]
    fn test_restore_state_keeps_older_directed_frontier_marks_after_backtrack() {
        let graph = build_digraph(4, vec![(2, 0), (0, 1), (1, 3)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.query_in_frontier, vec![2]);
        assert_eq!(state.query_out_frontier, vec![1]);
        extend_state(&prepared, &prepared, &mut state, 1, 1);
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.query_in_frontier, vec![2]);
        assert_eq!(state.query_out_frontier, vec![3]);
        restore_state(&prepared, &prepared, &mut state, 1, 1);

        assert!(state.is_query_in_terminal(2));
        assert!(state.is_query_out_terminal(1));
        assert!(!state.is_query_in_terminal(3));
        assert!(!state.is_query_out_terminal(3));
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.target_unmapped_in_count, 1);
        assert_eq!(state.target_unmapped_out_count, 1);
        assert_eq!(state.query_in_frontier, vec![2]);
        assert_eq!(state.query_out_frontier, vec![1]);
        assert_eq!(state.target_in_frontier, vec![2]);
        assert_eq!(state.target_out_frontier, vec![1]);
        assert_eq!(
            next_query_node(&prepared, Vf2Mode::SubgraphIsomorphism, &state),
            Some((1, CandidateClass::Out))
        );
    }

    #[test]
    fn test_restore_state_restores_unmatched_directed_frontier_counts() {
        let graph = build_digraph(4, vec![(2, 0), (0, 1), (1, 3)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        extend_state(&prepared, &prepared, &mut state, 1, 1);
        restore_state(&prepared, &prepared, &mut state, 1, 1);

        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.target_unmapped_in_count, 1);
        assert_eq!(state.target_unmapped_out_count, 1);
    }

    #[test]
    fn test_restore_state_clears_depth_marks_of_unmatched_isolated_mapping() {
        let graph = build_digraph(3, Vec::new());
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        extend_state(&prepared, &prepared, &mut state, 1, 1);
        restore_state(&prepared, &prepared, &mut state, 1, 1);

        assert!(!state.is_query_terminal(1));
        assert_eq!(state.query_in_frontier, Vec::<usize>::new());
        assert_eq!(state.query_out_frontier, Vec::<usize>::new());
        assert_eq!(state.query_unmapped_in_count, 0);
        assert_eq!(state.query_unmapped_out_count, 0);
    }

    #[test]
    fn test_restore_state_keeps_dual_frontier_membership_on_each_side_independent() {
        let graph = build_digraph(3, vec![(2, 0), (0, 1), (1, 2)]);
        let prepared = PreparedGraph::new(&graph);
        let mut state: Vf2State<usize, usize> = Vf2State::new(&prepared, &prepared);

        extend_state(&prepared, &prepared, &mut state, 0, 0);
        extend_state(&prepared, &prepared, &mut state, 1, 1);

        assert!(state.is_query_in_terminal(2));
        assert!(state.is_query_out_terminal(2));
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);

        restore_state(&prepared, &prepared, &mut state, 1, 1);

        assert!(state.is_query_in_terminal(2));
        assert!(!state.is_query_out_terminal(2));
        assert_eq!(state.query_unmapped_in_count, 1);
        assert_eq!(state.query_unmapped_out_count, 1);
        assert_eq!(state.query_in_frontier, vec![2]);
        assert_eq!(state.query_out_frontier, vec![1]);
    }
}
