//! Submodule declaring planarity detection for simple undirected graphs.
//!
//! The current implementation builds a local simple-graph view, performs DFS
//! preprocessing, and then runs the crate's internal edge-addition embedding
//! engine. The public contract is intentionally limited to simple undirected
//! graphs, so self-loops and parallel edges are rejected.
#![cfg_attr(test, allow(clippy::pedantic))]

use num_traits::AsPrimitive;

use crate::traits::{MonopartiteGraph, UndirectedMonopartiteMonoplexGraph};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for planarity detection.
pub enum PlanarityError {
    /// The graph contains self-loops, which are unsupported by the intended
    /// simple undirected implementation.
    #[error(
        "The planarity algorithm currently supports only simple undirected graphs and does not accept self-loops."
    )]
    SelfLoopsUnsupported,
    /// Parallel edges are unsupported by the intended public contract.
    #[error(
        "The planarity algorithm currently supports only simple undirected graphs and does not accept parallel edges."
    )]
    ParallelEdgesUnsupported,
    /// The graph implementation exposed an endpoint outside the node range.
    #[error(
        "The graph exposed edge endpoint {endpoint}, which is out of range for node_count={node_count}."
    )]
    InvalidEdgeEndpoint {
        /// The offending endpoint value exposed by the graph.
        endpoint: usize,
        /// The graph node count used to validate endpoints.
        node_count: usize,
    },
}

impl From<PlanarityError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: PlanarityError) -> Self {
        Self::PlanarityError(error)
    }
}

impl<G: MonopartiteGraph> From<PlanarityError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: PlanarityError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

/// Trait providing planarity detection for simple undirected graphs.
pub trait PlanarityDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns whether the graph is planar.
    ///
    /// The implementation uses the crate's internal simple-undirected
    /// edge-addition embedding engine.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph violates the simple-undirected contract,
    /// such as by containing self-loops, parallel edges, or malformed edge
    /// endpoints from a custom graph implementation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, PlanarityDetection, VocabularyBuilder},
    /// };
    ///
    /// fn build_undigraph(node_count: usize, edges: &[(usize, usize)]) -> UndiGraph<usize> {
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(node_count)
    ///         .symbols((0..node_count).enumerate())
    ///         .build()
    ///         .unwrap();
    ///
    ///     let mut normalized_edges = edges.to_vec();
    ///     for (left, right) in &mut normalized_edges {
    ///         if *left > *right {
    ///             core::mem::swap(left, right);
    ///         }
    ///     }
    ///     normalized_edges.sort_unstable();
    ///
    ///     let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
    ///         .expected_number_of_edges(normalized_edges.len())
    ///         .expected_shape(node_count)
    ///         .edges(normalized_edges.into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    ///     UndiGraph::from((nodes, matrix))
    /// }
    ///
    /// let planar = build_undigraph(4, &[(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]);
    /// let nonplanar = build_undigraph(
    ///     6,
    ///     &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)],
    /// );
    ///
    /// assert!(planar.is_planar()?);
    /// assert!(!nonplanar.is_planar()?);
    /// # Ok::<(), geometric_traits::errors::MonopartiteError<UndiGraph<usize>>>(())
    /// ```
    #[inline]
    fn is_planar(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
    {
        Ok(is_planar_simple_undirected_graph(self)?)
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> PlanarityDetection for G {}

pub(crate) fn is_planar_simple_undirected_graph<G>(graph: &G) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let simple_graph = preprocessing::LocalSimpleGraph::try_from_undirected_graph(graph)?;
    let preprocessing = simple_graph.preprocess();
    Ok(run_planarity_engine(&preprocessing))
}

pub(crate) fn is_outerplanar_simple_undirected_graph<G>(graph: &G) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let simple_graph = preprocessing::LocalSimpleGraph::try_from_undirected_graph(graph)?;
    let preprocessing = simple_graph.preprocess();
    Ok(run_outerplanarity_engine(&preprocessing))
}

pub(crate) fn has_k23_homeomorph_simple_undirected_graph<G>(
    graph: &G,
) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let simple_graph = preprocessing::LocalSimpleGraph::try_from_undirected_graph(graph)?;
    let preprocessing = simple_graph.preprocess();
    Ok(run_k23_homeomorph_engine(&preprocessing))
}

#[allow(dead_code)]
pub(crate) fn has_k4_homeomorph_simple_undirected_graph<G>(
    graph: &G,
) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let simple_graph = preprocessing::LocalSimpleGraph::try_from_undirected_graph(graph)?;
    let preprocessing = simple_graph.preprocess();
    Ok(run_k4_homeomorph_engine(&preprocessing))
}

pub(crate) fn run_planarity_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
    matches!(
        run_embedding_engine(preprocessing, EmbeddingRunMode::Planarity),
        EmbeddingRunOutcome::Embedded(_)
    )
}

fn run_outerplanarity_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
    matches!(
        run_embedding_engine(preprocessing, EmbeddingRunMode::Outerplanarity),
        EmbeddingRunOutcome::Embedded(embedding)
            if embedding.all_primary_vertices_on_external_face(preprocessing)
    )
}

pub(crate) fn run_k23_homeomorph_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
    if preprocessing.vertices.len() < 5 || preprocessing.arcs.len() / 2 < 6 {
        return false;
    }

    matches!(
        run_embedding_engine(preprocessing, EmbeddingRunMode::K23Search),
        EmbeddingRunOutcome::K23Found
    )
}

pub(crate) fn run_k33_homeomorph_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
    if preprocessing.vertices.len() < 6 || preprocessing.arcs.len() / 2 < 9 {
        return false;
    }

    matches!(
        run_embedding_engine(preprocessing, EmbeddingRunMode::K33Search),
        EmbeddingRunOutcome::K33Found
    )
}

#[allow(dead_code)]
pub(crate) fn run_k4_homeomorph_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
    if preprocessing.vertices.len() < 4 || preprocessing.arcs.len() / 2 < 6 {
        return false;
    }

    matches!(
        run_embedding_engine(preprocessing, EmbeddingRunMode::K4Search),
        EmbeddingRunOutcome::K4Found
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingRunMode {
    Planarity,
    Outerplanarity,
    K23Search,
    K33Search,
    K4Search,
}

enum EmbeddingRunOutcome {
    Embedded(embedding::EmbeddingState),
    K23Found,
    K33Found,
    K4Found,
    Failed,
}

fn run_embedding_engine(
    preprocessing: &preprocessing::DfsPreprocessing,
    mode: EmbeddingRunMode,
) -> EmbeddingRunOutcome {
    let mut embedding = embedding::EmbeddingState::from_preprocessing(preprocessing);

    for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
        for slot in &mut embedding.slots {
            slot.pertinent_edge = None;
        }

        let original_vertex = match embedding.slots[current_primary_slot].kind {
            embedding::EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
            embedding::EmbeddingSlotKind::RootCopy { .. } => {
                panic!("primary slots must occupy the first N embedding positions")
            }
        };

        for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
            embedding.walk_up(current_primary_slot, forward_arc);
        }
        embedding.slots[current_primary_slot].pertinent_roots.clear();

        let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
        for child_index in 0..child_count {
            let child_primary_slot =
                embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
            if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                continue;
            }
            let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
            else {
                continue;
            };
            if embedding.slots[root_copy_slot].first_arc.is_none() {
                continue;
            }

            match embedding.walk_down_child(
                preprocessing,
                current_primary_slot,
                root_copy_slot,
                mode,
            ) {
                Ok(embedding::WalkDownChildOutcome::Completed) => {}
                Ok(embedding::WalkDownChildOutcome::K23Found) => {
                    return EmbeddingRunOutcome::K23Found;
                }
                Ok(embedding::WalkDownChildOutcome::K33Found) => {
                    return EmbeddingRunOutcome::K33Found;
                }
                Ok(embedding::WalkDownChildOutcome::K4Found) => {
                    return EmbeddingRunOutcome::K4Found;
                }
                Err(_) => {
                    return EmbeddingRunOutcome::Failed;
                }
            }
        }
    }

    EmbeddingRunOutcome::Embedded(embedding)
}

#[cfg(test)]
fn child_subtree_has_blocking_forward_arc_head(
    preprocessing: &preprocessing::DfsPreprocessing,
    embedding: &embedding::EmbeddingState,
    original_vertex: usize,
    child_primary_slot: usize,
    next_child_primary_slot: Option<usize>,
) -> bool {
    child_subtree_forward_arc_head(
        preprocessing,
        embedding,
        original_vertex,
        child_primary_slot,
        next_child_primary_slot,
    )
    .is_some()
}

#[cfg_attr(not(test), allow(dead_code))]
fn child_subtree_forward_arc_head(
    preprocessing: &preprocessing::DfsPreprocessing,
    embedding: &embedding::EmbeddingState,
    original_vertex: usize,
    child_primary_slot: usize,
    next_child_primary_slot: Option<usize>,
) -> Option<usize> {
    let current_primary_slot = embedding.primary_slot_by_original_vertex[original_vertex];
    let forward_arc = embedding.peek_forward_arc_head(preprocessing, current_primary_slot)?;
    let descendant_primary_slot = embedding.arcs[forward_arc].target_slot;
    (descendant_primary_slot >= child_primary_slot
        && next_child_primary_slot.is_none_or(|next_child| descendant_primary_slot < next_child))
    .then_some(forward_arc)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod preprocessing {
    use alloc::{collections::BTreeSet, vec, vec::Vec};

    use num_traits::AsPrimitive;

    use super::{PlanarityError, UndirectedMonopartiteMonoplexGraph};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum DfsArcType {
        Unclassified,
        Child,
        Parent,
        Back,
        Forward,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct DfsArcRecord {
        pub(crate) source: usize,
        pub(crate) target: usize,
        pub(crate) twin: usize,
        pub(crate) kind: DfsArcType,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct DfsVertexState {
        pub(crate) parent: Option<usize>,
        pub(crate) parent_arc: Option<usize>,
        pub(crate) dfi: usize,
        pub(crate) least_ancestor: usize,
        pub(crate) lowpoint: usize,
        pub(crate) sorted_dfs_children: Vec<usize>,
        pub(crate) sorted_forward_arcs: Vec<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct DfsPreprocessing {
        pub(crate) adjacency_arcs: Vec<Vec<usize>>,
        pub(crate) arcs: Vec<DfsArcRecord>,
        pub(crate) vertices: Vec<DfsVertexState>,
        pub(crate) vertex_by_dfi: Vec<usize>,
        pub(crate) dfs_roots: Vec<usize>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    pub(crate) enum LocalSimpleGraphError {
        #[error("self-loops are unsupported in the local simple-graph preprocessing builder")]
        SelfLoop,
        #[error("parallel edges are unsupported in the local simple-graph preprocessing builder")]
        ParallelEdge,
        #[error("edge endpoint {endpoint} is out of range for node_count={node_count}")]
        OutOfRange { endpoint: usize, node_count: usize },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct LocalSimpleGraph {
        adjacency_arcs: Vec<Vec<usize>>,
        arcs: Vec<DfsArcRecord>,
    }

    impl LocalSimpleGraph {
        #[inline]
        pub(crate) fn map_local_simple_graph_error(error: LocalSimpleGraphError) -> PlanarityError {
            match error {
                LocalSimpleGraphError::SelfLoop => PlanarityError::SelfLoopsUnsupported,
                LocalSimpleGraphError::ParallelEdge => PlanarityError::ParallelEdgesUnsupported,
                LocalSimpleGraphError::OutOfRange { endpoint, node_count } => {
                    PlanarityError::InvalidEdgeEndpoint { endpoint, node_count }
                }
            }
        }

        pub(crate) fn from_edges(
            node_count: usize,
            edges: &[[usize; 2]],
        ) -> Result<Self, LocalSimpleGraphError> {
            let mut seen = BTreeSet::new();
            let mut adjacency_arcs = vec![Vec::new(); node_count];
            let mut arcs = Vec::with_capacity(edges.len() * 2);

            for &[left, right] in edges {
                if left >= node_count {
                    return Err(LocalSimpleGraphError::OutOfRange { endpoint: left, node_count });
                }
                if right >= node_count {
                    return Err(LocalSimpleGraphError::OutOfRange { endpoint: right, node_count });
                }
                if left == right {
                    return Err(LocalSimpleGraphError::SelfLoop);
                }
                let normalized = if left <= right { [left, right] } else { [right, left] };
                if !seen.insert(normalized) {
                    return Err(LocalSimpleGraphError::ParallelEdge);
                }

                let left_arc = arcs.len();
                let right_arc = left_arc + 1;
                arcs.push(DfsArcRecord {
                    source: left,
                    target: right,
                    twin: right_arc,
                    kind: DfsArcType::Unclassified,
                });
                arcs.push(DfsArcRecord {
                    source: right,
                    target: left,
                    twin: left_arc,
                    kind: DfsArcType::Unclassified,
                });
                adjacency_arcs[left].push(left_arc);
                adjacency_arcs[right].push(right_arc);
            }

            for incident_arcs in &mut adjacency_arcs {
                incident_arcs.sort_unstable_by_key(|&arc_id| arcs[arc_id].target);
            }

            Ok(Self { adjacency_arcs, arcs })
        }

        pub(crate) fn try_from_undirected_graph<G>(graph: &G) -> Result<Self, PlanarityError>
        where
            G: UndirectedMonopartiteMonoplexGraph,
            G::NodeId: AsPrimitive<usize>,
        {
            if graph.has_self_loops() {
                return Err(PlanarityError::SelfLoopsUnsupported);
            }

            let mut edges = Vec::with_capacity(graph.number_of_edges().as_() / 2);
            for source in graph.node_ids() {
                let source_index = source.as_();
                for target in graph.neighbors(source) {
                    let target_index = target.as_();
                    if source_index < target_index {
                        edges.push([source_index, target_index]);
                    }
                }
            }

            Self::from_edges(graph.number_of_nodes().as_(), &edges)
                .map_err(Self::map_local_simple_graph_error)
        }

        pub(crate) fn preprocess(&self) -> DfsPreprocessing {
            let mut preprocessing = DfsPreprocessing {
                adjacency_arcs: self.adjacency_arcs.clone(),
                arcs: self.arcs.clone(),
                vertices: vec![
                    DfsVertexState {
                        parent: None,
                        parent_arc: None,
                        dfi: usize::MAX,
                        least_ancestor: usize::MAX,
                        lowpoint: usize::MAX,
                        sorted_dfs_children: Vec::new(),
                        sorted_forward_arcs: Vec::new(),
                    };
                    self.adjacency_arcs.len()
                ],
                vertex_by_dfi: Vec::with_capacity(self.adjacency_arcs.len()),
                dfs_roots: Vec::new(),
            };
            let mut visited = vec![false; self.adjacency_arcs.len()];
            let mut next_dfi = 0usize;

            for root in 0..self.adjacency_arcs.len() {
                if visited[root] {
                    continue;
                }
                preprocessing.dfs_roots.push(root);
                Self::dfs_visit(
                    root,
                    None,
                    None,
                    &self.adjacency_arcs,
                    &mut preprocessing,
                    &mut visited,
                    &mut next_dfi,
                );
            }

            preprocessing
        }

        fn dfs_visit(
            vertex: usize,
            parent: Option<usize>,
            parent_arc: Option<usize>,
            adjacency_arcs: &[Vec<usize>],
            preprocessing: &mut DfsPreprocessing,
            visited: &mut [bool],
            next_dfi: &mut usize,
        ) {
            visited[vertex] = true;
            let vertex_dfi = *next_dfi;
            *next_dfi += 1;

            preprocessing.vertices[vertex].parent = parent;
            preprocessing.vertices[vertex].parent_arc = parent_arc;
            preprocessing.vertices[vertex].dfi = vertex_dfi;
            preprocessing.vertices[vertex].least_ancestor = vertex_dfi;
            preprocessing.vertex_by_dfi.push(vertex);

            let mut child_lowpoints = Vec::new();
            for &arc_id in &adjacency_arcs[vertex] {
                let target = preprocessing.arcs[arc_id].target;
                match preprocessing.arcs[arc_id].kind {
                    DfsArcType::Child
                    | DfsArcType::Parent
                    | DfsArcType::Back
                    | DfsArcType::Forward => continue,
                    DfsArcType::Unclassified => {}
                }

                if !visited[target] {
                    preprocessing.arcs[arc_id].kind = DfsArcType::Child;
                    let twin = preprocessing.arcs[arc_id].twin;
                    preprocessing.arcs[twin].kind = DfsArcType::Parent;
                    preprocessing.vertices[vertex].sorted_dfs_children.push(target);
                    Self::dfs_visit(
                        target,
                        Some(vertex),
                        Some(twin),
                        adjacency_arcs,
                        preprocessing,
                        visited,
                        next_dfi,
                    );
                    child_lowpoints.push(preprocessing.vertices[target].lowpoint);
                } else if Some(target) != parent {
                    debug_assert!(
                        preprocessing.vertices[target].dfi < preprocessing.vertices[vertex].dfi
                    );
                    preprocessing.arcs[arc_id].kind = DfsArcType::Back;
                    let twin = preprocessing.arcs[arc_id].twin;
                    preprocessing.arcs[twin].kind = DfsArcType::Forward;
                    preprocessing.vertices[target].sorted_forward_arcs.push(twin);
                    preprocessing.vertices[vertex].least_ancestor = preprocessing.vertices[vertex]
                        .least_ancestor
                        .min(preprocessing.vertices[target].dfi);
                }
            }

            let mut lowpoint = preprocessing.vertices[vertex].least_ancestor;
            for child_lowpoint in child_lowpoints {
                lowpoint = lowpoint.min(child_lowpoint);
            }
            preprocessing.vertices[vertex].lowpoint = lowpoint;
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod embedding {
    use alloc::vec::Vec;
    use core::mem;

    use super::preprocessing::{DfsArcType, DfsPreprocessing};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum EmbeddingSlotKind {
        Primary { original_vertex: usize },
        RootCopy { parent_primary_slot: usize, dfs_child_primary_slot: usize },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct EmbeddingSlot {
        pub(crate) kind: EmbeddingSlotKind,
        pub(crate) first_arc: Option<usize>,
        pub(crate) last_arc: Option<usize>,
        pub(crate) ext_face: [Option<usize>; 2],
        pub(crate) visited: bool,
        pub(crate) visited_info: usize,
        pub(crate) pertinent_roots: Vec<usize>,
        pub(crate) future_pertinent_child: Option<usize>,
        pub(crate) pertinent_edge: Option<usize>,
        pub(crate) k33_merge_blocker: Option<usize>,
        pub(crate) k33_minor_e_reduced: bool,
        pub(crate) sorted_dfs_children: Vec<usize>,
        pub(crate) separated_dfs_children: Vec<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct EmbeddingArcRecord {
        pub(crate) original_arc: Option<usize>,
        pub(crate) source_slot: usize,
        pub(crate) target_slot: usize,
        pub(crate) twin: usize,
        pub(crate) next: Option<usize>,
        pub(crate) prev: Option<usize>,
        pub(crate) visited: bool,
        pub(crate) kind: DfsArcType,
        pub(crate) embedded: bool,
        pub(crate) inverted: bool,
        pub(crate) reduction_endpoint_arc: Option<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct EmbeddingState {
        pub(crate) slots: Vec<EmbeddingSlot>,
        pub(crate) arcs: Vec<EmbeddingArcRecord>,
        pub(crate) primary_slot_by_original_vertex: Vec<usize>,
        pub(crate) root_copy_by_primary_dfi: Vec<Option<usize>>,
        pub(crate) least_ancestor_by_primary_slot: Vec<usize>,
        pub(crate) lowpoint_by_primary_slot: Vec<usize>,
        pub(crate) forward_arc_head_index_by_primary_slot: Vec<Option<usize>>,
        pub(crate) handling_k4_blocked_bicomp: bool,
        pub(crate) k4_reblocked_same_root: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
    pub(crate) enum EmbeddingMutationError {
        #[error("walk-down trace did not end at a descendant suitable for back-edge embedding")]
        TraceDidNotReachDescendant,
        #[error("expected an embedded arc at slot {slot} side {side}")]
        MissingSlotArc { slot: usize, side: usize },
        #[error("expected slot {slot} to have a pertinent edge before back-edge insertion")]
        MissingPertinentEdge { slot: usize },
        #[error(
            "could not trace an external-face path from slot {start_slot} side {start_side} to slot {end_slot}"
        )]
        MissingExternalFacePath { start_slot: usize, start_side: usize, end_slot: usize },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct BlockedBicompContext {
        pub(crate) current_primary_slot: usize,
        pub(crate) walk_root_copy_slot: usize,
        pub(crate) walk_root_side: usize,
        pub(crate) cut_vertex_slot: usize,
        pub(crate) cut_vertex_entry_side: usize,
        pub(crate) blocked_root_copy_slot: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    struct NonOuterplanarityContext {
        current_primary_slot: usize,
        x_slot: usize,
        y_slot: usize,
        w_slot: usize,
        x_prev_link: usize,
        y_prev_link: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct NonplanarityContext {
        current_primary_slot: usize,
        root_copy_slot: usize,
        x_slot: usize,
        y_slot: usize,
        w_slot: usize,
        x_prev_link: usize,
        y_prev_link: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum WalkDownExecutionError {
        BlockedBicomp { context: BlockedBicompContext },
        InvalidK23Context,
        InvalidK4Context,
        InvalidK33Context,
        UnembeddedForwardArcInChildSubtree { forward_arc: usize },
        Mutation(EmbeddingMutationError),
    }

    impl core::fmt::Display for WalkDownExecutionError {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::BlockedBicomp { .. } => formatter.write_str(
                    "walk-down was blocked by a pertinent bicomp during outerplanarity traversal",
                ),
                Self::InvalidK23Context => formatter.write_str(
                    "K23 obstruction search failed to initialize non-outerplanarity context",
                ),
                Self::InvalidK4Context => formatter.write_str(
                    "K4 search failed to initialize blocked-bicomp context",
                ),
                Self::InvalidK33Context => formatter.write_str(
                    "K33 search failed to initialize nonplanarity context",
                ),
                Self::UnembeddedForwardArcInChildSubtree { .. } => formatter.write_str(
                    "walk-down finished with an unembedded forward arc remaining in the child subtree",
                ),
                Self::Mutation(error) => error.fmt(formatter),
            }
        }
    }

    impl From<EmbeddingMutationError> for WalkDownExecutionError {
        fn from(error: EmbeddingMutationError) -> Self {
            Self::Mutation(error)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct WalkDownFrame {
        pub(crate) cut_vertex_slot: usize,
        pub(crate) cut_vertex_entry_side: usize,
        pub(crate) root_copy_slot: usize,
        pub(crate) root_side: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum WalkDownOutcome {
        DescendantFound { slot: usize, entry_side: usize },
        StoppingVertex { slot: usize, entry_side: usize },
        BlockedBicomp { root_copy_slot: usize },
        CompletedToRoot,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct WalkDownTrace {
        pub(crate) root_copy_slot: usize,
        pub(crate) root_side: usize,
        pub(crate) visited_slots: Vec<usize>,
        pub(crate) frames: Vec<WalkDownFrame>,
        pub(crate) outcome: WalkDownOutcome,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum WalkDownChildOutcome {
        Completed,
        K23Found,
        K33Found,
        K4Found,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PertinentRootWalkAction {
        Descend { next_slot: usize, next_entry_side: usize, chosen_root_side: usize },
        ContinueWalkdown,
        Return(WalkDownChildOutcome),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ChildSubtreeAction {
        AdvanceAndReturn(WalkDownChildOutcome),
        Return(WalkDownChildOutcome),
        ErrorUnembeddedForwardArc,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum K33BicompSearchOutcome {
        MinorFound,
        ContinueMinorE { context: K33MinorEContext },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum K33ExtraTestOutcome {
        E1,
        E4Like,
        E5,
        E6,
        E7,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum K23BicompSearchOutcome {
        MinorA,
        MinorB,
        MinorE1OrE2,
        MinorE3OrE4,
        SeparableK4,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) enum K4BicompSearchOutcome {
        MinorFound,
        Continue,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum K4BlockedBicompOutcome {
        ContinueWalkdown,
        Completed,
        Found,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) enum K4MinorType {
        A,
        B,
        E,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    struct K4Context {
        current_primary_slot: usize,
        root_copy_slot: usize,
        x_slot: usize,
        y_slot: usize,
        w_slot: usize,
        x_prev_link: usize,
        y_prev_link: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub(crate) enum K33MinorType {
        A,
        B,
        C,
        D,
        E,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum K33ObstructionMark {
        Unmarked,
        HighRxw,
        LowRxw,
        HighRyw,
        LowRyw,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum K4ObstructionMark {
        Unmarked,
        HighRxw,
        LowRxw,
        HighRyw,
        LowRyw,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct K33MarkedPath {
        px_slot: usize,
        py_slot: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct K4MarkedPath {
        px_slot: usize,
        py_slot: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct HiddenArcPair {
        arc: usize,
        arc_prev: Option<usize>,
        arc_next: Option<usize>,
        twin_prev: Option<usize>,
        twin_next: Option<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct K33MinorEContext {
        pub(crate) current_primary_slot: usize,
        pub(crate) root_copy_slot: usize,
        pub(crate) x_slot: usize,
        pub(crate) y_slot: usize,
        pub(crate) w_slot: usize,
        pub(crate) z_slot: usize,
        pub(crate) px_slot: usize,
        pub(crate) py_slot: usize,
        pub(crate) ux: usize,
        pub(crate) uy: usize,
        pub(crate) uz: usize,
        pub(crate) obstruction_marks: Vec<K33ObstructionMark>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum K33MinorSearchOutcome {
        Minor(K33MinorType),
        MinorE(K33MinorEContext),
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum K33ContextInitFailure {
        NoActiveX,
        NoActiveY,
        NoDescendantBicompRoot,
        NoPertinentBetweenActiveSides,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum K33ZToRPathFailure {
        Structural,
        MarkedSlot { slot: usize, mark: K33ObstructionMark },
    }

    impl EmbeddingState {
        #[allow(clippy::too_many_lines)]
        pub(crate) fn from_preprocessing(preprocessing: &DfsPreprocessing) -> Self {
            let number_of_vertices = preprocessing.vertices.len();
            let number_of_root_copies = number_of_vertices - preprocessing.dfs_roots.len();

            let mut slots = Vec::with_capacity(number_of_vertices + number_of_root_copies);
            let mut primary_slot_by_original_vertex = vec![usize::MAX; number_of_vertices];

            for (dfi, &original_vertex) in preprocessing.vertex_by_dfi.iter().enumerate() {
                primary_slot_by_original_vertex[original_vertex] = dfi;
                slots.push(EmbeddingSlot {
                    kind: EmbeddingSlotKind::Primary { original_vertex },
                    first_arc: None,
                    last_arc: None,
                    ext_face: [None, None],
                    visited: false,
                    visited_info: number_of_vertices,
                    pertinent_roots: Vec::new(),
                    future_pertinent_child: None,
                    pertinent_edge: None,
                    k33_merge_blocker: None,
                    k33_minor_e_reduced: false,
                    sorted_dfs_children: Vec::new(),
                    separated_dfs_children: Vec::new(),
                });
            }

            let mut root_copy_by_primary_dfi = vec![None; number_of_vertices];
            for (dfi, &original_vertex) in preprocessing.vertex_by_dfi.iter().enumerate() {
                let Some(parent_original_vertex) = preprocessing.vertices[original_vertex].parent
                else {
                    continue;
                };
                let parent_primary_slot = primary_slot_by_original_vertex[parent_original_vertex];
                let root_copy_slot = slots.len();
                root_copy_by_primary_dfi[dfi] = Some(root_copy_slot);
                slots.push(EmbeddingSlot {
                    kind: EmbeddingSlotKind::RootCopy {
                        parent_primary_slot,
                        dfs_child_primary_slot: dfi,
                    },
                    first_arc: None,
                    last_arc: None,
                    ext_face: [Some(dfi), Some(dfi)],
                    visited: false,
                    visited_info: number_of_vertices,
                    pertinent_roots: Vec::new(),
                    future_pertinent_child: None,
                    pertinent_edge: None,
                    k33_merge_blocker: None,
                    k33_minor_e_reduced: false,
                    sorted_dfs_children: Vec::new(),
                    separated_dfs_children: Vec::new(),
                });
            }

            let mut least_ancestor_by_primary_slot = vec![usize::MAX; number_of_vertices];
            let mut lowpoint_by_primary_slot = vec![usize::MAX; number_of_vertices];
            for (primary_slot, &original_vertex) in preprocessing.vertex_by_dfi.iter().enumerate() {
                let sorted_dfs_children = preprocessing.vertices[original_vertex]
                    .sorted_dfs_children
                    .iter()
                    .map(|&child_original_vertex| {
                        primary_slot_by_original_vertex[child_original_vertex]
                    })
                    .collect::<Vec<_>>();
                let mut separated_dfs_children = sorted_dfs_children.clone();
                separated_dfs_children.sort_unstable_by_key(|&child_slot| {
                    (
                        preprocessing.vertices[preprocessing.vertex_by_dfi[child_slot]].lowpoint,
                        child_slot,
                    )
                });
                slots[primary_slot].future_pertinent_child = sorted_dfs_children.first().copied();
                slots[primary_slot].sorted_dfs_children = sorted_dfs_children;
                slots[primary_slot].separated_dfs_children = separated_dfs_children;
                least_ancestor_by_primary_slot[primary_slot] =
                    preprocessing.vertices[original_vertex].least_ancestor;
                lowpoint_by_primary_slot[primary_slot] =
                    preprocessing.vertices[original_vertex].lowpoint;
            }

            let mut arcs = preprocessing
                .arcs
                .iter()
                .enumerate()
                .map(|(original_arc, arc)| {
                    EmbeddingArcRecord {
                        original_arc: Some(original_arc),
                        source_slot: primary_slot_by_original_vertex[arc.source],
                        target_slot: primary_slot_by_original_vertex[arc.target],
                        twin: arc.twin,
                        next: None,
                        prev: None,
                        visited: false,
                        kind: arc.kind,
                        embedded: false,
                        inverted: false,
                        reduction_endpoint_arc: None,
                    }
                })
                .collect::<Vec<_>>();

            for (primary_slot, &original_vertex) in preprocessing.vertex_by_dfi.iter().enumerate() {
                let Some(parent_arc) = preprocessing.vertices[original_vertex].parent_arc else {
                    continue;
                };
                let child_arc = preprocessing.arcs[parent_arc].twin;
                let root_copy_slot = root_copy_by_primary_dfi[primary_slot]
                    .expect("non-root DFS child must have an associated root copy");

                arcs[parent_arc].source_slot = primary_slot;
                arcs[parent_arc].target_slot = root_copy_slot;
                arcs[parent_arc].embedded = true;

                arcs[child_arc].source_slot = root_copy_slot;
                arcs[child_arc].target_slot = primary_slot;
                arcs[child_arc].embedded = true;

                slots[primary_slot].first_arc = Some(parent_arc);
                slots[primary_slot].last_arc = Some(parent_arc);
                slots[primary_slot].ext_face = [Some(root_copy_slot), Some(root_copy_slot)];

                slots[root_copy_slot].first_arc = Some(child_arc);
                slots[root_copy_slot].last_arc = Some(child_arc);
                slots[root_copy_slot].ext_face = [Some(primary_slot), Some(primary_slot)];
            }

            Self {
                slots,
                arcs,
                primary_slot_by_original_vertex,
                root_copy_by_primary_dfi,
                least_ancestor_by_primary_slot,
                lowpoint_by_primary_slot,
                forward_arc_head_index_by_primary_slot: preprocessing
                    .vertex_by_dfi
                    .iter()
                    .map(|&original_vertex| {
                        (!preprocessing.vertices[original_vertex].sorted_forward_arcs.is_empty())
                            .then_some(0)
                    })
                    .collect(),
                handling_k4_blocked_bicomp: false,
                k4_reblocked_same_root: false,
            }
        }

        fn ext_face_vertex(&self, slot: usize, side: usize) -> usize {
            self.slots[slot].ext_face[side].unwrap_or_else(|| {
                panic!(
                    "external-face link must be initialized: slot={slot}, side={side}, kind={:?}, first_arc={:?}, last_arc={:?}, ext_face={:?}",
                    self.slots[slot].kind,
                    self.slots[slot].first_arc,
                    self.slots[slot].last_arc,
                    self.slots[slot].ext_face
                )
            })
        }

        fn shortcut_ext_face_neighbor(
            &self,
            current_slot: usize,
            previous_link: &mut usize,
        ) -> usize {
            let next_slot = self.ext_face_vertex(current_slot, 1 ^ *previous_link);
            if !self.is_singleton_slot(next_slot) {
                *previous_link = usize::from(self.ext_face_vertex(next_slot, 0) != current_slot);
            }
            next_slot
        }

        fn walk_ext_face_neighbor(
            &self,
            mode: super::EmbeddingRunMode,
            current_slot: usize,
            previous_link: &mut usize,
        ) -> usize {
            let _ = mode;
            self.shortcut_ext_face_neighbor(current_slot, previous_link)
        }

        fn slot_arc(&self, slot: usize, side: usize) -> Option<usize> {
            if side == 0 { self.slots[slot].first_arc } else { self.slots[slot].last_arc }
        }

        fn set_slot_arc(&mut self, slot: usize, side: usize, arc: Option<usize>) {
            if side == 0 {
                self.slots[slot].first_arc = arc;
            } else {
                self.slots[slot].last_arc = arc;
            }
        }

        fn set_adjacent_arc(&mut self, arc: usize, link: usize, adjacent: Option<usize>) {
            if link == 0 {
                self.arcs[arc].next = adjacent;
            } else {
                self.arcs[arc].prev = adjacent;
            }
        }

        fn insert_arc_at_position(
            &mut self,
            slot: usize,
            arc: usize,
            prev_arc: Option<usize>,
            next_arc: Option<usize>,
        ) {
            self.arcs[arc].source_slot = slot;
            self.arcs[arc].prev = prev_arc;
            self.arcs[arc].next = next_arc;

            if let Some(prev_arc) = prev_arc {
                self.arcs[prev_arc].next = Some(arc);
            } else {
                self.slots[slot].first_arc = Some(arc);
            }

            if let Some(next_arc) = next_arc {
                self.arcs[next_arc].prev = Some(arc);
            } else {
                self.slots[slot].last_arc = Some(arc);
            }
        }

        fn unlink_arc_from_slot(&mut self, arc: usize) {
            let slot = self.arcs[arc].source_slot;
            let prev_arc = self.arcs[arc].prev;
            let next_arc = self.arcs[arc].next;

            if let Some(prev_arc) = prev_arc {
                self.arcs[prev_arc].next = next_arc;
            } else {
                self.slots[slot].first_arc = next_arc;
            }

            if let Some(next_arc) = next_arc {
                self.arcs[next_arc].prev = prev_arc;
            } else {
                self.slots[slot].last_arc = prev_arc;
            }
        }

        fn unlink_arc_pair(&mut self, arc: usize) {
            let twin = self.arcs[arc].twin;
            self.unlink_arc_from_slot(arc);
            self.unlink_arc_from_slot(twin);
        }

        fn normalize_slot_boundary_arcs(&mut self, slot: usize) {
            let first_live =
                self.slots[slot].first_arc.filter(|&arc| self.arcs[arc].source_slot == slot);
            let last_live =
                self.slots[slot].last_arc.filter(|&arc| self.arcs[arc].source_slot == slot);

            match (first_live, last_live) {
                (Some(mut first_arc), Some(mut last_arc)) => {
                    while let Some(prev_arc) = self.arcs[first_arc].prev {
                        if self.arcs[prev_arc].source_slot != slot {
                            break;
                        }
                        first_arc = prev_arc;
                    }
                    while let Some(next_arc) = self.arcs[last_arc].next {
                        if self.arcs[next_arc].source_slot != slot {
                            break;
                        }
                        last_arc = next_arc;
                    }
                    self.slots[slot].first_arc = Some(first_arc);
                    self.slots[slot].last_arc = Some(last_arc);
                }
                (Some(first_arc), None) => {
                    let mut last_arc = first_arc;
                    while let Some(next_arc) = self.arcs[last_arc].next {
                        if self.arcs[next_arc].source_slot != slot {
                            break;
                        }
                        last_arc = next_arc;
                    }
                    self.slots[slot].first_arc = Some(first_arc);
                    self.slots[slot].last_arc = Some(last_arc);
                }
                (None, Some(last_arc)) => {
                    let mut first_arc = last_arc;
                    while let Some(prev_arc) = self.arcs[first_arc].prev {
                        if self.arcs[prev_arc].source_slot != slot {
                            break;
                        }
                        first_arc = prev_arc;
                    }
                    self.slots[slot].first_arc = Some(first_arc);
                    self.slots[slot].last_arc = Some(last_arc);
                }
                (None, None) => {
                    self.slots[slot].first_arc = None;
                    self.slots[slot].last_arc = None;
                }
            }
        }

        fn delete_arc_pair_permanently(&mut self, arc: usize) {
            if self.arcs[arc].source_slot == usize::MAX {
                return;
            }

            let source_slot = self.arcs[arc].source_slot;
            let twin = self.arcs[arc].twin;
            let twin_source_slot = self.arcs[twin].source_slot;
            self.unlink_arc_pair(arc);

            for dead_arc in [arc, twin] {
                self.arcs[dead_arc].source_slot = usize::MAX;
                self.arcs[dead_arc].target_slot = usize::MAX;
                self.arcs[dead_arc].prev = None;
                self.arcs[dead_arc].next = None;
                self.arcs[dead_arc].visited = false;
                self.arcs[dead_arc].embedded = false;
                self.arcs[dead_arc].reduction_endpoint_arc = None;
            }

            self.normalize_slot_boundary_arcs(source_slot);
            if twin_source_slot != source_slot {
                self.normalize_slot_boundary_arcs(twin_source_slot);
            }
        }

        fn push_synthetic_arc_pair(
            &mut self,
            source_slot: usize,
            target_slot: usize,
            source_kind: DfsArcType,
            target_kind: DfsArcType,
            source_reduction_endpoint_arc: Option<usize>,
            target_reduction_endpoint_arc: Option<usize>,
        ) -> usize {
            let forward_arc = self.arcs.len();
            let backward_arc = forward_arc + 1;

            self.arcs.push(EmbeddingArcRecord {
                original_arc: None,
                source_slot,
                target_slot,
                twin: backward_arc,
                next: None,
                prev: None,
                visited: false,
                kind: source_kind,
                embedded: true,
                inverted: false,
                reduction_endpoint_arc: source_reduction_endpoint_arc,
            });
            self.arcs.push(EmbeddingArcRecord {
                original_arc: None,
                source_slot: target_slot,
                target_slot: source_slot,
                twin: forward_arc,
                next: None,
                prev: None,
                visited: false,
                kind: target_kind,
                embedded: true,
                inverted: false,
                reduction_endpoint_arc: target_reduction_endpoint_arc,
            });

            forward_arc
        }

        fn restore_reduced_path_edge(&mut self, reduction_arc: usize) -> bool {
            let Some(source_endpoint_arc) = self.arcs[reduction_arc].reduction_endpoint_arc else {
                return false;
            };
            let twin = self.arcs[reduction_arc].twin;
            let target_endpoint_arc = self.arcs[twin]
                .reduction_endpoint_arc
                .expect("reduction edges must store both endpoints");

            let source_prev = self.arcs[reduction_arc].prev;
            let source_next = self.arcs[reduction_arc].next;
            let target_prev = self.arcs[twin].prev;
            let target_next = self.arcs[twin].next;
            let source_slot = self.arcs[reduction_arc].source_slot;
            let target_slot = self.arcs[twin].source_slot;
            let source_endpoint_twin = self.arcs[source_endpoint_arc].twin;
            let target_endpoint_twin = self.arcs[target_endpoint_arc].twin;

            self.unlink_arc_pair(reduction_arc);
            self.arcs[reduction_arc].reduction_endpoint_arc = None;
            self.arcs[twin].reduction_endpoint_arc = None;

            self.insert_arc_at_position(source_slot, source_endpoint_arc, source_prev, source_next);
            self.insert_arc_at_position(
                self.arcs[source_endpoint_twin].source_slot,
                source_endpoint_twin,
                self.arcs[source_endpoint_twin].prev,
                self.arcs[source_endpoint_twin].next,
            );
            self.insert_arc_at_position(target_slot, target_endpoint_arc, target_prev, target_next);
            self.insert_arc_at_position(
                self.arcs[target_endpoint_twin].source_slot,
                target_endpoint_twin,
                self.arcs[target_endpoint_twin].prev,
                self.arcs[target_endpoint_twin].next,
            );

            true
        }

        fn reduce_external_face_path_to_edge(
            &mut self,
            start_slot: usize,
            start_side: usize,
            end_slot: usize,
            end_side: usize,
            start_kind: DfsArcType,
            end_kind: DfsArcType,
        ) -> Result<Option<usize>, EmbeddingMutationError> {
            let mut start_arc = self.slot_arc(start_slot, start_side).ok_or(
                EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: start_side },
            )?;
            if self.restore_reduced_path_edge(start_arc) {
                start_arc = self.slot_arc(start_slot, start_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: start_side },
                )?;
            }

            if self.arcs[start_arc].target_slot == end_slot {
                self.slots[start_slot].ext_face[start_side] = Some(end_slot);
                self.slots[end_slot].ext_face[end_side] = Some(start_slot);
                return Ok(None);
            }

            let mut end_arc = self
                .slot_arc(end_slot, end_side)
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: end_side })?;
            if self.restore_reduced_path_edge(end_arc) {
                end_arc = self.slot_arc(end_slot, end_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: end_side },
                )?;
            }

            let start_prev = self.arcs[start_arc].prev;
            let start_next = self.arcs[start_arc].next;
            let end_prev = self.arcs[end_arc].prev;
            let end_next = self.arcs[end_arc].next;

            self.unlink_arc_pair(start_arc);
            self.unlink_arc_pair(end_arc);

            let reduction_arc = self.push_synthetic_arc_pair(
                start_slot,
                end_slot,
                start_kind,
                end_kind,
                Some(start_arc),
                Some(end_arc),
            );
            let reduction_twin = self.arcs[reduction_arc].twin;

            self.insert_arc_at_position(start_slot, reduction_arc, start_prev, start_next);
            self.insert_arc_at_position(end_slot, reduction_twin, end_prev, end_next);

            self.slots[start_slot].ext_face[start_side] = Some(end_slot);
            self.slots[end_slot].ext_face[end_side] = Some(start_slot);

            Ok(Some(reduction_arc))
        }

        fn external_face_entry_side(
            &self,
            start_slot: usize,
            start_side: usize,
            end_slot: usize,
        ) -> Option<usize> {
            let mut previous_link = 1 ^ start_side;
            let mut current_slot = start_slot;

            loop {
                let next_slot = self.real_ext_face_neighbor(current_slot, &mut previous_link);
                if next_slot == end_slot {
                    return Some(previous_link);
                }
                if next_slot == start_slot {
                    return None;
                }
                current_slot = next_slot;
            }
        }

        fn rebuild_shortcut_ext_face_for_bicomp(
            &mut self,
            root_copy_slot: usize,
        ) -> Result<(), WalkDownExecutionError> {
            let bicomp_slots = self.collect_bicomp_slots(root_copy_slot);
            for slot in bicomp_slots {
                self.slots[slot].ext_face = [None, None];
            }

            let mut current_slot = root_copy_slot;
            let mut previous_link = 1usize;
            let mut steps_remaining = self.slots.len();

            loop {
                if steps_remaining == 0 {
                    return Err(WalkDownExecutionError::InvalidK4Context);
                }
                let entry_link = previous_link;
                let next_slot = self.real_ext_face_neighbor(current_slot, &mut previous_link);
                self.slots[current_slot].ext_face[1 ^ entry_link] = Some(next_slot);
                self.slots[next_slot].ext_face[previous_link] = Some(current_slot);

                if next_slot == root_copy_slot {
                    break;
                }

                current_slot = next_slot;
                steps_remaining -= 1;
            }

            Ok(())
        }

        fn reduce_xy_path_to_edge_with_explicit_kinds(
            &mut self,
            start_slot: usize,
            end_slot: usize,
            start_kind: DfsArcType,
            end_kind: DfsArcType,
        ) -> Result<Option<usize>, EmbeddingMutationError> {
            let mut start_outer_arc = self.slots[start_slot]
                .first_arc
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: 0 })?;
            let mut start_xy_arc = self.next_arc_circular(start_slot, start_outer_arc);
            if self.restore_reduced_path_edge(start_xy_arc) {
                start_outer_arc = self.slots[start_slot]
                    .first_arc
                    .ok_or(EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: 0 })?;
                start_xy_arc = self.next_arc_circular(start_slot, start_outer_arc);
            }

            if self.arcs[start_xy_arc].target_slot == end_slot {
                return Ok(None);
            }

            let mut end_outer_arc = self.slots[end_slot]
                .first_arc
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: 0 })?;
            let mut end_xy_arc = self.next_arc_circular(end_slot, end_outer_arc);
            if self.restore_reduced_path_edge(end_xy_arc) {
                end_outer_arc = self.slots[end_slot]
                    .first_arc
                    .ok_or(EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: 0 })?;
                end_xy_arc = self.next_arc_circular(end_slot, end_outer_arc);
            }

            let start_prev = self.arcs[start_xy_arc].prev;
            let start_next = self.arcs[start_xy_arc].next;
            let end_prev = self.arcs[end_xy_arc].prev;
            let end_next = self.arcs[end_xy_arc].next;

            self.unlink_arc_pair(start_xy_arc);
            self.unlink_arc_pair(end_xy_arc);

            let reduction_arc = self.push_synthetic_arc_pair(
                start_slot,
                end_slot,
                start_kind,
                end_kind,
                Some(start_xy_arc),
                Some(end_xy_arc),
            );
            let reduction_twin = self.arcs[reduction_arc].twin;

            self.insert_arc_at_position(start_slot, reduction_arc, start_prev, start_next);
            self.insert_arc_at_position(end_slot, reduction_twin, end_prev, end_next);

            Ok(Some(reduction_arc))
        }

        fn real_ext_face_neighbor(&self, current_slot: usize, previous_link: &mut usize) -> usize {
            let exit_arc = self.slot_arc(current_slot, 1 ^ *previous_link).unwrap_or_else(|| {
                panic!(
                    "external-face traversal requires an incident arc: slot={current_slot}, kind={:?}, first_arc={:?}, last_arc={:?}",
                    self.slots[current_slot].kind,
                    self.slots[current_slot].first_arc,
                    self.slots[current_slot].last_arc,
                )
            });
            let next_slot = self.arcs[exit_arc].target_slot;
            let entry_arc = self.arcs[exit_arc].twin;

            if self.slots[next_slot].first_arc != self.slots[next_slot].last_arc {
                *previous_link = usize::from(self.slots[next_slot].first_arc != Some(entry_arc));
            }

            next_slot
        }

        fn k4_ext_face_neighbor(&self, current_slot: usize, previous_link: &mut usize) -> usize {
            self.real_ext_face_neighbor(current_slot, previous_link)
        }

        #[allow(dead_code)]
        fn is_active(&self, slot: usize, current_primary_slot: usize) -> bool {
            self.is_pertinent(slot) || self.is_future_pertinent(slot, current_primary_slot)
        }

        fn find_pertinent_vertex_on_lower_face(
            &self,
            x_slot: usize,
            y_slot: usize,
        ) -> Option<usize> {
            let mut candidate_slot = x_slot;
            let mut previous_link = 1usize;

            candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            while candidate_slot != y_slot && candidate_slot != x_slot {
                if self.is_pertinent(candidate_slot) {
                    return Some(candidate_slot);
                }
                candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            }

            None
        }

        #[allow(dead_code)]
        fn find_nonouterplanarity_context(
            &self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
        ) -> Option<NonOuterplanarityContext> {
            let mut x_prev_link = 1usize;
            let mut y_prev_link = 0usize;
            let x_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut x_prev_link);
            let y_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut y_prev_link);
            let w_slot = self.find_pertinent_vertex_on_lower_face(x_slot, y_slot)?;

            Some(NonOuterplanarityContext {
                current_primary_slot,
                x_slot,
                y_slot,
                w_slot,
                x_prev_link,
                y_prev_link,
            })
        }

        #[allow(dead_code)]
        fn find_active_vertices(
            &mut self,
            bicomp_root_copy_slot: usize,
            current_primary_slot: usize,
        ) -> Result<(usize, usize, usize, usize), K33ContextInitFailure> {
            let mut x_prev_link = 1usize;
            let mut x_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut x_prev_link);
            let x_start_slot = x_slot;
            loop {
                self.update_future_pertinent_child(x_slot, current_primary_slot);
                if self.is_active(x_slot, current_primary_slot) {
                    break;
                }
                x_slot = self.real_ext_face_neighbor(x_slot, &mut x_prev_link);
                if x_slot == bicomp_root_copy_slot || x_slot == x_start_slot {
                    return Err(K33ContextInitFailure::NoActiveX);
                }
            }

            let mut y_prev_link = 0usize;
            let mut y_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut y_prev_link);
            let y_start_slot = y_slot;
            loop {
                self.update_future_pertinent_child(y_slot, current_primary_slot);
                if self.is_active(y_slot, current_primary_slot) {
                    break;
                }
                y_slot = self.real_ext_face_neighbor(y_slot, &mut y_prev_link);
                if y_slot == bicomp_root_copy_slot || y_slot == y_start_slot {
                    return Err(K33ContextInitFailure::NoActiveY);
                }
            }

            Ok((x_slot, y_slot, x_prev_link, y_prev_link))
        }

        #[allow(dead_code)]
        fn find_pertinent_vertex_between_active_sides(
            &self,
            x_slot: usize,
            y_slot: usize,
        ) -> Option<usize> {
            let mut candidate_slot = x_slot;
            let mut previous_link = 1usize;

            candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            while candidate_slot != y_slot && candidate_slot != x_slot {
                if self.is_pertinent(candidate_slot) {
                    return Some(candidate_slot);
                }
                candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            }

            None
        }

        fn initialize_nonouterplanarity_context(
            &mut self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
            stack_root_copy_slot: Option<usize>,
        ) -> Result<NonplanarityContext, K33ContextInitFailure> {
            let root_copy_slot = stack_root_copy_slot.unwrap_or(bicomp_root_copy_slot);

            self.orient_bicomp_from_root(root_copy_slot, true);
            self.clear_all_visited_flags_in_bicomp(root_copy_slot);

            let mut x_prev_link = 1usize;
            let x_slot = self.real_ext_face_neighbor(root_copy_slot, &mut x_prev_link);
            let mut y_prev_link = 0usize;
            let y_slot = self.real_ext_face_neighbor(root_copy_slot, &mut y_prev_link);
            let Some(w_slot) = self.find_pertinent_vertex_between_active_sides(x_slot, y_slot)
            else {
                return Err(K33ContextInitFailure::NoPertinentBetweenActiveSides);
            };

            Ok(NonplanarityContext {
                current_primary_slot,
                root_copy_slot,
                x_slot,
                y_slot,
                w_slot,
                x_prev_link,
                y_prev_link,
            })
        }

        #[allow(dead_code)]
        fn find_pertinent_vertex_between_active_sides_parallel(
            &self,
            x_slot: usize,
            x_prev_link: usize,
            y_slot: usize,
            y_prev_link: usize,
        ) -> Option<usize> {
            let mut left_slot = x_slot;
            let mut left_prev_link = x_prev_link;
            let mut right_slot = y_slot;
            let mut right_prev_link = y_prev_link;

            while left_slot != y_slot {
                left_slot = self.real_ext_face_neighbor(left_slot, &mut left_prev_link);
                if self.is_pertinent(left_slot) {
                    return Some(left_slot);
                }

                right_slot = self.real_ext_face_neighbor(right_slot, &mut right_prev_link);
                if self.is_pertinent(right_slot) {
                    return Some(right_slot);
                }
            }

            None
        }

        fn find_k4_pertinent_vertex_between_active_sides_parallel(
            &self,
            x_slot: usize,
            x_prev_link: usize,
            y_slot: usize,
            y_prev_link: usize,
        ) -> Option<usize> {
            let mut left_slot = x_slot;
            let mut left_prev_link = x_prev_link;
            let mut right_slot = y_slot;
            let mut right_prev_link = y_prev_link;

            while left_slot != y_slot {
                left_slot = self.k4_ext_face_neighbor(left_slot, &mut left_prev_link);
                if self.is_pertinent(left_slot) {
                    return Some(left_slot);
                }

                right_slot = self.k4_ext_face_neighbor(right_slot, &mut right_prev_link);
                if self.is_pertinent(right_slot) {
                    return Some(right_slot);
                }
            }

            None
        }

        #[allow(dead_code)]
        #[allow(clippy::uninlined_format_args)]
        fn initialize_k4_context(
            &mut self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
        ) -> Result<K4Context, WalkDownExecutionError> {
            let root_copy_slot = bicomp_root_copy_slot;
            let mut x_prev_link = 1usize;
            let x_slot = self.k4_ext_face_neighbor(root_copy_slot, &mut x_prev_link);
            let mut y_prev_link = 0usize;
            let y_slot = self.k4_ext_face_neighbor(root_copy_slot, &mut y_prev_link);
            let Some(w_slot) = self.find_k4_pertinent_vertex_between_active_sides_parallel(
                x_slot,
                x_prev_link,
                y_slot,
                y_prev_link,
            ) else {
                return Err(WalkDownExecutionError::InvalidK4Context);
            };

            Ok(K4Context {
                current_primary_slot,
                root_copy_slot,
                x_slot,
                y_slot,
                w_slot,
                x_prev_link,
                y_prev_link,
            })
        }

        #[allow(dead_code)]
        fn classify_k4_minor(
            &mut self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
        ) -> Result<(K4Context, K4MinorType), WalkDownExecutionError> {
            let context =
                self.initialize_k4_context(current_primary_slot, bicomp_root_copy_slot)?;

            let minor_type =
                if self.primary_from_root(context.root_copy_slot) != current_primary_slot {
                    K4MinorType::A
                } else if !self.slots[context.w_slot].pertinent_roots.is_empty() {
                    K4MinorType::B
                } else {
                    K4MinorType::E
                };

            Ok((context, minor_type))
        }

        #[allow(dead_code)]
        #[allow(clippy::uninlined_format_args)]
        fn find_k4_second_active_vertex_on_low_ext_face_path(
            &mut self,
            context: &K4Context,
        ) -> Option<(usize, usize)> {
            let mut z_slot = context.root_copy_slot;
            let mut z_prev_link = 1usize;

            z_slot = self.k4_ext_face_neighbor(z_slot, &mut z_prev_link);

            self.update_future_pertinent_child(z_slot, context.current_primary_slot);
            if self.is_future_pertinent(z_slot, context.current_primary_slot) {
                let uz = self.least_ancestor_connection(z_slot, context.current_primary_slot)?;
                return Some((z_slot, uz));
            }

            z_slot = self.k4_ext_face_neighbor(z_slot, &mut z_prev_link);
            while z_slot != context.y_slot {
                if z_slot != context.w_slot {
                    self.update_future_pertinent_child(z_slot, context.current_primary_slot);
                    if self.is_future_pertinent(z_slot, context.current_primary_slot) {
                        let uz =
                            self.least_ancestor_connection(z_slot, context.current_primary_slot)?;
                        return Some((z_slot, uz));
                    }
                    if self.is_pertinent(z_slot) {
                        return Some((z_slot, context.current_primary_slot));
                    }
                }
                z_slot = self.k4_ext_face_neighbor(z_slot, &mut z_prev_link);
            }

            self.update_future_pertinent_child(z_slot, context.current_primary_slot);
            if self.is_future_pertinent(z_slot, context.current_primary_slot) {
                let uz = self.least_ancestor_connection(z_slot, context.current_primary_slot)?;
                return Some((z_slot, uz));
            }

            None
        }

        #[allow(dead_code)]
        fn find_k4_planarity_active_vertex(
            &mut self,
            context: &K4Context,
            prev_link: usize,
        ) -> Option<usize> {
            let mut slot = context.root_copy_slot;
            let mut slot_prev_link = prev_link;

            slot = self.k4_ext_face_neighbor(slot, &mut slot_prev_link);
            while slot != context.root_copy_slot {
                if self.is_pertinent(slot) {
                    return Some(slot);
                }

                self.update_future_pertinent_child(slot, context.current_primary_slot);
                if self.is_future_pertinent(slot, context.current_primary_slot) {
                    return Some(slot);
                }

                slot = self.k4_ext_face_neighbor(slot, &mut slot_prev_link);
            }

            None
        }

        #[allow(dead_code)]
        fn find_k4_separating_internal_edge(
            &mut self,
            context: &K4Context,
            prev_link: usize,
            active_slot: usize,
        ) -> Result<Option<(usize, usize, usize)>, WalkDownExecutionError> {
            let mut path_marks = vec![false; self.slots.len()];
            path_marks[context.root_copy_slot] = true;
            let mut mark_prev_link = prev_link;
            let mut marked_slot = context.root_copy_slot;
            loop {
                if marked_slot == active_slot {
                    break;
                }
                marked_slot = self.k4_ext_face_neighbor(marked_slot, &mut mark_prev_link);
                path_marks[marked_slot] = true;
                if marked_slot == context.root_copy_slot {
                    return Err(WalkDownExecutionError::InvalidK4Context);
                }
            }

            let mut separator = None;
            let mut slot_prev_link = prev_link;
            let mut slot = self.k4_ext_face_neighbor(context.root_copy_slot, &mut slot_prev_link);

            while slot != active_slot {
                let mut arc = self.slots[slot].first_arc;
                while let Some(current_arc) = arc {
                    let neighbor_slot = self.arcs[current_arc].target_slot;
                    if !path_marks[neighbor_slot] {
                        separator = Some((active_slot, slot, neighbor_slot));
                        break;
                    }
                    arc = self.arcs[current_arc].next;
                }

                if separator.is_some() {
                    break;
                }

                slot = self.k4_ext_face_neighbor(slot, &mut slot_prev_link);
            }
            Ok(separator)
        }

        fn set_k4_vertex_types_for_marking_xy_path(
            &self,
            context: &K4Context,
        ) -> Vec<K4ObstructionMark> {
            let mut marks = vec![K4ObstructionMark::Unmarked; self.slots.len()];

            let mut slot = context.root_copy_slot;
            let mut previous_link = 1usize;
            let mut mark = K4ObstructionMark::HighRxw;
            loop {
                slot = self.k4_ext_face_neighbor(slot, &mut previous_link);
                if slot == context.w_slot {
                    break;
                }
                if slot == context.x_slot {
                    mark = K4ObstructionMark::LowRxw;
                }
                marks[slot] = mark;
            }

            slot = context.root_copy_slot;
            previous_link = 0usize;
            mark = K4ObstructionMark::HighRyw;
            loop {
                slot = self.k4_ext_face_neighbor(slot, &mut previous_link);
                if slot == context.w_slot {
                    break;
                }
                if slot == context.y_slot {
                    mark = K4ObstructionMark::LowRyw;
                }
                marks[slot] = mark;
            }

            marks
        }

        #[allow(clippy::uninlined_format_args)]
        fn mark_k4_closest_xy_path(
            &mut self,
            context: &K4Context,
            target_slot: usize,
            obstruction_marks: &[K4ObstructionMark],
        ) -> Result<Option<K4MarkedPath>, WalkDownExecutionError> {
            let mut marked_path = K4MarkedPath { px_slot: usize::MAX, py_slot: usize::MAX };
            let mut stack = Vec::new();
            let target_is_root = target_slot == context.root_copy_slot;
            let antipodal_slot = if target_is_root {
                context.w_slot
            } else if target_slot == context.w_slot {
                context.root_copy_slot
            } else {
                return Err(WalkDownExecutionError::InvalidK4Context);
            };
            let hidden_pairs = self.hide_internal_edges_at_slot(target_slot);

            let mut current_slot = target_slot;
            let mut entry_arc = self
                .slot_arc(target_slot, usize::from(target_is_root))
                .ok_or(WalkDownExecutionError::InvalidK4Context)?;

            loop {
                let exit_arc = if current_slot == target_slot {
                    if target_is_root {
                        self.prev_arc_circular(current_slot, entry_arc)
                    } else {
                        self.next_arc_circular(current_slot, entry_arc)
                    }
                } else if target_is_root {
                    self.prev_arc_circular(current_slot, entry_arc)
                } else {
                    self.next_arc_circular(current_slot, entry_arc)
                };
                let next_slot = self.arcs[exit_arc].target_slot;
                let next_entry_arc = self.arcs[exit_arc].twin;

                if self.slots[next_slot].visited {
                    self.pop_marked_path_until_slot(&mut stack, next_slot);
                } else {
                    if next_slot == antipodal_slot {
                        self.clear_marked_path(&mut stack);
                        self.restore_hidden_arc_pairs(hidden_pairs);
                        return Ok(None);
                    }

                    match obstruction_marks[next_slot] {
                        K4ObstructionMark::HighRxw | K4ObstructionMark::LowRxw => {
                            marked_path.px_slot = next_slot;
                            self.clear_marked_path(&mut stack);
                        }
                        _ => {}
                    }

                    stack.push((next_slot, next_entry_arc));
                    self.slots[next_slot].visited = true;
                    if next_slot != marked_path.px_slot {
                        let twin = self.arcs[next_entry_arc].twin;
                        self.arcs[next_entry_arc].visited = true;
                        self.arcs[twin].visited = true;
                    }

                    if matches!(
                        obstruction_marks[next_slot],
                        K4ObstructionMark::HighRyw | K4ObstructionMark::LowRyw
                    ) {
                        marked_path.py_slot = next_slot;
                        self.restore_hidden_arc_pairs(hidden_pairs);
                        return Ok(Some(marked_path));
                    }
                }

                current_slot = next_slot;
                entry_arc = next_entry_arc;
            }
        }

        fn mark_k4_highest_xy_path(
            &mut self,
            context: &K4Context,
            obstruction_marks: &[K4ObstructionMark],
        ) -> Result<Option<K4MarkedPath>, WalkDownExecutionError> {
            self.mark_k4_closest_xy_path(context, context.root_copy_slot, obstruction_marks)
        }

        #[allow(dead_code)]
        fn find_k4_xy_path(&mut self, context: &K4Context) -> Result<bool, WalkDownExecutionError> {
            let obstruction_marks = self.set_k4_vertex_types_for_marking_xy_path(context);
            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            let marked_path = self.mark_k4_highest_xy_path(context, &obstruction_marks)?;
            let has_xy_path = marked_path.is_some();
            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            Ok(has_xy_path)
        }

        #[allow(dead_code)]
        fn arc_side(&self, slot: usize, arc: usize) -> Option<usize> {
            if self.slots[slot].first_arc == Some(arc) {
                Some(0)
            } else if self.slots[slot].last_arc == Some(arc) {
                Some(1)
            } else {
                None
            }
        }

        #[allow(dead_code)]
        fn find_arc_to_target(&self, slot: usize, target_slot: usize) -> Option<usize> {
            let mut arc = self.slots[slot].first_arc;
            while let Some(current_arc) = arc {
                if self.arcs[current_arc].target_slot == target_slot {
                    return Some(current_arc);
                }
                arc = self.arcs[current_arc].next;
            }
            None
        }

        fn is_parent_primary_to_root_copy_step(
            &self,
            parent_slot: usize,
            child_slot: usize,
        ) -> bool {
            matches!(
                self.slots[child_slot].kind,
                EmbeddingSlotKind::RootCopy {
                    parent_primary_slot,
                    ..
                } if parent_primary_slot == parent_slot
            )
        }

        #[allow(dead_code)]
        fn tree_path_slots_between_ancestor_and_descendant(
            &self,
            ancestor_slot: usize,
            descendant_slot: usize,
        ) -> Result<Vec<usize>, EmbeddingMutationError> {
            let mut reverse_slots = vec![descendant_slot];
            let mut current_slot = descendant_slot;
            let step_limit = self.slots.len() + 1;

            for _ in 0..step_limit {
                if current_slot == ancestor_slot {
                    reverse_slots.reverse();
                    return Ok(reverse_slots);
                }

                current_slot = match self.slots[current_slot].kind {
                    EmbeddingSlotKind::Primary { .. } => {
                        let parent_arc = self.parent_arc_in_embedding(current_slot).ok_or(
                            EmbeddingMutationError::MissingExternalFacePath {
                                start_slot: ancestor_slot,
                                start_side: 0,
                                end_slot: descendant_slot,
                            },
                        )?;
                        self.arcs[parent_arc].target_slot
                    }
                    EmbeddingSlotKind::RootCopy { parent_primary_slot, .. } => parent_primary_slot,
                };
                reverse_slots.push(current_slot);
            }

            Err(EmbeddingMutationError::MissingExternalFacePath {
                start_slot: ancestor_slot,
                start_side: 0,
                end_slot: descendant_slot,
            })
        }

        #[allow(dead_code)]
        fn mark_tree_path_slots_visited(
            &mut self,
            path_slots: &[usize],
        ) -> Result<(), EmbeddingMutationError> {
            for &slot in path_slots {
                self.slots[slot].visited = true;
            }

            for slots in path_slots.windows(2) {
                if let Some(arc) = self.find_arc_to_target(slots[0], slots[1]) {
                    self.arcs[arc].visited = true;
                    let twin = self.arcs[arc].twin;
                    self.arcs[twin].visited = true;
                    continue;
                }

                if self.is_parent_primary_to_root_copy_step(slots[0], slots[1]) {
                    continue;
                }

                return Err(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: path_slots[0],
                    start_side: 0,
                    end_slot: *path_slots.last().unwrap_or(&path_slots[0]),
                });
            }

            Ok(())
        }

        fn parent_arc_in_embedding(&self, slot: usize) -> Option<usize> {
            let mut arc = self.slots[slot].first_arc;
            while let Some(current_arc) = arc {
                if self.arcs[current_arc].kind == DfsArcType::Parent {
                    return Some(current_arc);
                }
                arc = self.arcs[current_arc].next;
            }
            None
        }

        fn mark_current_dfs_path_in_bicomp(
            &mut self,
            root_copy_slot: usize,
            descendant_slot: usize,
        ) -> Result<(), EmbeddingMutationError> {
            let mut current_slot = descendant_slot;
            let step_limit = self.slots.len() + 1;

            self.slots[current_slot].visited = true;
            for _ in 0..step_limit {
                if current_slot == root_copy_slot {
                    return Ok(());
                }

                let parent_arc = self.parent_arc_in_embedding(current_slot).ok_or(
                    EmbeddingMutationError::MissingExternalFacePath {
                        start_slot: root_copy_slot,
                        start_side: 0,
                        end_slot: descendant_slot,
                    },
                )?;
                let parent_slot = self.arcs[parent_arc].target_slot;
                self.arcs[parent_arc].visited = true;
                let twin = self.arcs[parent_arc].twin;
                self.arcs[twin].visited = true;
                current_slot = parent_slot;
                self.slots[current_slot].visited = true;
            }

            Err(EmbeddingMutationError::MissingExternalFacePath {
                start_slot: root_copy_slot,
                start_side: 0,
                end_slot: descendant_slot,
            })
        }

        #[allow(dead_code)]
        fn collect_bicomp_slots(&self, root_copy_slot: usize) -> Vec<usize> {
            let mut stack = vec![root_copy_slot];
            let mut seen = vec![false; self.slots.len()];
            let mut slots = Vec::new();

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;
                slots.push(slot);

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    if !self.arc_stays_in_bicomp(root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }

            slots
        }

        #[allow(dead_code)]
        fn clear_bicomp_search_state(&mut self, root_copy_slot: usize) {
            let reset_value = self.primary_slot_by_original_vertex.len();
            for slot in self.collect_bicomp_slots(root_copy_slot) {
                self.slots[slot].visited = false;
                self.slots[slot].visited_info = reset_value;

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    self.arcs[arc].visited = false;
                    let twin = self.arcs[arc].twin;
                    self.arcs[twin].visited = false;
                }
            }
        }

        #[allow(dead_code)]
        fn k4_path_component_slots(
            &self,
            root_copy_slot: usize,
            prev_link: usize,
            active_slot: usize,
        ) -> Result<Vec<usize>, WalkDownExecutionError> {
            let mut slots = Vec::new();
            let mut slot_prev_link = prev_link;
            let mut slot = self.k4_ext_face_neighbor(root_copy_slot, &mut slot_prev_link);

            while slot != active_slot {
                slots.push(slot);
                slot = self.k4_ext_face_neighbor(slot, &mut slot_prev_link);
                if slot == root_copy_slot {
                    return Err(WalkDownExecutionError::InvalidK4Context);
                }
            }

            Ok(slots)
        }

        #[allow(dead_code)]
        fn test_k4_path_component_for_ancestor(
            &self,
            root_copy_slot: usize,
            prev_link: usize,
            active_slot: usize,
        ) -> bool {
            let mut slot_prev_link = prev_link;
            let mut slot = root_copy_slot;
            loop {
                slot = self.k4_ext_face_neighbor(slot, &mut slot_prev_link);
                if let EmbeddingSlotKind::Primary { .. } = self.slots[slot].kind {
                    if slot < active_slot {
                        return true;
                    }
                }
                if slot == active_slot {
                    return false;
                }
            }
        }

        #[allow(dead_code)]
        fn clear_visited_in_k4_path_component(&mut self, component_slots: &[usize]) {
            for &slot in component_slots {
                self.slots[slot].visited = false;
                let mut arc = self.slots[slot].first_arc;
                while let Some(current_arc) = arc {
                    arc = self.arcs[current_arc].next;
                    self.arcs[current_arc].visited = false;
                    let twin = self.arcs[current_arc].twin;
                    self.arcs[twin].visited = false;
                    let neighbor = self.arcs[current_arc].target_slot;
                    self.slots[neighbor].visited = false;
                }
            }
        }

        #[allow(dead_code)]
        fn delete_unmarked_edges_in_k4_path_component(
            &mut self,
            root_copy_slot: usize,
            active_slot: usize,
            component_slots: &[usize],
        ) {
            let mut arcs_to_unlink = Vec::new();

            for &slot in component_slots {
                let mut arc = self.slots[slot].first_arc;
                while let Some(current_arc) = arc {
                    arc = self.arcs[current_arc].next;
                    let neighbor = self.arcs[current_arc].target_slot;
                    if !self.arcs[current_arc].visited
                        && (current_arc < self.arcs[current_arc].twin
                            || neighbor == root_copy_slot
                            || neighbor == active_slot)
                    {
                        arcs_to_unlink.push(current_arc);
                    }
                }
            }

            arcs_to_unlink.sort_unstable();
            arcs_to_unlink.dedup();
            for arc in arcs_to_unlink {
                if self.arcs[arc].source_slot == usize::MAX {
                    continue;
                }
                self.delete_arc_pair_permanently(arc);
            }
        }

        #[allow(dead_code)]
        fn tree_path_slots_from_bicomp_root_to_descendant(
            &self,
            root_copy_slot: usize,
            descendant_slot: usize,
        ) -> Result<Vec<usize>, EmbeddingMutationError> {
            self.tree_path_slots_between_ancestor_and_descendant(root_copy_slot, descendant_slot)
        }

        #[allow(dead_code)]
        fn reduce_path_to_edge_by_endpoint_arcs(
            &mut self,
            start_slot: usize,
            mut start_arc: usize,
            end_slot: usize,
            mut end_arc: usize,
        ) -> Result<Option<usize>, EmbeddingMutationError> {
            let start_side = self
                .arc_side(start_slot, start_arc)
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: 0 })?;
            let end_side = self
                .arc_side(end_slot, end_arc)
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: 0 })?;

            if self.restore_reduced_path_edge(start_arc) {
                start_arc = self.slot_arc(start_slot, start_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: start_side },
                )?;
            }

            if self.restore_reduced_path_edge(end_arc) {
                end_arc = self.slot_arc(end_slot, end_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: end_side },
                )?;
            }

            if self.arcs[start_arc].target_slot == end_slot {
                self.slots[start_slot].ext_face[start_side] = Some(end_slot);
                self.slots[end_slot].ext_face[end_side] = Some(start_slot);
                if self.is_singleton_slot(start_slot) {
                    self.slots[start_slot].ext_face[1 ^ start_side] = Some(end_slot);
                }
                if self.is_singleton_slot(end_slot) {
                    self.slots[end_slot].ext_face[1 ^ end_side] = Some(start_slot);
                }
                return Ok(None);
            }

            let start_kind = self.arcs[start_arc].kind;
            let end_kind = self.arcs[end_arc].kind;
            let start_prev = self.arcs[start_arc].prev;
            let start_next = self.arcs[start_arc].next;
            let end_prev = self.arcs[end_arc].prev;
            let end_next = self.arcs[end_arc].next;

            self.unlink_arc_pair(start_arc);
            self.unlink_arc_pair(end_arc);

            let reduction_arc = self.push_synthetic_arc_pair(
                start_slot,
                end_slot,
                start_kind,
                end_kind,
                Some(start_arc),
                Some(end_arc),
            );
            let reduction_twin = self.arcs[reduction_arc].twin;

            self.insert_arc_at_position(start_slot, reduction_arc, start_prev, start_next);
            self.insert_arc_at_position(end_slot, reduction_twin, end_prev, end_next);

            self.slots[start_slot].ext_face[start_side] = Some(end_slot);
            self.slots[end_slot].ext_face[end_side] = Some(start_slot);
            if self.is_singleton_slot(start_slot) {
                self.slots[start_slot].ext_face[1 ^ start_side] = Some(end_slot);
            }
            if self.is_singleton_slot(end_slot) {
                self.slots[end_slot].ext_face[1 ^ end_side] = Some(start_slot);
            }

            Ok(Some(reduction_arc))
        }

        #[allow(dead_code)]
        fn slot_primary_dfi_for_reduced_edge(&self, slot: usize) -> usize {
            match self.slots[slot].kind {
                EmbeddingSlotKind::Primary { .. } => slot,
                EmbeddingSlotKind::RootCopy { parent_primary_slot, .. } => parent_primary_slot,
            }
        }

        #[allow(dead_code)]
        fn reduction_edge_kinds(
            &self,
            start_slot: usize,
            end_slot: usize,
            is_tree_edge: bool,
        ) -> (DfsArcType, DfsArcType) {
            let start_primary_slot = self.slot_primary_dfi_for_reduced_edge(start_slot);
            let end_primary_slot = self.slot_primary_dfi_for_reduced_edge(end_slot);

            if is_tree_edge {
                if start_primary_slot < end_primary_slot {
                    (DfsArcType::Child, DfsArcType::Parent)
                } else {
                    (DfsArcType::Parent, DfsArcType::Child)
                }
            } else if start_primary_slot < end_primary_slot {
                (DfsArcType::Forward, DfsArcType::Back)
            } else {
                (DfsArcType::Back, DfsArcType::Forward)
            }
        }

        fn reduce_path_to_edge_with_explicit_kinds(
            &mut self,
            start_slot: usize,
            mut start_arc: usize,
            end_slot: usize,
            mut end_arc: usize,
            start_kind: DfsArcType,
            end_kind: DfsArcType,
        ) -> Result<Option<usize>, EmbeddingMutationError> {
            let start_side = self
                .arc_side(start_slot, start_arc)
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: 0 })?;
            let end_side = self
                .arc_side(end_slot, end_arc)
                .ok_or(EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: 0 })?;

            if self.restore_reduced_path_edge(start_arc) {
                start_arc = self.slot_arc(start_slot, start_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: start_slot, side: start_side },
                )?;
            }

            if self.restore_reduced_path_edge(end_arc) {
                end_arc = self.slot_arc(end_slot, end_side).ok_or(
                    EmbeddingMutationError::MissingSlotArc { slot: end_slot, side: end_side },
                )?;
            }

            if self.arcs[start_arc].target_slot == end_slot {
                self.slots[start_slot].ext_face[start_side] = Some(end_slot);
                self.slots[end_slot].ext_face[end_side] = Some(start_slot);
                if self.is_singleton_slot(start_slot) {
                    self.slots[start_slot].ext_face[1 ^ start_side] = Some(end_slot);
                }
                if self.is_singleton_slot(end_slot) {
                    self.slots[end_slot].ext_face[1 ^ end_side] = Some(start_slot);
                }
                return Ok(None);
            }

            let start_prev = self.arcs[start_arc].prev;
            let start_next = self.arcs[start_arc].next;
            let end_prev = self.arcs[end_arc].prev;
            let end_next = self.arcs[end_arc].next;

            self.unlink_arc_pair(start_arc);
            self.unlink_arc_pair(end_arc);

            let reduction_arc = self.push_synthetic_arc_pair(
                start_slot,
                end_slot,
                start_kind,
                end_kind,
                Some(start_arc),
                Some(end_arc),
            );
            let reduction_twin = self.arcs[reduction_arc].twin;

            self.insert_arc_at_position(start_slot, reduction_arc, start_prev, start_next);
            self.insert_arc_at_position(end_slot, reduction_twin, end_prev, end_next);

            self.slots[start_slot].ext_face[start_side] = Some(end_slot);
            self.slots[end_slot].ext_face[end_side] = Some(start_slot);
            if self.is_singleton_slot(start_slot) {
                self.slots[start_slot].ext_face[1 ^ start_side] = Some(end_slot);
            }
            if self.is_singleton_slot(end_slot) {
                self.slots[end_slot].ext_face[1 ^ end_side] = Some(start_slot);
            }

            Ok(Some(reduction_arc))
        }

        fn cumulative_orientation_on_tree_path(
            &self,
            ancestor_slot: usize,
            descendant_slot: usize,
        ) -> Result<bool, EmbeddingMutationError> {
            let path_slots = self
                .tree_path_slots_between_ancestor_and_descendant(ancestor_slot, descendant_slot)?;
            let mut inverted = false;

            for slots in path_slots.windows(2) {
                if let Some(arc) = self.find_arc_to_target(slots[0], slots[1]) {
                    if self.arcs[arc].kind == DfsArcType::Child {
                        inverted ^= self.arcs[arc].inverted;
                    } else {
                        let twin = self.arcs[arc].twin;
                        if self.arcs[twin].kind == DfsArcType::Child {
                            inverted ^= self.arcs[twin].inverted;
                        }
                    }
                    continue;
                }

                if self.is_parent_primary_to_root_copy_step(slots[0], slots[1]) {
                    continue;
                }

                return Err(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: ancestor_slot,
                    start_side: 0,
                    end_slot: descendant_slot,
                });
            }

            Ok(inverted)
        }

        #[allow(dead_code)]
        fn reduce_k4_bicomp_to_edge(
            &mut self,
            root_copy_slot: usize,
            descendant_slot: usize,
        ) -> Result<(), EmbeddingMutationError> {
            self.orient_bicomp_from_root(root_copy_slot, false);
            self.clear_all_visited_flags_in_bicomp(root_copy_slot);
            self.mark_current_dfs_path_in_bicomp(root_copy_slot, descendant_slot)?;

            let mut arcs_to_unlink = Vec::new();
            for slot in self.collect_bicomp_slots(root_copy_slot) {
                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    if !self.arcs[arc].visited && arc < self.arcs[arc].twin {
                        arcs_to_unlink.push(arc);
                    }
                }
            }
            for arc in arcs_to_unlink {
                self.delete_arc_pair_permanently(arc);
            }

            let start_arc = self.slots[root_copy_slot].first_arc.ok_or(
                EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: root_copy_slot,
                    start_side: 0,
                    end_slot: descendant_slot,
                },
            )?;
            let end_arc = self.slots[descendant_slot].first_arc.ok_or(
                EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: root_copy_slot,
                    start_side: 0,
                    end_slot: descendant_slot,
                },
            )?;
            let _ = self.reduce_path_to_edge_with_explicit_kinds(
                root_copy_slot,
                start_arc,
                descendant_slot,
                end_arc,
                DfsArcType::Child,
                DfsArcType::Parent,
            )?;

            self.slots[descendant_slot].visited_info = self.primary_slot_by_original_vertex.len();
            Ok(())
        }

        #[allow(dead_code)]
        #[allow(clippy::too_many_lines, clippy::uninlined_format_args)]
        fn reduce_k4_path_component(
            &mut self,
            context: &K4Context,
            prev_link: usize,
            active_slot: usize,
        ) -> Result<(), WalkDownExecutionError> {
            let mut path_is_tree_edge = false;
            let mut cumulative_inverted = false;
            let start_side = 1 ^ prev_link;
            let start_arc = self.slot_arc(context.root_copy_slot, start_side).ok_or(
                WalkDownExecutionError::Mutation(EmbeddingMutationError::MissingSlotArc {
                    slot: context.root_copy_slot,
                    side: start_side,
                }),
            )?;
            if self.arcs[start_arc].target_slot == active_slot {
                return Ok(());
            }

            let component_slots =
                self.k4_path_component_slots(context.root_copy_slot, prev_link, active_slot)?;
            if self.test_k4_path_component_for_ancestor(
                context.root_copy_slot,
                prev_link,
                active_slot,
            ) {
                self.clear_visited_in_k4_path_component(&component_slots);
                path_is_tree_edge = true;
                cumulative_inverted = self
                    .cumulative_orientation_on_tree_path(context.root_copy_slot, active_slot)
                    .map_err(WalkDownExecutionError::Mutation)?;
                let path_slots = self
                    .tree_path_slots_from_bicomp_root_to_descendant(
                        context.root_copy_slot,
                        active_slot,
                    )
                    .map_err(WalkDownExecutionError::Mutation)?;
                self.mark_tree_path_slots_visited(&path_slots)
                    .map_err(WalkDownExecutionError::Mutation)?;
            } else {
                self.clear_visited_in_k4_path_component(&component_slots);

                let mut slot_prev_link = prev_link;
                let descendant_slot =
                    self.k4_ext_face_neighbor(context.root_copy_slot, &mut slot_prev_link);
                let back_arc = self
                    .find_arc_to_target(context.root_copy_slot, descendant_slot)
                    .ok_or(WalkDownExecutionError::InvalidK4Context)?;
                self.arcs[back_arc].visited = true;
                let back_twin = self.arcs[back_arc].twin;
                self.arcs[back_twin].visited = true;
                self.slots[context.root_copy_slot].visited = true;
                self.slots[descendant_slot].visited = true;

                let path_slots = self
                    .tree_path_slots_between_ancestor_and_descendant(active_slot, descendant_slot)
                    .map_err(WalkDownExecutionError::Mutation)?;
                self.mark_tree_path_slots_visited(&path_slots)
                    .map_err(WalkDownExecutionError::Mutation)?;
            }

            self.delete_unmarked_edges_in_k4_path_component(
                context.root_copy_slot,
                active_slot,
                &component_slots,
            );

            self.clear_visited_in_k4_path_component(&component_slots);
            self.slots[active_slot].visited_info = self.primary_slot_by_original_vertex.len();

            let end_side = self
                .external_face_entry_side(context.root_copy_slot, start_side, active_slot)
                .ok_or(WalkDownExecutionError::InvalidK4Context)?;
            let start_arc = self.slot_arc(context.root_copy_slot, start_side).ok_or(
                WalkDownExecutionError::Mutation(EmbeddingMutationError::MissingSlotArc {
                    slot: context.root_copy_slot,
                    side: start_side,
                }),
            )?;
            let end_arc =
                self.slot_arc(active_slot, end_side).ok_or(WalkDownExecutionError::Mutation(
                    EmbeddingMutationError::MissingSlotArc { slot: active_slot, side: end_side },
                ))?;
            let (start_kind, end_kind) = if path_is_tree_edge {
                (DfsArcType::Child, DfsArcType::Parent)
            } else {
                (DfsArcType::Forward, DfsArcType::Back)
            };
            let reduction_arc = self
                .reduce_path_to_edge_with_explicit_kinds(
                    context.root_copy_slot,
                    start_arc,
                    active_slot,
                    end_arc,
                    start_kind,
                    end_kind,
                )
                .map_err(WalkDownExecutionError::Mutation)?;
            if path_is_tree_edge {
                if let Some(reduction_arc) = reduction_arc {
                    if self.arcs[reduction_arc].kind == DfsArcType::Child {
                        self.arcs[reduction_arc].inverted = cumulative_inverted;
                    } else {
                        let reduction_twin = self.arcs[reduction_arc].twin;
                        if self.arcs[reduction_twin].kind == DfsArcType::Child {
                            self.arcs[reduction_twin].inverted = cumulative_inverted;
                        }
                    }
                }
            }

            self.rebuild_shortcut_ext_face_for_bicomp(context.root_copy_slot)?;

            Ok(())
        }

        #[allow(dead_code)]
        fn continue_after_k4_minor_a(
            &mut self,
            context: &K4Context,
        ) -> Result<(), EmbeddingMutationError> {
            self.reduce_k4_bicomp_to_edge(context.root_copy_slot, context.w_slot)?;
            Ok(())
        }

        #[allow(dead_code)]
        fn continue_after_k4_minor_b(
            &mut self,
            context: &K4Context,
            x_active_slot: usize,
            y_active_slot: usize,
        ) -> Result<(), WalkDownExecutionError> {
            if x_active_slot == y_active_slot {
                self.reduce_k4_bicomp_to_edge(context.root_copy_slot, x_active_slot)
                    .map_err(WalkDownExecutionError::Mutation)?;
            } else {
                self.reduce_k4_path_component(context, 1, x_active_slot)?;
                self.reduce_k4_path_component(context, 0, y_active_slot)?;
            }

            Ok(())
        }

        #[allow(dead_code)]
        #[allow(clippy::too_many_lines, clippy::uninlined_format_args)]
        pub(crate) fn search_for_k4_in_bicomp(
            &mut self,
            current_primary_slot: usize,
            _walk_root_copy_slot: usize,
            bicomp_root_copy_slot: usize,
        ) -> Result<K4BicompSearchOutcome, WalkDownExecutionError> {
            let (context, minor_type) =
                self.classify_k4_minor(current_primary_slot, bicomp_root_copy_slot)?;

            match minor_type {
                K4MinorType::A => {
                    self.orient_bicomp_from_root(context.root_copy_slot, true);
                    let low_active =
                        self.find_k4_second_active_vertex_on_low_ext_face_path(&context);
                    if low_active.is_some() {
                        return Ok(K4BicompSearchOutcome::MinorFound);
                    }
                    let has_xy_path = self.find_k4_xy_path(&context)?;
                    if has_xy_path {
                        return Ok(K4BicompSearchOutcome::MinorFound);
                    }

                    if let Err(error) = self.continue_after_k4_minor_a(&context) {
                        return Err(WalkDownExecutionError::Mutation(error));
                    }
                    Ok(K4BicompSearchOutcome::Continue)
                }
                K4MinorType::B => {
                    let x_active_slot = self
                        .find_k4_planarity_active_vertex(&context, 1)
                        .ok_or(WalkDownExecutionError::InvalidK4Context)?;
                    let y_active_slot = self
                        .find_k4_planarity_active_vertex(&context, 0)
                        .ok_or(WalkDownExecutionError::InvalidK4Context)?;

                    self.update_future_pertinent_child(x_active_slot, context.current_primary_slot);
                    self.update_future_pertinent_child(y_active_slot, context.current_primary_slot);

                    if x_active_slot != y_active_slot
                        && self.is_future_pertinent(x_active_slot, context.current_primary_slot)
                        && self.is_future_pertinent(y_active_slot, context.current_primary_slot)
                    {
                        return Ok(K4BicompSearchOutcome::MinorFound);
                    }

                    if x_active_slot == y_active_slot && !self.is_pertinent(x_active_slot) {
                        return Err(WalkDownExecutionError::InvalidK4Context);
                    }

                    let x_separator =
                        self.find_k4_separating_internal_edge(&context, 1, x_active_slot)?;
                    let y_separator =
                        self.find_k4_separating_internal_edge(&context, 0, y_active_slot)?;
                    if x_separator.is_some() || y_separator.is_some() {
                        return Ok(K4BicompSearchOutcome::MinorFound);
                    }

                    self.continue_after_k4_minor_b(&context, x_active_slot, y_active_slot)?;
                    Ok(K4BicompSearchOutcome::Continue)
                }
                K4MinorType::E => Ok(K4BicompSearchOutcome::MinorFound),
            }
        }

        fn handle_k4_blocked_bicomp(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            blocked_root_copy_slot: usize,
        ) -> Result<K4BlockedBicompOutcome, WalkDownExecutionError> {
            if blocked_root_copy_slot != walk_root_copy_slot {
                return match self.search_for_k4_in_bicomp(
                    current_primary_slot,
                    blocked_root_copy_slot,
                    blocked_root_copy_slot,
                )? {
                    K4BicompSearchOutcome::MinorFound => Ok(K4BlockedBicompOutcome::Found),
                    K4BicompSearchOutcome::Continue => Ok(K4BlockedBicompOutcome::ContinueWalkdown),
                };
            }

            if self.handling_k4_blocked_bicomp {
                self.k4_reblocked_same_root = true;
                return Ok(K4BlockedBicompOutcome::Completed);
            }

            self.handling_k4_blocked_bicomp = true;
            let outcome = loop {
                match self.search_for_k4_in_bicomp(
                    current_primary_slot,
                    walk_root_copy_slot,
                    walk_root_copy_slot,
                )? {
                    K4BicompSearchOutcome::MinorFound => break K4BlockedBicompOutcome::Found,
                    K4BicompSearchOutcome::Continue => {}
                }

                self.k4_reblocked_same_root = false;
                match self.walk_down_child(
                    preprocessing,
                    current_primary_slot,
                    walk_root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                )? {
                    WalkDownChildOutcome::K4Found => break K4BlockedBicompOutcome::Found,
                    WalkDownChildOutcome::Completed => {
                        if self.k4_reblocked_same_root {
                            continue;
                        }
                        break K4BlockedBicompOutcome::Completed;
                    }
                    WalkDownChildOutcome::K23Found | WalkDownChildOutcome::K33Found => {
                        unreachable!()
                    }
                }
            };
            self.handling_k4_blocked_bicomp = false;
            self.k4_reblocked_same_root = false;
            Ok(outcome)
        }

        fn orient_bicomp_from_root(&mut self, bicomp_root_copy_slot: usize, preserve_signs: bool) {
            let mut stack = vec![(bicomp_root_copy_slot, false)];
            let mut seen = vec![false; self.slots.len()];

            while let Some((slot, inverted_flag)) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;

                if slot != bicomp_root_copy_slot && inverted_flag {
                    self.invert_vertex(slot);
                }

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;

                    if !self.arcs[arc].embedded || self.arcs[arc].kind != DfsArcType::Child {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if seen[neighbor_slot] {
                        continue;
                    }

                    stack.push((neighbor_slot, inverted_flag ^ self.arcs[arc].inverted));
                    if !preserve_signs {
                        self.arcs[arc].inverted = false;
                    }
                }
            }
        }

        fn child_arc_enters_descendant_bicomp(
            &self,
            bicomp_root_copy_slot: usize,
            arc: usize,
        ) -> bool {
            if self.arcs[arc].kind != DfsArcType::Child {
                return false;
            }

            let child_primary_slot = self.arcs[arc].target_slot;
            if child_primary_slot >= self.root_copy_by_primary_dfi.len()
                || self.is_virtual(child_primary_slot)
            {
                return false;
            }
            let Some(descendant_root_copy_slot) = self.root_copy_by_primary_dfi[child_primary_slot]
            else {
                return false;
            };

            descendant_root_copy_slot != bicomp_root_copy_slot
                && self.slots[descendant_root_copy_slot].first_arc.is_some()
        }

        #[allow(dead_code)]
        fn arc_stays_in_bicomp(&self, bicomp_root_copy_slot: usize, arc: usize) -> bool {
            if !self.arcs[arc].embedded {
                return false;
            }

            if self.arcs[arc].kind == DfsArcType::Child {
                return !self.child_arc_enters_descendant_bicomp(bicomp_root_copy_slot, arc);
            }

            if self.arcs[arc].kind == DfsArcType::Parent {
                return !self.child_arc_enters_descendant_bicomp(
                    bicomp_root_copy_slot,
                    self.arcs[arc].twin,
                );
            }

            true
        }

        #[allow(dead_code)]
        fn orient_full_bicomp_from_root(
            &mut self,
            bicomp_root_copy_slot: usize,
            preserve_signs: bool,
        ) {
            let mut stack = vec![(bicomp_root_copy_slot, false)];
            let mut seen = vec![false; self.slots.len()];

            while let Some((slot, inverted_flag)) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;

                if slot != bicomp_root_copy_slot && inverted_flag {
                    self.invert_vertex(slot);
                }

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;

                    if !self.arc_stays_in_bicomp(bicomp_root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if seen[neighbor_slot] {
                        continue;
                    }

                    stack.push((neighbor_slot, inverted_flag ^ self.arcs[arc].inverted));
                    if !preserve_signs {
                        self.arcs[arc].inverted = false;
                    }
                }
            }
        }

        fn clear_all_visited_flags_in_bicomp(&mut self, bicomp_root_copy_slot: usize) {
            let mut stack = vec![bicomp_root_copy_slot];
            let mut seen = vec![false; self.slots.len()];

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;
                self.slots[slot].visited = false;

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    self.arcs[arc].visited = false;
                    let twin = self.arcs[arc].twin;
                    self.arcs[twin].visited = false;

                    if !self.arc_stays_in_bicomp(bicomp_root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }
        }

        #[allow(dead_code)]
        fn clear_all_visited_flags_in_full_bicomp(&mut self, bicomp_root_copy_slot: usize) {
            let mut stack = vec![bicomp_root_copy_slot];
            let mut seen = vec![false; self.slots.len()];

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;
                self.slots[slot].visited = false;

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    self.arcs[arc].visited = false;
                    let twin = self.arcs[arc].twin;
                    self.arcs[twin].visited = false;

                    if !self.arc_stays_in_bicomp(bicomp_root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }
        }

        fn fill_visited_info_in_bicomp(&mut self, bicomp_root_copy_slot: usize, value: usize) {
            let mut stack = vec![bicomp_root_copy_slot];
            let mut seen = vec![false; self.slots.len()];

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;
                self.slots[slot].visited_info = value;

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;

                    if !self.arc_stays_in_bicomp(bicomp_root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }
        }

        fn clear_inverted_flags_in_bicomp(&mut self, bicomp_root_copy_slot: usize) {
            let mut stack = vec![bicomp_root_copy_slot];
            let mut seen = vec![false; self.slots.len()];

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    self.arcs[arc].inverted = false;

                    if !self.arc_stays_in_bicomp(bicomp_root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }
        }

        fn mark_arc_pair_visited(&mut self, arc: usize) {
            self.arcs[arc].visited = true;
            let twin = self.arcs[arc].twin;
            self.arcs[twin].visited = true;
            self.slots[self.arcs[arc].source_slot].visited = true;
            self.slots[self.arcs[arc].target_slot].visited = true;
        }

        fn mark_tree_path_between_slots_visited(
            &mut self,
            first_slot: usize,
            second_slot: usize,
        ) -> Result<(), EmbeddingMutationError> {
            if let Ok(path_slots) =
                self.tree_path_slots_between_ancestor_and_descendant(first_slot, second_slot)
            {
                return self.mark_tree_path_slots_visited(&path_slots);
            }

            let path_slots =
                self.tree_path_slots_between_ancestor_and_descendant(second_slot, first_slot)?;
            self.mark_tree_path_slots_visited(&path_slots)
        }

        fn first_visited_arc_on_slot(&self, slot: usize) -> Option<usize> {
            let first_arc = self.slots[slot].first_arc?;
            let mut arc = first_arc;
            loop {
                if self.arcs[arc].visited {
                    return Some(arc);
                }
                arc = self.next_arc_circular(slot, arc);
                if arc == first_arc {
                    return None;
                }
            }
        }

        fn last_visited_arc_on_slot(&self, slot: usize) -> Option<usize> {
            let last_arc = self.slots[slot].last_arc?;
            let mut arc = last_arc;
            loop {
                if self.arcs[arc].visited {
                    return Some(arc);
                }
                arc = self.prev_arc_circular(slot, arc);
                if arc == last_arc {
                    return None;
                }
            }
        }

        fn delete_unmarked_edges_in_bicomp(&mut self, bicomp_root_copy_slot: usize) {
            let mut arcs_to_unlink = Vec::new();

            for slot in self.collect_full_bicomp_slots(bicomp_root_copy_slot) {
                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;
                    if !self.arcs[arc].visited && arc < self.arcs[arc].twin {
                        arcs_to_unlink.push(arc);
                    }
                }
            }

            arcs_to_unlink.sort_unstable();
            arcs_to_unlink.dedup();
            for arc in arcs_to_unlink {
                self.delete_arc_pair_permanently(arc);
            }
        }

        #[allow(clippy::too_many_lines)]
        fn reduce_k33_minor_e_bicomp(
            &mut self,
            context: &K33MinorEContext,
        ) -> Result<(), EmbeddingMutationError> {
            self.orient_bicomp_from_root(context.root_copy_slot, false);

            let min_slot =
                core::cmp::min(context.x_slot, core::cmp::min(context.y_slot, context.w_slot));
            let max_slot =
                core::cmp::max(context.x_slot, core::cmp::max(context.y_slot, context.w_slot));

            let mut root_to_x_is_tree = true;
            let mut x_to_w_is_tree = true;
            let mut w_to_y_is_tree = true;
            let mut y_to_root_is_tree = true;
            let mut cross_path_is_tree = true;

            let (a_arc, a_slot, b_arc, b_slot) = if min_slot == context.x_slot {
                let a_arc = self.slots[context.root_copy_slot].last_arc.ok_or(
                    EmbeddingMutationError::MissingSlotArc {
                        slot: context.root_copy_slot,
                        side: 1,
                    },
                )?;
                let a_slot = self.arcs[a_arc].target_slot;
                y_to_root_is_tree = false;

                if max_slot == context.y_slot {
                    let b_arc = self.last_visited_arc_on_slot(context.x_slot).ok_or(
                        EmbeddingMutationError::MissingExternalFacePath {
                            start_slot: context.x_slot,
                            start_side: 1,
                            end_slot: context.y_slot,
                        },
                    )?;
                    let b_slot = self.arcs[b_arc].target_slot;
                    cross_path_is_tree = false;
                    (a_arc, a_slot, b_arc, b_slot)
                } else if max_slot == context.w_slot {
                    let b_arc = self.slots[context.x_slot].first_arc.ok_or(
                        EmbeddingMutationError::MissingSlotArc { slot: context.x_slot, side: 0 },
                    )?;
                    let b_slot = self.arcs[b_arc].target_slot;
                    x_to_w_is_tree = false;
                    (a_arc, a_slot, b_arc, b_slot)
                } else {
                    return Err(EmbeddingMutationError::MissingExternalFacePath {
                        start_slot: context.root_copy_slot,
                        start_side: 0,
                        end_slot: max_slot,
                    });
                }
            } else {
                let a_arc = self.slots[context.root_copy_slot].first_arc.ok_or(
                    EmbeddingMutationError::MissingSlotArc {
                        slot: context.root_copy_slot,
                        side: 0,
                    },
                )?;
                let a_slot = self.arcs[a_arc].target_slot;
                root_to_x_is_tree = false;

                if max_slot == context.x_slot {
                    let b_arc = self.first_visited_arc_on_slot(context.y_slot).ok_or(
                        EmbeddingMutationError::MissingExternalFacePath {
                            start_slot: context.y_slot,
                            start_side: 0,
                            end_slot: context.x_slot,
                        },
                    )?;
                    let b_slot = self.arcs[b_arc].target_slot;
                    cross_path_is_tree = false;
                    (a_arc, a_slot, b_arc, b_slot)
                } else if max_slot == context.w_slot {
                    let b_arc = self.slots[context.y_slot].last_arc.ok_or(
                        EmbeddingMutationError::MissingSlotArc { slot: context.y_slot, side: 1 },
                    )?;
                    let b_slot = self.arcs[b_arc].target_slot;
                    w_to_y_is_tree = false;
                    (a_arc, a_slot, b_arc, b_slot)
                } else {
                    return Err(EmbeddingMutationError::MissingExternalFacePath {
                        start_slot: context.root_copy_slot,
                        start_side: 1,
                        end_slot: max_slot,
                    });
                }
            };

            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            self.mark_tree_path_between_slots_visited(context.root_copy_slot, max_slot)?;
            self.mark_tree_path_between_slots_visited(
                if min_slot == context.x_slot { context.y_slot } else { context.x_slot },
                a_slot,
            )?;
            self.mark_arc_pair_visited(a_arc);
            self.mark_tree_path_between_slots_visited(max_slot, b_slot)?;
            self.mark_arc_pair_visited(b_arc);

            self.delete_unmarked_edges_in_bicomp(context.root_copy_slot);
            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            self.clear_inverted_flags_in_bicomp(context.root_copy_slot);

            let rx_end_side = self
                .external_face_entry_side(context.root_copy_slot, 0, context.x_slot)
                .ok_or(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: context.root_copy_slot,
                    start_side: 0,
                    end_slot: context.x_slot,
                })?;
            let xw_end_side = self
                .external_face_entry_side(context.x_slot, 1 ^ rx_end_side, context.w_slot)
                .ok_or(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: context.x_slot,
                    start_side: 1 ^ rx_end_side,
                    end_slot: context.w_slot,
                })?;
            let wy_end_side = self
                .external_face_entry_side(context.w_slot, 1 ^ xw_end_side, context.y_slot)
                .ok_or(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: context.w_slot,
                    start_side: 1 ^ xw_end_side,
                    end_slot: context.y_slot,
                })?;
            let yr_end_side = self
                .external_face_entry_side(context.y_slot, 1 ^ wy_end_side, context.root_copy_slot)
                .ok_or(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: context.y_slot,
                    start_side: 1 ^ wy_end_side,
                    end_slot: context.root_copy_slot,
                })?;

            let (rx_start_kind, rx_end_kind) = self.reduction_edge_kinds(
                context.root_copy_slot,
                context.x_slot,
                root_to_x_is_tree,
            );
            let (xw_start_kind, xw_end_kind) =
                self.reduction_edge_kinds(context.x_slot, context.w_slot, x_to_w_is_tree);
            let (wy_start_kind, wy_end_kind) =
                self.reduction_edge_kinds(context.w_slot, context.y_slot, w_to_y_is_tree);
            let (yr_start_kind, yr_end_kind) = self.reduction_edge_kinds(
                context.y_slot,
                context.root_copy_slot,
                y_to_root_is_tree,
            );
            let (cross_path_start_kind, cross_path_end_kind) =
                self.reduction_edge_kinds(context.x_slot, context.y_slot, cross_path_is_tree);

            let _ = self.reduce_external_face_path_to_edge(
                context.root_copy_slot,
                0,
                context.x_slot,
                rx_end_side,
                rx_start_kind,
                rx_end_kind,
            )?;
            let _ = self.reduce_external_face_path_to_edge(
                context.x_slot,
                1 ^ rx_end_side,
                context.w_slot,
                xw_end_side,
                xw_start_kind,
                xw_end_kind,
            )?;
            let _ = self.reduce_external_face_path_to_edge(
                context.w_slot,
                1 ^ xw_end_side,
                context.y_slot,
                wy_end_side,
                wy_start_kind,
                wy_end_kind,
            )?;
            let _ = self.reduce_external_face_path_to_edge(
                context.y_slot,
                1 ^ wy_end_side,
                context.root_copy_slot,
                yr_end_side,
                yr_start_kind,
                yr_end_kind,
            )?;
            let _ = self.reduce_xy_path_to_edge_with_explicit_kinds(
                context.x_slot,
                context.y_slot,
                cross_path_start_kind,
                cross_path_end_kind,
            )?;

            Ok(())
        }

        #[allow(dead_code)]
        fn collect_full_bicomp_slots(&self, root_copy_slot: usize) -> Vec<usize> {
            let mut stack = vec![root_copy_slot];
            let mut seen = vec![false; self.slots.len()];
            let mut slots = Vec::new();

            while let Some(slot) = stack.pop() {
                if seen[slot] {
                    continue;
                }
                seen[slot] = true;
                slots.push(slot);

                let mut current_arc = self.slots[slot].first_arc;
                while let Some(arc) = current_arc {
                    current_arc = self.arcs[arc].next;

                    if !self.arc_stays_in_bicomp(root_copy_slot, arc) {
                        continue;
                    }

                    let neighbor_slot = self.arcs[arc].target_slot;
                    if !seen[neighbor_slot] {
                        stack.push(neighbor_slot);
                    }
                }
            }

            slots
        }

        fn root_copy_child_arc(&self, root_copy_slot: usize) -> Option<usize> {
            let mut current_arc = self.slots[root_copy_slot].first_arc;
            while let Some(arc) = current_arc {
                if self.arcs[arc].kind == DfsArcType::Child {
                    return Some(arc);
                }
                current_arc = self.arcs[arc].next;
            }

            None
        }

        #[allow(dead_code)]
        fn find_nonplanarity_descendant_bicomp_root(
            &mut self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
            active_sides: &[usize],
        ) -> Result<usize, K33ContextInitFailure> {
            let mut descendant_root_candidate = None;
            for &root_side in active_sides {
                let trace = self.walk_down_trace(
                    current_primary_slot,
                    bicomp_root_copy_slot,
                    root_side,
                    false,
                );
                if let WalkDownOutcome::BlockedBicomp { root_copy_slot } = trace.outcome {
                    return Ok(root_copy_slot);
                }
                if let Some(root_copy_slot) = trace.frames.last().map(|frame| frame.root_copy_slot)
                {
                    descendant_root_candidate.get_or_insert(root_copy_slot);
                }
            }

            descendant_root_candidate.ok_or(K33ContextInitFailure::NoDescendantBicompRoot)
        }

        #[allow(dead_code)]
        fn initialize_nonplanarity_context(
            &mut self,
            _preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            _walk_root_copy_slot: usize,
            bicomp_root_copy_slot: usize,
            stack_root_copy_slot: Option<usize>,
        ) -> Result<NonplanarityContext, K33ContextInitFailure> {
            let root_copy_slot = stack_root_copy_slot.unwrap_or(bicomp_root_copy_slot);

            self.orient_bicomp_from_root(root_copy_slot, true);
            self.clear_all_visited_flags_in_bicomp(root_copy_slot);
            let active_vertices =
                self.find_active_vertices(root_copy_slot, current_primary_slot)?;
            let x_slot = active_vertices.0;
            let y_slot = active_vertices.1;
            let x_prev_link = active_vertices.2;
            let y_prev_link = active_vertices.3;
            let Some(w_slot) = self.find_pertinent_vertex_between_active_sides(x_slot, y_slot)
            else {
                return Err(K33ContextInitFailure::NoPertinentBetweenActiveSides);
            };

            Ok(NonplanarityContext {
                current_primary_slot,
                root_copy_slot,
                x_slot,
                y_slot,
                w_slot,
                x_prev_link,
                y_prev_link,
            })
        }

        fn prev_arc_circular(&self, slot: usize, arc: usize) -> usize {
            self.arcs[arc].prev.unwrap_or_else(|| {
                self.slots[slot]
                    .last_arc
                    .expect("nonempty slot must have a last arc for circular predecessor lookup")
            })
        }

        fn next_arc_circular(&self, slot: usize, arc: usize) -> usize {
            self.arcs[arc].next.unwrap_or_else(|| {
                self.slots[slot]
                    .first_arc
                    .expect("nonempty slot must have a first arc for circular successor lookup")
            })
        }

        fn hide_internal_edges_at_slot(&mut self, target_slot: usize) -> Vec<HiddenArcPair> {
            let Some(ext_face_arc_a) = self.slot_arc(target_slot, 0) else {
                return Vec::new();
            };
            let Some(ext_face_arc_b) = self.slot_arc(target_slot, 1) else {
                return Vec::new();
            };

            if ext_face_arc_a == ext_face_arc_b {
                return Vec::new();
            }

            let mut hidden_arcs = Vec::new();
            let mut current_arc = self.slots[target_slot].first_arc;
            while let Some(arc) = current_arc {
                if arc != ext_face_arc_a && arc != ext_face_arc_b {
                    hidden_arcs.push(arc);
                }
                current_arc = self.arcs[arc].next;
            }

            let mut hidden_pairs = Vec::with_capacity(hidden_arcs.len());
            for arc in hidden_arcs {
                let twin = self.arcs[arc].twin;
                hidden_pairs.push(HiddenArcPair {
                    arc,
                    arc_prev: self.arcs[arc].prev,
                    arc_next: self.arcs[arc].next,
                    twin_prev: self.arcs[twin].prev,
                    twin_next: self.arcs[twin].next,
                });
                self.unlink_arc_pair(arc);
            }

            hidden_pairs
        }

        fn restore_hidden_arc_pairs(&mut self, hidden_pairs: Vec<HiddenArcPair>) {
            for pair in hidden_pairs.into_iter().rev() {
                let twin = self.arcs[pair.arc].twin;
                let source_slot = self.arcs[pair.arc].source_slot;
                let target_slot = self.arcs[twin].source_slot;
                self.insert_arc_at_position(source_slot, pair.arc, pair.arc_prev, pair.arc_next);
                self.insert_arc_at_position(target_slot, twin, pair.twin_prev, pair.twin_next);
            }
        }

        fn least_ancestor_connection(
            &self,
            slot: usize,
            _current_primary_slot: usize,
        ) -> Option<usize> {
            match self.slots[slot].kind {
                EmbeddingSlotKind::Primary { .. } => {
                    let mut ancestor = self.least_ancestor_by_primary_slot[slot];
                    for &child_slot in &self.slots[slot].separated_dfs_children {
                        if self.is_separated_dfs_child(child_slot) {
                            ancestor = ancestor.min(self.lowpoint_by_primary_slot[child_slot]);
                        }
                    }
                    Some(ancestor)
                }
                EmbeddingSlotKind::RootCopy { .. } => None,
            }
        }

        fn parent_primary_slot(&self, primary_slot: usize) -> Option<usize> {
            let root_copy_slot = self.root_copy_by_primary_dfi[primary_slot]?;
            match self.slots[root_copy_slot].kind {
                EmbeddingSlotKind::RootCopy { parent_primary_slot, .. } => {
                    Some(parent_primary_slot)
                }
                EmbeddingSlotKind::Primary { .. } => None,
            }
        }

        fn set_k33_obstruction_marks(
            &self,
            context: &NonplanarityContext,
        ) -> Vec<K33ObstructionMark> {
            let mut marks = vec![K33ObstructionMark::Unmarked; self.slots.len()];

            let mut slot = context.root_copy_slot;
            let mut previous_link = 1usize;
            let mut mark = K33ObstructionMark::HighRxw;
            loop {
                slot = self.real_ext_face_neighbor(slot, &mut previous_link);
                if slot == context.w_slot {
                    break;
                }
                if slot == context.x_slot {
                    mark = K33ObstructionMark::LowRxw;
                }
                marks[slot] = mark;
            }

            slot = context.root_copy_slot;
            previous_link = 0usize;
            mark = K33ObstructionMark::HighRyw;
            loop {
                slot = self.real_ext_face_neighbor(slot, &mut previous_link);
                if slot == context.w_slot {
                    break;
                }
                if slot == context.y_slot {
                    mark = K33ObstructionMark::LowRyw;
                }
                marks[slot] = mark;
            }

            marks
        }

        fn clear_marked_path(&mut self, stack: &mut Vec<(usize, usize)>) {
            for (slot, entry_arc) in stack.drain(..) {
                let twin = self.arcs[entry_arc].twin;
                self.slots[slot].visited = false;
                self.arcs[entry_arc].visited = false;
                self.arcs[twin].visited = false;
            }
        }

        fn pop_marked_path_until_slot(
            &mut self,
            stack: &mut Vec<(usize, usize)>,
            target_slot: usize,
        ) {
            while let Some(&(slot, entry_arc)) = stack.last() {
                if slot == target_slot {
                    break;
                }
                stack.pop();
                let twin = self.arcs[entry_arc].twin;
                self.slots[slot].visited = false;
                self.arcs[entry_arc].visited = false;
                self.arcs[twin].visited = false;
            }
        }

        #[allow(clippy::too_many_lines, clippy::uninlined_format_args)]
        fn mark_closest_xy_path(
            &mut self,
            context: &NonplanarityContext,
            obstruction_marks: &[K33ObstructionMark],
            target_slot: usize,
        ) -> Result<Option<K33MarkedPath>, WalkDownExecutionError> {
            #[cfg(test)]
            let mut traversal_trace = Vec::new();
            let mut marked_path = K33MarkedPath { px_slot: usize::MAX, py_slot: usize::MAX };
            let mut stack = Vec::new();
            let target_is_root = target_slot == context.root_copy_slot;
            let antipodal_slot = if target_is_root {
                context.w_slot
            } else if target_slot == context.w_slot {
                context.root_copy_slot
            } else {
                return Err(WalkDownExecutionError::InvalidK33Context);
            };
            let hidden_pairs = self.hide_internal_edges_at_slot(target_slot);

            let mut current_slot = target_slot;
            let mut entry_arc = self
                .slot_arc(target_slot, usize::from(target_is_root))
                .ok_or(WalkDownExecutionError::InvalidK33Context)?;

            loop {
                let exit_arc = if current_slot == target_slot {
                    if target_is_root {
                        self.prev_arc_circular(current_slot, entry_arc)
                    } else {
                        self.next_arc_circular(current_slot, entry_arc)
                    }
                } else if target_is_root {
                    self.prev_arc_circular(current_slot, entry_arc)
                } else {
                    self.next_arc_circular(current_slot, entry_arc)
                };
                let next_slot = self.arcs[exit_arc].target_slot;
                let next_entry_arc = self.arcs[exit_arc].twin;
                #[cfg(test)]
                {
                    let prev_candidate = if current_slot == target_slot {
                        None
                    } else {
                        Some(self.prev_arc_circular(current_slot, entry_arc))
                    };
                    let next_candidate = if current_slot == target_slot {
                        None
                    } else {
                        Some(self.next_arc_circular(current_slot, entry_arc))
                    };
                    traversal_trace.push((
                        current_slot,
                        entry_arc,
                        exit_arc,
                        next_slot,
                        obstruction_marks[next_slot],
                        prev_candidate.map(|arc| self.arcs[arc].target_slot),
                        next_candidate.map(|arc| self.arcs[arc].target_slot),
                    ));
                }

                if self.slots[next_slot].visited {
                    self.pop_marked_path_until_slot(&mut stack, next_slot);
                } else {
                    if next_slot == antipodal_slot {
                        self.clear_marked_path(&mut stack);
                        self.restore_hidden_arc_pairs(hidden_pairs);
                        return Ok(None);
                    }

                    match obstruction_marks[next_slot] {
                        K33ObstructionMark::HighRxw | K33ObstructionMark::LowRxw => {
                            marked_path.px_slot = next_slot;
                            self.clear_marked_path(&mut stack);
                        }
                        _ => {}
                    }

                    stack.push((next_slot, next_entry_arc));
                    self.slots[next_slot].visited = true;
                    if next_slot != marked_path.px_slot {
                        let twin = self.arcs[next_entry_arc].twin;
                        self.arcs[next_entry_arc].visited = true;
                        self.arcs[twin].visited = true;
                    }

                    if matches!(
                        obstruction_marks[next_slot],
                        K33ObstructionMark::HighRyw | K33ObstructionMark::LowRyw
                    ) {
                        marked_path.py_slot = next_slot;
                        self.restore_hidden_arc_pairs(hidden_pairs);
                        return Ok(Some(marked_path));
                    }
                }

                current_slot = next_slot;
                entry_arc = next_entry_arc;
            }
        }

        fn mark_z_to_r_path(
            &mut self,
            context: &NonplanarityContext,
            obstruction_marks: &[K33ObstructionMark],
            marked_path: &mut K33MarkedPath,
        ) -> Result<Option<usize>, K33ZToRPathFailure> {
            let mut candidate_arc =
                self.slots[marked_path.px_slot].last_arc.ok_or(K33ZToRPathFailure::Structural)?;
            let first_arc =
                self.slots[marked_path.px_slot].first_arc.ok_or(K33ZToRPathFailure::Structural)?;

            while candidate_arc != first_arc && !self.arcs[candidate_arc].visited {
                candidate_arc = self.prev_arc_circular(marked_path.px_slot, candidate_arc);
            }
            if !self.arcs[candidate_arc].visited {
                return Err(K33ZToRPathFailure::Structural);
            }

            let mut next_arc = candidate_arc;
            while self.arcs[next_arc].visited {
                let prev_arc = self.arcs[next_arc].twin;
                let slot = self.arcs[prev_arc].source_slot;
                next_arc = self.prev_arc_circular(slot, prev_arc);
            }

            let mut prev_arc = self.arcs[next_arc].twin;
            let mut slot = self.arcs[next_arc].source_slot;
            if slot == marked_path.py_slot {
                return Ok(None);
            }
            let z_slot = slot;

            while slot != context.root_copy_slot {
                if obstruction_marks[slot] != K33ObstructionMark::Unmarked {
                    return Err(K33ZToRPathFailure::MarkedSlot {
                        slot,
                        mark: obstruction_marks[slot],
                    });
                }

                let next_slot = self.arcs[next_arc].target_slot;
                self.arcs[next_arc].visited = true;
                self.arcs[prev_arc].visited = true;
                self.slots[next_slot].visited = true;

                next_arc = self.prev_arc_circular(next_slot, prev_arc);
                prev_arc = self.arcs[next_arc].twin;
                slot = next_slot;
            }

            Ok(Some(z_slot))
        }

        fn find_future_pertinence_below_xy_path(
            &mut self,
            context: &NonplanarityContext,
            marked_path: &K33MarkedPath,
        ) -> Option<usize> {
            let mut slot = marked_path.px_slot;
            let mut previous_link = 1usize;
            slot = self.real_ext_face_neighbor(slot, &mut previous_link);

            while slot != marked_path.py_slot {
                self.update_future_pertinent_child(slot, context.current_primary_slot);
                if self.is_future_pertinent(slot, context.current_primary_slot) {
                    return Some(slot);
                }
                slot = self.real_ext_face_neighbor(slot, &mut previous_link);
            }

            None
        }

        fn search_for_k33_minor_e1(
            &mut self,
            current_primary_slot: usize,
            w_slot: usize,
            path_start_slot: usize,
            path_end_slot: usize,
        ) -> bool {
            let mut slot = path_start_slot;
            let mut previous_link = 1usize;
            slot = self.real_ext_face_neighbor(slot, &mut previous_link);

            while slot != path_end_slot {
                if slot != w_slot {
                    self.update_future_pertinent_child(slot, current_primary_slot);
                    if self.is_future_pertinent(slot, current_primary_slot)
                        || self.is_pertinent(slot)
                    {
                        return true;
                    }
                }

                slot = self.real_ext_face_neighbor(slot, &mut previous_link);
            }

            false
        }

        fn test_for_z_to_w_path(
            &mut self,
            w_slot: usize,
            obstruction_marks: &[K33ObstructionMark],
            _marked_path: &K33MarkedPath,
        ) -> bool {
            let mut stack: Vec<(usize, Option<usize>)> = vec![(w_slot, None)];
            let mut found_path = false;

            while let Some((slot, entry_arc)) = stack.pop() {
                let mut arc = if let Some(entry_arc) = entry_arc {
                    self.arcs[entry_arc].next
                } else {
                    if self.slots[slot].visited {
                        found_path = true;
                        break;
                    }

                    self.slots[slot].visited_info = usize::MAX;
                    self.slots[slot].first_arc
                };

                while let Some(current_arc) = arc {
                    let neighbor = self.arcs[current_arc].target_slot;

                    if !self.is_virtual(neighbor)
                        && self.slots[neighbor].visited_info != usize::MAX
                        && obstruction_marks[neighbor] == K33ObstructionMark::Unmarked
                    {
                        stack.push((slot, Some(current_arc)));
                        stack.push((neighbor, None));
                        break;
                    }

                    arc = self.arcs[current_arc].next;
                }
            }

            if found_path {
                while let Some((slot, entry_arc)) = stack.pop() {
                    self.slots[slot].visited = true;
                    if let Some(entry_arc) = entry_arc {
                        let twin = self.arcs[entry_arc].twin;
                        self.arcs[entry_arc].visited = true;
                        self.arcs[twin].visited = true;
                    }
                }
            }

            found_path
        }

        fn find_descendant_with_least_ancestor_below(
            &self,
            subtree_root_slot: usize,
            u_max: usize,
        ) -> Option<usize> {
            let mut stack = vec![subtree_root_slot];

            while let Some(slot) = stack.pop() {
                if self.least_ancestor_by_primary_slot[slot] < u_max {
                    return Some(slot);
                }
                for &child_slot in self.slots[slot].sorted_dfs_children.iter().rev() {
                    stack.push(child_slot);
                }
            }

            None
        }

        fn test_for_straddling_bridge(
            &self,
            current_primary_slot: usize,
            root_copy_slot: usize,
            u_max: usize,
        ) -> Option<usize> {
            let mut p_slot = current_primary_slot;
            let mut excluded_child = self.dfs_child_from_root(root_copy_slot);

            while p_slot > u_max {
                if self.least_ancestor_by_primary_slot[p_slot] < u_max {
                    return Some(p_slot);
                }

                let mut best_child = None;
                let mut best_lowpoint = usize::MAX;
                for &child_slot in &self.slots[p_slot].separated_dfs_children {
                    if child_slot == excluded_child || !self.is_separated_dfs_child(child_slot) {
                        continue;
                    }
                    let lowpoint = self.lowpoint_by_primary_slot[child_slot];
                    if lowpoint < u_max && lowpoint < best_lowpoint {
                        best_lowpoint = lowpoint;
                        best_child = Some(child_slot);
                    }
                }

                if let Some(best_child) = best_child {
                    if let Some(descendant_slot) =
                        self.find_descendant_with_least_ancestor_below(best_child, u_max)
                    {
                        return Some(descendant_slot);
                    }
                }

                excluded_child = p_slot;
                p_slot = self.parent_primary_slot(p_slot)?;
            }

            None
        }

        pub(crate) fn run_extra_k33_tests(
            &mut self,
            context: &K33MinorEContext,
        ) -> Result<Option<K33ExtraTestOutcome>, WalkDownExecutionError> {
            if self.search_for_k33_minor_e1(
                context.current_primary_slot,
                context.w_slot,
                context.px_slot,
                context.py_slot,
            ) {
                return Ok(Some(K33ExtraTestOutcome::E1));
            }

            let nonplanarity_context = NonplanarityContext {
                current_primary_slot: context.current_primary_slot,
                root_copy_slot: context.root_copy_slot,
                x_slot: context.x_slot,
                y_slot: context.y_slot,
                w_slot: context.w_slot,
                x_prev_link: 1,
                y_prev_link: 0,
            };
            let obstruction_marks = &context.obstruction_marks;
            let u_max = core::cmp::max(context.ux, core::cmp::max(context.uy, context.uz));

            self.record_k33_merge_blocker(context.x_slot, u_max);
            self.record_k33_merge_blocker(context.y_slot, u_max);

            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            if let Some(lowest_xy_path) =
                self.mark_closest_xy_path(&nonplanarity_context, obstruction_marks, context.w_slot)?
            {
                if lowest_xy_path.px_slot != context.x_slot
                    || lowest_xy_path.py_slot != context.y_slot
                {
                    return Ok(Some(K33ExtraTestOutcome::E4Like));
                }

                if self.test_for_z_to_w_path(context.w_slot, obstruction_marks, &lowest_xy_path) {
                    return Ok(Some(K33ExtraTestOutcome::E5));
                }
            }

            if context.uz < u_max
                && self
                    .test_for_straddling_bridge(
                        context.current_primary_slot,
                        context.root_copy_slot,
                        u_max,
                    )
                    .is_some()
            {
                return Ok(Some(K33ExtraTestOutcome::E6));
            }

            if (context.ux < u_max || context.uy < u_max)
                && self
                    .test_for_straddling_bridge(
                        context.current_primary_slot,
                        context.root_copy_slot,
                        u_max,
                    )
                    .is_some()
            {
                return Ok(Some(K33ExtraTestOutcome::E7));
            }
            Ok(None)
        }

        fn record_k33_merge_blocker(&mut self, slot: usize, u_max: usize) {
            self.slots[slot].k33_merge_blocker = Some(u_max);
        }

        fn find_k33_merge_blocker(
            &self,
            current_primary_slot: usize,
            current_slot: usize,
            current_root_copy_slot: usize,
            frames: &[WalkDownFrame],
        ) -> Option<(usize, usize, usize)> {
            if frames.is_empty() {
                return None;
            }

            self.slots[current_slot]
                .k33_merge_blocker
                .filter(|&u_max| current_primary_slot > u_max)
                .map(|u_max| (current_slot, u_max, current_root_copy_slot))
                .or_else(|| {
                    frames.iter().rev().find_map(|frame| {
                        self.slots[frame.cut_vertex_slot]
                            .k33_merge_blocker
                            .filter(|&u_max| current_primary_slot > u_max)
                            .map(|u_max| (frame.cut_vertex_slot, u_max, frame.root_copy_slot))
                    })
                })
        }

        fn adjacent_unembedded_ancestor_in_range(
            &self,
            preprocessing: &DfsPreprocessing,
            slot: usize,
            closer_ancestor: usize,
            farther_ancestor: usize,
        ) -> Option<usize> {
            let original_vertex = match self.slots[slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => return None,
            };

            preprocessing.adjacency_arcs[original_vertex].iter().copied().find_map(|arc| {
                let record = &self.arcs[arc];
                (!record.embedded
                    && preprocessing.arcs[arc].kind == DfsArcType::Back
                    && record.target_slot < closer_ancestor
                    && record.target_slot > farther_ancestor)
                    .then_some(record.target_slot)
            })
        }

        fn find_descendant_external_connection_ancestor(
            &self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            cut_vertex_slot: usize,
            u_max: usize,
        ) -> Option<usize> {
            if let Some(ancestor_slot) = self.adjacent_unembedded_ancestor_in_range(
                preprocessing,
                cut_vertex_slot,
                current_primary_slot,
                u_max,
            ) {
                return Some(ancestor_slot);
            }

            let mut stack = Vec::new();
            for &child_slot in &self.slots[cut_vertex_slot].sorted_dfs_children {
                if self.lowpoint_by_primary_slot[child_slot] < current_primary_slot
                    && self.is_separated_dfs_child(child_slot)
                {
                    stack.push(child_slot);
                }
            }

            while let Some(descendant_slot) = stack.pop() {
                if self.lowpoint_by_primary_slot[descendant_slot] >= current_primary_slot {
                    continue;
                }

                if let Some(ancestor_slot) = self.adjacent_unembedded_ancestor_in_range(
                    preprocessing,
                    descendant_slot,
                    current_primary_slot,
                    u_max,
                ) {
                    return Some(ancestor_slot);
                }

                for &child_slot in &self.slots[descendant_slot].sorted_dfs_children {
                    if self.lowpoint_by_primary_slot[child_slot] < current_primary_slot {
                        stack.push(child_slot);
                    }
                }
            }

            None
        }

        fn probe_k33_merge_blocker(
            &self,
            preprocessing: &DfsPreprocessing,
            merge_blocker_slot: usize,
            u_max: usize,
            root_copy_slot: usize,
        ) -> Result<bool, WalkDownExecutionError> {
            let mut probe = self.clone();
            match probe.find_k33_with_merge_blocker(
                preprocessing,
                merge_blocker_slot,
                u_max,
                root_copy_slot,
            ) {
                Ok(found) => Ok(found),
                Err(WalkDownExecutionError::InvalidK33Context) => Ok(false),
                Err(error) => Err(error),
            }
        }

        fn find_k33_with_merge_blocker(
            &mut self,
            preprocessing: &DfsPreprocessing,
            merge_blocker_slot: usize,
            _u_max: usize,
            _root_copy_slot: usize,
        ) -> Result<bool, WalkDownExecutionError> {
            self.orient_embedding_from_active_bicomps(true);
            self.restore_all_reduced_paths();
            self.orient_embedding_from_active_bicomps(false);

            let mut root_prev_link = 1usize;
            let mut root_copy_slot = merge_blocker_slot;
            while !matches!(self.slots[root_copy_slot].kind, EmbeddingSlotKind::RootCopy { .. }) {
                root_copy_slot = self.real_ext_face_neighbor(root_copy_slot, &mut root_prev_link);
                if root_copy_slot == merge_blocker_slot {
                    return Err(WalkDownExecutionError::InvalidK33Context);
                }
            }

            let current_primary_slot = self.primary_from_root(root_copy_slot);

            self.reset_k33_reconstruction_state(preprocessing);
            let original_vertex = match self.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => {
                    return Err(WalkDownExecutionError::InvalidK33Context);
                }
            };
            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                if self.arcs[forward_arc].source_slot == usize::MAX
                    || self.arcs[forward_arc].target_slot == usize::MAX
                    || self.arcs[forward_arc].embedded
                {
                    continue;
                }
                self.walk_up(current_primary_slot, forward_arc);
            }

            let context = match self.classify_k33_minor(
                preprocessing,
                current_primary_slot,
                root_copy_slot,
                root_copy_slot,
                None,
            )? {
                K33MinorSearchOutcome::Minor(_) => return Ok(true),
                K33MinorSearchOutcome::MinorE(context) => context,
            };

            let reconstructed_u_max =
                core::cmp::max(context.ux, core::cmp::max(context.uy, context.uz));

            let candidate_slot = if merge_blocker_slot == context.x_slot {
                context.x_slot
            } else if merge_blocker_slot == context.y_slot {
                context.y_slot
            } else {
                return Err(WalkDownExecutionError::InvalidK33Context);
            };

            let ancestor = self.find_descendant_external_connection_ancestor(
                preprocessing,
                current_primary_slot,
                candidate_slot,
                reconstructed_u_max,
            );

            if ancestor.is_some_and(|ancestor_slot| ancestor_slot > reconstructed_u_max) {
                Ok(true)
            } else {
                Err(WalkDownExecutionError::InvalidK33Context)
            }
        }

        fn restore_all_reduced_paths(&mut self) {
            // Reduced paths can nest. Restoring newest-first rebuilds the
            // original face structure before older synthetic edges are expanded.
            while let Some(reduction_arc) =
                self.arcs.iter().rposition(|arc| arc.reduction_endpoint_arc.is_some())
            {
                let _ = self.restore_reduced_path_edge(reduction_arc);
            }
        }

        fn orient_embedding_from_active_bicomps(&mut self, preserve_signs: bool) {
            let active_root_copy_slots = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, record)| {
                    matches!(record.kind, EmbeddingSlotKind::RootCopy { .. })
                        .then_some(slot)
                        .filter(|_| record.first_arc.is_some())
                })
                .collect::<Vec<_>>();

            for root_copy_slot in active_root_copy_slots {
                self.orient_bicomp_from_root(root_copy_slot, preserve_signs);
            }
        }

        fn reset_k33_reconstruction_state(&mut self, preprocessing: &DfsPreprocessing) {
            let reset_value = self.primary_slot_by_original_vertex.len();

            for slot in &mut self.slots {
                slot.visited_info = reset_value;
                slot.pertinent_roots.clear();
                slot.pertinent_edge = None;
                slot.k33_merge_blocker = None;
                slot.k33_minor_e_reduced = false;
                slot.future_pertinent_child = match slot.kind {
                    EmbeddingSlotKind::Primary { .. } => slot.sorted_dfs_children.first().copied(),
                    EmbeddingSlotKind::RootCopy { .. } => None,
                };
            }

            for (primary_slot, head_index) in
                self.forward_arc_head_index_by_primary_slot.iter_mut().enumerate()
            {
                let original_vertex = match self.slots[primary_slot].kind {
                    EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                    EmbeddingSlotKind::RootCopy { .. } => continue,
                };
                *head_index =
                    (!preprocessing.vertices[original_vertex].sorted_forward_arcs.is_empty())
                        .then_some(0);
            }
        }

        fn continue_after_k33_minor_e(
            &mut self,
            _current_primary_slot: usize,
            context: &K33MinorEContext,
        ) -> Result<(), EmbeddingMutationError> {
            self.orient_bicomp_from_root(context.root_copy_slot, true);
            self.reduce_k33_minor_e_bicomp(context)?;

            let reset_value = self.primary_slot_by_original_vertex.len();
            self.fill_visited_info_in_bicomp(context.root_copy_slot, reset_value);

            self.slots[context.root_copy_slot].k33_minor_e_reduced = true;
            self.slots[context.w_slot].pertinent_edge = None;
            self.slots[context.w_slot].pertinent_roots.clear();
            Ok(())
        }

        pub(crate) fn search_for_k33_in_bicomp(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            bicomp_root_copy_slot: usize,
            stack_root_copy_slot: Option<usize>,
        ) -> Result<K33BicompSearchOutcome, WalkDownExecutionError> {
            let search_outcome = self.classify_k33_minor(
                preprocessing,
                current_primary_slot,
                walk_root_copy_slot,
                bicomp_root_copy_slot,
                stack_root_copy_slot,
            )?;
            match search_outcome {
                K33MinorSearchOutcome::Minor(_) => Ok(K33BicompSearchOutcome::MinorFound),
                K33MinorSearchOutcome::MinorE(context) => {
                    let xy_max = core::cmp::max(context.ux, context.uy);
                    if context.z_slot != context.w_slot
                        || context.uz > xy_max
                        || (context.uz < xy_max && context.ux != context.uy)
                        || (context.x_slot != context.px_slot || context.y_slot != context.py_slot)
                    {
                        return Ok(K33BicompSearchOutcome::MinorFound);
                    }

                    match self.run_extra_k33_tests(&context) {
                        Ok(Some(_)) => Ok(K33BicompSearchOutcome::MinorFound),
                        Ok(None) => Ok(K33BicompSearchOutcome::ContinueMinorE { context }),
                        Err(error) => Err(error),
                    }
                }
            }
        }

        #[allow(dead_code)]
        fn pertinent_root_has_lowpoint_below(
            &self,
            slot: usize,
            current_primary_slot: usize,
        ) -> bool {
            self.slots[slot].pertinent_roots.last().copied().is_some_and(|root_copy_slot| {
                self.lowpoint_by_primary_slot[self.dfs_child_from_root(root_copy_slot)]
                    < current_primary_slot
            })
        }

        #[allow(dead_code)]
        #[allow(clippy::too_many_lines)]
        pub(crate) fn classify_k33_minor(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            bicomp_root_copy_slot: usize,
            stack_root_copy_slot: Option<usize>,
        ) -> Result<K33MinorSearchOutcome, WalkDownExecutionError> {
            let Ok(context) = self.initialize_nonplanarity_context(
                preprocessing,
                current_primary_slot,
                walk_root_copy_slot,
                bicomp_root_copy_slot,
                stack_root_copy_slot,
            ) else {
                return Err(WalkDownExecutionError::InvalidK33Context);
            };

            if self.primary_from_root(context.root_copy_slot) != current_primary_slot {
                return Ok(K33MinorSearchOutcome::Minor(K33MinorType::A));
            }

            if self.pertinent_root_has_lowpoint_below(context.w_slot, current_primary_slot) {
                return Ok(K33MinorSearchOutcome::Minor(K33MinorType::B));
            }

            let obstruction_marks = self.set_k33_obstruction_marks(&context);
            self.clear_all_visited_flags_in_bicomp(context.root_copy_slot);
            let mut highest_xy_path = match self.mark_closest_xy_path(
                &context,
                &obstruction_marks,
                context.root_copy_slot,
            ) {
                Ok(Some(highest_xy_path)) => highest_xy_path,
                Ok(None) => {
                    return Err(WalkDownExecutionError::InvalidK33Context);
                }
                Err(error) => return Err(error),
            };

            if highest_xy_path.px_slot == usize::MAX || highest_xy_path.py_slot == usize::MAX {
                return Err(WalkDownExecutionError::InvalidK33Context);
            }

            if matches!(obstruction_marks[highest_xy_path.px_slot], K33ObstructionMark::HighRxw)
                || matches!(obstruction_marks[highest_xy_path.py_slot], K33ObstructionMark::HighRyw)
            {
                return Ok(K33MinorSearchOutcome::Minor(K33MinorType::C));
            }

            match self.mark_z_to_r_path(&context, &obstruction_marks, &mut highest_xy_path) {
                Ok(Some(_)) => return Ok(K33MinorSearchOutcome::Minor(K33MinorType::D)),
                Ok(None) => {}
                Err(K33ZToRPathFailure::Structural | K33ZToRPathFailure::MarkedSlot { .. }) => {
                    return Err(WalkDownExecutionError::InvalidK33Context);
                }
            }

            let Some(z_slot) =
                self.find_future_pertinence_below_xy_path(&context, &highest_xy_path)
            else {
                return Err(WalkDownExecutionError::InvalidK33Context);
            };

            let ux = self
                .least_ancestor_connection(context.x_slot, current_primary_slot)
                .ok_or(WalkDownExecutionError::InvalidK33Context)?;
            let uy = self
                .least_ancestor_connection(context.y_slot, current_primary_slot)
                .ok_or(WalkDownExecutionError::InvalidK33Context)?;
            let uz = self
                .least_ancestor_connection(z_slot, current_primary_slot)
                .ok_or(WalkDownExecutionError::InvalidK33Context)?;

            let outcome = K33MinorSearchOutcome::MinorE(K33MinorEContext {
                current_primary_slot,
                root_copy_slot: context.root_copy_slot,
                x_slot: context.x_slot,
                y_slot: context.y_slot,
                w_slot: context.w_slot,
                z_slot,
                px_slot: highest_xy_path.px_slot,
                py_slot: highest_xy_path.py_slot,
                ux,
                uy,
                uz,
                obstruction_marks,
            });
            Ok(outcome)
        }

        pub(crate) fn search_for_k23_in_bicomp(
            &mut self,
            _preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            blocked_root_copy_slot: usize,
            _stack_root_copy_slot: Option<usize>,
        ) -> Result<K23BicompSearchOutcome, WalkDownExecutionError> {
            if blocked_root_copy_slot != walk_root_copy_slot {
                return Ok(K23BicompSearchOutcome::MinorA);
            }

            let context = self
                .initialize_nonouterplanarity_context(
                    current_primary_slot,
                    blocked_root_copy_slot,
                    None,
                )
                .map_err(|_| WalkDownExecutionError::InvalidK23Context)?;

            if !self.slots[context.w_slot].pertinent_roots.is_empty() {
                return Ok(K23BicompSearchOutcome::MinorB);
            }

            let mut x_prev_link = context.x_prev_link;
            let mut y_prev_link = context.y_prev_link;
            if self.real_ext_face_neighbor(context.x_slot, &mut x_prev_link) != context.w_slot
                || self.real_ext_face_neighbor(context.y_slot, &mut y_prev_link) != context.w_slot
            {
                return Ok(K23BicompSearchOutcome::MinorE1OrE2);
            }

            self.update_future_pertinent_child(context.x_slot, context.current_primary_slot);
            self.update_future_pertinent_child(context.y_slot, context.current_primary_slot);
            self.update_future_pertinent_child(context.w_slot, context.current_primary_slot);

            if self.is_future_pertinent(context.x_slot, context.current_primary_slot)
                || self.is_future_pertinent(context.y_slot, context.current_primary_slot)
                || self.is_future_pertinent(context.w_slot, context.current_primary_slot)
            {
                Ok(K23BicompSearchOutcome::MinorE3OrE4)
            } else {
                // Boyer restores the bicomp orientation before letting WalkDown
                // retry after discovering that the obstruction was only a
                // separable K4.
                self.orient_bicomp_from_root(context.root_copy_slot, true);
                Ok(K23BicompSearchOutcome::SeparableK4)
            }
        }

        pub(crate) fn continue_after_same_root_separable_k4(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
        ) {
            let child_primary_slot = self.dfs_child_from_root(walk_root_copy_slot);
            let next_child_primary_slot =
                self.next_dfs_child(current_primary_slot, child_primary_slot);
            self.advance_forward_arc_list(
                preprocessing,
                current_primary_slot,
                child_primary_slot,
                next_child_primary_slot,
            );
        }

        fn is_virtual(&self, slot: usize) -> bool {
            matches!(self.slots[slot].kind, EmbeddingSlotKind::RootCopy { .. })
        }

        fn primary_from_root(&self, root_copy_slot: usize) -> usize {
            match self.slots[root_copy_slot].kind {
                EmbeddingSlotKind::RootCopy { parent_primary_slot, .. } => parent_primary_slot,
                EmbeddingSlotKind::Primary { .. } => {
                    panic!("expected root-copy slot while stepping up during walk-up")
                }
            }
        }

        fn dfs_child_from_root(&self, root_copy_slot: usize) -> usize {
            match self.slots[root_copy_slot].kind {
                EmbeddingSlotKind::RootCopy { dfs_child_primary_slot, .. } => {
                    dfs_child_primary_slot
                }
                EmbeddingSlotKind::Primary { .. } => {
                    panic!("expected root-copy slot while querying DFS child")
                }
            }
        }

        fn record_pertinent_root(
            &mut self,
            primary_slot: usize,
            root_copy_slot: usize,
            current_primary_slot: usize,
        ) {
            let dfs_child_primary_slot = self.dfs_child_from_root(root_copy_slot);
            if self.lowpoint_by_primary_slot[dfs_child_primary_slot] < current_primary_slot {
                self.slots[primary_slot].pertinent_roots.push(root_copy_slot);
            } else {
                self.slots[primary_slot].pertinent_roots.insert(0, root_copy_slot);
            }
        }

        fn is_separated_dfs_child(&self, primary_slot: usize) -> bool {
            self.root_copy_by_primary_dfi[primary_slot]
                .and_then(|root_copy_slot| self.slots[root_copy_slot].first_arc)
                .is_some()
        }

        fn is_singleton_slot(&self, slot: usize) -> bool {
            self.slots[slot].first_arc.is_some()
                && self.slots[slot].first_arc == self.slots[slot].last_arc
        }

        pub(crate) fn next_dfs_child(
            &self,
            primary_slot: usize,
            child_slot: usize,
        ) -> Option<usize> {
            let sorted_children = &self.slots[primary_slot].sorted_dfs_children;
            sorted_children
                .iter()
                .position(|&candidate| candidate == child_slot)
                .and_then(|position| sorted_children.get(position + 1).copied())
        }

        fn normalize_forward_arc_head(
            &self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            head_index: Option<usize>,
        ) -> Option<(usize, usize)> {
            let original_vertex = match self.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => {
                    panic!("forward-arc heads are tracked only for primary slots")
                }
            };
            let sorted_forward_arcs = &preprocessing.vertices[original_vertex].sorted_forward_arcs;
            let mut head_index = head_index?;
            if sorted_forward_arcs.is_empty() {
                return None;
            }
            if head_index >= sorted_forward_arcs.len() {
                return None;
            }

            while head_index < sorted_forward_arcs.len() {
                let forward_arc = sorted_forward_arcs[head_index];
                if self.arcs[forward_arc].source_slot != usize::MAX
                    && self.arcs[forward_arc].target_slot != usize::MAX
                    && !self.arcs[forward_arc].embedded
                {
                    return Some((head_index, forward_arc));
                }

                head_index += 1;
            }

            None
        }

        pub(crate) fn peek_forward_arc_head(
            &self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
        ) -> Option<usize> {
            self.normalize_forward_arc_head(
                preprocessing,
                current_primary_slot,
                self.forward_arc_head_index_by_primary_slot[current_primary_slot],
            )
            .map(|(_, forward_arc)| forward_arc)
        }

        fn current_forward_arc_head(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
        ) -> Option<usize> {
            let normalized_head = self.normalize_forward_arc_head(
                preprocessing,
                current_primary_slot,
                self.forward_arc_head_index_by_primary_slot[current_primary_slot],
            );
            self.forward_arc_head_index_by_primary_slot[current_primary_slot] =
                normalized_head.map(|(head_index, _)| head_index);
            normalized_head.map(|(_, forward_arc)| forward_arc)
        }

        #[allow(dead_code)]
        fn find_nonplanarity_bicomp_root(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
        ) -> Option<usize> {
            let forward_arc = self.current_forward_arc_head(preprocessing, current_primary_slot)?;
            let descendant_primary_slot = self.arcs[forward_arc].target_slot;

            let dfs_child_primary_slot = self.slots[current_primary_slot]
                .sorted_dfs_children
                .iter()
                .copied()
                .take_while(|&child_slot| child_slot <= descendant_primary_slot)
                .last()?;

            self.root_copy_by_primary_dfi[dfs_child_primary_slot]
        }

        pub(crate) fn child_subtree_forward_arc_head(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            child_primary_slot: usize,
            next_child_primary_slot: Option<usize>,
        ) -> Option<usize> {
            let forward_arc = self.current_forward_arc_head(preprocessing, current_primary_slot)?;
            let descendant_primary_slot = self.arcs[forward_arc].target_slot;

            (descendant_primary_slot > child_primary_slot
                && next_child_primary_slot
                    .is_none_or(|next_child| descendant_primary_slot < next_child))
            .then_some(forward_arc)
        }

        pub(crate) fn advance_forward_arc_list(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            child_primary_slot: usize,
            next_child_primary_slot: Option<usize>,
        ) {
            let original_vertex = match self.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => {
                    panic!("forward-arc heads are tracked only for primary slots")
                }
            };
            let sorted_forward_arcs = &preprocessing.vertices[original_vertex].sorted_forward_arcs;
            let Some((start_head_index, _)) = self.normalize_forward_arc_head(
                preprocessing,
                current_primary_slot,
                self.forward_arc_head_index_by_primary_slot[current_primary_slot],
            ) else {
                self.forward_arc_head_index_by_primary_slot[current_primary_slot] = None;
                return;
            };

            let mut head_index = start_head_index;
            while head_index < sorted_forward_arcs.len() {
                let forward_arc = sorted_forward_arcs[head_index];
                let descendant_primary_slot = self.arcs[forward_arc].target_slot;
                if descendant_primary_slot < child_primary_slot
                    || next_child_primary_slot
                        .is_some_and(|next_child| next_child <= descendant_primary_slot)
                {
                    self.forward_arc_head_index_by_primary_slot[current_primary_slot] =
                        Some(head_index);
                    return;
                }

                head_index += 1;
            }

            self.forward_arc_head_index_by_primary_slot[current_primary_slot] = None;
        }

        pub(crate) fn update_future_pertinent_child(
            &mut self,
            primary_slot: usize,
            current_primary_slot: usize,
        ) {
            while let Some(child_slot) = self.slots[primary_slot].future_pertinent_child {
                if self.lowpoint_by_primary_slot[child_slot] >= current_primary_slot
                    || !self.is_separated_dfs_child(child_slot)
                {
                    self.slots[primary_slot].future_pertinent_child =
                        self.next_dfs_child(primary_slot, child_slot);
                } else {
                    break;
                }
            }
        }

        fn remove_separated_dfs_child(&mut self, primary_slot: usize, child_slot: usize) {
            self.slots[primary_slot]
                .separated_dfs_children
                .retain(|&candidate| candidate != child_slot);
        }

        fn invert_vertex(&mut self, slot: usize) {
            let mut current_arc = self.slots[slot].first_arc;
            while let Some(arc) = current_arc {
                current_arc = self.arcs[arc].next;
                let arc_record = &mut self.arcs[arc];
                mem::swap(&mut arc_record.next, &mut arc_record.prev);
            }

            let slot_record = &mut self.slots[slot];
            mem::swap(&mut slot_record.first_arc, &mut slot_record.last_arc);
            self.slots[slot].ext_face.swap(0, 1);
        }

        fn redirect_merged_root_references(
            &mut self,
            root_copy_slot: usize,
            cut_vertex_slot: usize,
        ) {
            self.slots[cut_vertex_slot]
                .pertinent_roots
                .retain(|&candidate| candidate != root_copy_slot);
        }

        fn is_pertinent(&self, primary_slot: usize) -> bool {
            self.slots[primary_slot].pertinent_edge.is_some()
                || !self.slots[primary_slot].pertinent_roots.is_empty()
        }

        pub(crate) fn is_future_pertinent(&self, slot: usize, current_primary_slot: usize) -> bool {
            match self.slots[slot].kind {
                EmbeddingSlotKind::Primary { .. } => {
                    self.least_ancestor_by_primary_slot[slot] < current_primary_slot
                        || self.slots[slot].future_pertinent_child.is_some()
                }
                EmbeddingSlotKind::RootCopy { .. } => false,
            }
        }

        #[allow(clippy::similar_names)]
        fn choose_root_descent_for_mode(
            &mut self,
            mode: super::EmbeddingRunMode,
            current_primary_slot: usize,
            root_copy_slot: usize,
        ) -> Option<(usize, usize, usize)> {
            let mut x_prev_link = 1usize;
            let x_slot = self.walk_ext_face_neighbor(mode, root_copy_slot, &mut x_prev_link);
            let mut y_prev_link = 0usize;
            let y_slot = self.walk_ext_face_neighbor(mode, root_copy_slot, &mut y_prev_link);

            self.update_future_pertinent_child(x_slot, current_primary_slot);
            self.update_future_pertinent_child(y_slot, current_primary_slot);

            if self.is_pertinent(x_slot) && !self.is_future_pertinent(x_slot, current_primary_slot)
            {
                return Some((x_slot, x_prev_link, 0));
            }
            if self.is_pertinent(y_slot) && !self.is_future_pertinent(y_slot, current_primary_slot)
            {
                return Some((y_slot, y_prev_link, 1));
            }
            if self.is_pertinent(x_slot) {
                return Some((x_slot, x_prev_link, 0));
            }
            if self.is_pertinent(y_slot) {
                return Some((y_slot, y_prev_link, 1));
            }

            None
        }

        #[allow(dead_code)]
        fn choose_root_descent(
            &mut self,
            current_primary_slot: usize,
            root_copy_slot: usize,
        ) -> Option<(usize, usize, usize)> {
            self.choose_root_descent_for_mode(
                super::EmbeddingRunMode::Planarity,
                current_primary_slot,
                root_copy_slot,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn resolve_pertinent_root_walk_action(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            walk_root_side: usize,
            current_slot: usize,
            current_entry_side: usize,
            blocked_root_copy_slot: usize,
            frames: &[WalkDownFrame],
            mode: super::EmbeddingRunMode,
        ) -> Result<PertinentRootWalkAction, WalkDownExecutionError> {
            if let Some((next_slot, next_entry_side, chosen_root_side)) = self
                .choose_root_descent_for_mode(mode, current_primary_slot, blocked_root_copy_slot)
            {
                return Ok(PertinentRootWalkAction::Descend {
                    next_slot,
                    next_entry_side,
                    chosen_root_side,
                });
            }

            let context = BlockedBicompContext {
                current_primary_slot,
                walk_root_copy_slot,
                walk_root_side,
                cut_vertex_slot: current_slot,
                cut_vertex_entry_side: current_entry_side,
                blocked_root_copy_slot,
            };

            match mode {
                super::EmbeddingRunMode::K23Search => {
                    match self.search_for_k23_in_bicomp(
                        preprocessing,
                        current_primary_slot,
                        walk_root_copy_slot,
                        blocked_root_copy_slot,
                        frames.last().map(|frame| frame.root_copy_slot),
                    )? {
                        K23BicompSearchOutcome::SeparableK4 => {
                            self.continue_after_same_root_separable_k4(
                                preprocessing,
                                current_primary_slot,
                                walk_root_copy_slot,
                            );
                            Ok(PertinentRootWalkAction::Return(WalkDownChildOutcome::Completed))
                        }
                        K23BicompSearchOutcome::MinorA
                        | K23BicompSearchOutcome::MinorB
                        | K23BicompSearchOutcome::MinorE1OrE2
                        | K23BicompSearchOutcome::MinorE3OrE4 => {
                            Ok(PertinentRootWalkAction::Return(WalkDownChildOutcome::K23Found))
                        }
                    }
                }
                super::EmbeddingRunMode::K33Search => {
                    match self.search_for_k33_in_bicomp(
                        preprocessing,
                        current_primary_slot,
                        walk_root_copy_slot,
                        walk_root_copy_slot,
                        (blocked_root_copy_slot != walk_root_copy_slot)
                            .then_some(blocked_root_copy_slot),
                    ) {
                        Err(WalkDownExecutionError::InvalidK33Context) => {
                            Err(WalkDownExecutionError::InvalidK33Context)
                        }
                        Err(error) => Err(error),
                        Ok(K33BicompSearchOutcome::MinorFound) => {
                            Ok(PertinentRootWalkAction::Return(WalkDownChildOutcome::K33Found))
                        }
                        Ok(K33BicompSearchOutcome::ContinueMinorE { context }) => {
                            self.continue_after_k33_minor_e(current_primary_slot, &context)?;
                            Ok(PertinentRootWalkAction::ContinueWalkdown)
                        }
                    }
                }
                super::EmbeddingRunMode::K4Search => {
                    match self.handle_k4_blocked_bicomp(
                        preprocessing,
                        current_primary_slot,
                        walk_root_copy_slot,
                        blocked_root_copy_slot,
                    )? {
                        K4BlockedBicompOutcome::Found => {
                            Ok(PertinentRootWalkAction::Return(WalkDownChildOutcome::K4Found))
                        }
                        K4BlockedBicompOutcome::ContinueWalkdown => {
                            Ok(PertinentRootWalkAction::ContinueWalkdown)
                        }
                        K4BlockedBicompOutcome::Completed => {
                            Ok(PertinentRootWalkAction::Return(WalkDownChildOutcome::Completed))
                        }
                    }
                }
                super::EmbeddingRunMode::Planarity | super::EmbeddingRunMode::Outerplanarity => {
                    Err(WalkDownExecutionError::BlockedBicomp { context })
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn resolve_child_subtree_action(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            root_copy_slot: usize,
            child_primary_slot: usize,
            next_child_primary_slot: Option<usize>,
            mode: super::EmbeddingRunMode,
        ) -> Result<ChildSubtreeAction, WalkDownExecutionError> {
            match mode {
                super::EmbeddingRunMode::K23Search => {
                    let _ = child_primary_slot;
                    let _ = next_child_primary_slot;
                    match self.search_for_k23_in_bicomp(
                        preprocessing,
                        current_primary_slot,
                        root_copy_slot,
                        root_copy_slot,
                        None,
                    )? {
                        K23BicompSearchOutcome::SeparableK4 => {
                            self.continue_after_same_root_separable_k4(
                                preprocessing,
                                current_primary_slot,
                                root_copy_slot,
                            );
                            Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::Completed))
                        }
                        K23BicompSearchOutcome::MinorA
                        | K23BicompSearchOutcome::MinorB
                        | K23BicompSearchOutcome::MinorE1OrE2
                        | K23BicompSearchOutcome::MinorE3OrE4 => {
                            Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::K23Found))
                        }
                    }
                }
                super::EmbeddingRunMode::K33Search => {
                    let _ = child_primary_slot;
                    let _ = next_child_primary_slot;
                    if self.slots[root_copy_slot].k33_minor_e_reduced {
                        return Ok(ChildSubtreeAction::AdvanceAndReturn(
                            WalkDownChildOutcome::Completed,
                        ));
                    }
                    match self.search_for_k33_in_bicomp(
                        preprocessing,
                        current_primary_slot,
                        root_copy_slot,
                        root_copy_slot,
                        None,
                    ) {
                        Err(WalkDownExecutionError::InvalidK33Context) => {
                            Ok(ChildSubtreeAction::AdvanceAndReturn(
                                WalkDownChildOutcome::Completed,
                            ))
                        }
                        Err(error) => Err(error),
                        Ok(K33BicompSearchOutcome::MinorFound) => {
                            Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::K33Found))
                        }
                        Ok(K33BicompSearchOutcome::ContinueMinorE { context }) => {
                            self.continue_after_k33_minor_e(current_primary_slot, &context)?;
                            Ok(ChildSubtreeAction::AdvanceAndReturn(
                                WalkDownChildOutcome::Completed,
                            ))
                        }
                    }
                }
                super::EmbeddingRunMode::K4Search => {
                    let _ = child_primary_slot;
                    let _ = next_child_primary_slot;
                    if self.handling_k4_blocked_bicomp {
                        self.k4_reblocked_same_root = true;
                        return Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::Completed));
                    }

                    match self.handle_k4_blocked_bicomp(
                        preprocessing,
                        current_primary_slot,
                        root_copy_slot,
                        root_copy_slot,
                    ) {
                        Err(WalkDownExecutionError::InvalidK4Context) => {
                            Ok(ChildSubtreeAction::AdvanceAndReturn(
                                WalkDownChildOutcome::Completed,
                            ))
                        }
                        Err(error) => Err(error),
                        Ok(K4BlockedBicompOutcome::Found) => {
                            Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::K4Found))
                        }
                        Ok(K4BlockedBicompOutcome::ContinueWalkdown) => unreachable!(),
                        Ok(K4BlockedBicompOutcome::Completed) => {
                            Ok(ChildSubtreeAction::Return(WalkDownChildOutcome::Completed))
                        }
                    }
                }
                super::EmbeddingRunMode::Planarity | super::EmbeddingRunMode::Outerplanarity => {
                    let _ = child_primary_slot;
                    let _ = next_child_primary_slot;
                    Ok(ChildSubtreeAction::ErrorUnembeddedForwardArc)
                }
            }
        }

        fn finalize_child_subtree_action(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            child_primary_slot: usize,
            next_child_primary_slot: Option<usize>,
            forward_arc: usize,
            action: ChildSubtreeAction,
        ) -> Result<WalkDownChildOutcome, WalkDownExecutionError> {
            match action {
                ChildSubtreeAction::AdvanceAndReturn(outcome) => {
                    self.advance_forward_arc_list(
                        preprocessing,
                        current_primary_slot,
                        child_primary_slot,
                        next_child_primary_slot,
                    );
                    Ok(outcome)
                }
                ChildSubtreeAction::Return(outcome) => Ok(outcome),
                ChildSubtreeAction::ErrorUnembeddedForwardArc => {
                    Err(WalkDownExecutionError::UnembeddedForwardArcInChildSubtree { forward_arc })
                }
            }
        }

        #[allow(clippy::similar_names)]
        #[allow(clippy::too_many_lines)]
        pub(crate) fn walk_up(&mut self, current_primary_slot: usize, forward_arc: usize) {
            if self.arcs[forward_arc].source_slot == usize::MAX
                || self.arcs[forward_arc].target_slot == usize::MAX
            {
                return;
            }

            let descendant_primary_slot = self.arcs[forward_arc].target_slot;
            self.slots[descendant_primary_slot].pertinent_edge = Some(forward_arc);

            let mut zig = descendant_primary_slot;
            let mut zag = descendant_primary_slot;
            let mut zig_entry_side = 1usize;
            let mut zag_entry_side = 0usize;

            while zig != current_primary_slot {
                let mut next_zig_entry_side = zig_entry_side;
                let zig_candidate = self.shortcut_ext_face_neighbor(zig, &mut next_zig_entry_side);
                let mut root_copy_slot = None;
                let (next_zig_vertex, next_zag_vertex) = if self.is_virtual(zig_candidate) {
                    if self.slots[zig].visited_info == current_primary_slot {
                        break;
                    }
                    let root = zig_candidate;
                    root_copy_slot = Some(root);
                    let mut other_root_entry_side =
                        usize::from(self.ext_face_vertex(root, 0) == zig);
                    let zag_vertex =
                        self.shortcut_ext_face_neighbor(root, &mut other_root_entry_side);
                    if self.slots[zag_vertex].visited_info == current_primary_slot {
                        break;
                    }
                    (zig_candidate, zag_vertex)
                } else {
                    let mut next_zag_entry_side = zag_entry_side;
                    let zag_candidate =
                        self.shortcut_ext_face_neighbor(zag, &mut next_zag_entry_side);
                    if self.is_virtual(zag_candidate) {
                        if self.slots[zag].visited_info == current_primary_slot {
                            break;
                        }
                        let root = zag_candidate;
                        root_copy_slot = Some(root);
                        let mut other_root_entry_side =
                            usize::from(self.ext_face_vertex(root, 0) == zag);
                        let zig_vertex =
                            self.shortcut_ext_face_neighbor(root, &mut other_root_entry_side);
                        if self.slots[zig_vertex].visited_info == current_primary_slot {
                            break;
                        }
                        (zig_vertex, zag_candidate)
                    } else {
                        if self.slots[zig].visited_info == current_primary_slot
                            || self.slots[zag].visited_info == current_primary_slot
                        {
                            break;
                        }
                        zig_entry_side = next_zig_entry_side;
                        zag_entry_side = next_zag_entry_side;
                        (zig_candidate, zag_candidate)
                    }
                };

                self.slots[zig].visited_info = current_primary_slot;
                self.slots[zag].visited_info = current_primary_slot;

                if let Some(root) = root_copy_slot {
                    let primary_slot = self.primary_from_root(root);
                    zig = primary_slot;
                    zag = primary_slot;
                    zig_entry_side = 1;
                    zag_entry_side = 0;
                    self.record_pertinent_root(primary_slot, root, current_primary_slot);
                } else {
                    zig = next_zig_vertex;
                    zag = next_zag_vertex;
                }
            }
        }

        pub(crate) fn walk_down_trace(
            &mut self,
            current_primary_slot: usize,
            root_copy_slot: usize,
            root_side: usize,
            outerplanar_mode: bool,
        ) -> WalkDownTrace {
            let mut trace = WalkDownTrace {
                root_copy_slot,
                root_side,
                visited_slots: Vec::new(),
                frames: Vec::new(),
                outcome: WalkDownOutcome::CompletedToRoot,
            };

            let mut current_entry_side = 1 ^ root_side;
            let mut current_slot =
                self.shortcut_ext_face_neighbor(root_copy_slot, &mut current_entry_side);

            while current_slot != root_copy_slot {
                trace.visited_slots.push(current_slot);

                if self.slots[current_slot].pertinent_edge.is_some() {
                    trace.outcome = WalkDownOutcome::DescendantFound {
                        slot: current_slot,
                        entry_side: current_entry_side,
                    };
                    return trace;
                }

                if let Some(root_to_descend) =
                    self.slots[current_slot].pertinent_roots.first().copied()
                {
                    if let Some((next_slot, next_entry_side, chosen_root_side)) = self
                        .choose_root_descent_for_mode(
                            super::EmbeddingRunMode::Planarity,
                            current_primary_slot,
                            root_to_descend,
                        )
                    {
                        trace.frames.push(WalkDownFrame {
                            cut_vertex_slot: current_slot,
                            cut_vertex_entry_side: current_entry_side,
                            root_copy_slot: root_to_descend,
                            root_side: chosen_root_side,
                        });
                        current_slot = next_slot;
                        current_entry_side = next_entry_side;
                        continue;
                    }

                    trace.outcome =
                        WalkDownOutcome::BlockedBicomp { root_copy_slot: root_to_descend };
                    return trace;
                }

                self.update_future_pertinent_child(current_slot, current_primary_slot);
                if self.is_future_pertinent(current_slot, current_primary_slot) || outerplanar_mode
                {
                    trace.outcome = WalkDownOutcome::StoppingVertex {
                        slot: current_slot,
                        entry_side: current_entry_side,
                    };
                    return trace;
                }

                current_slot =
                    self.shortcut_ext_face_neighbor(current_slot, &mut current_entry_side);
            }

            trace
        }

        fn merge_vertex(
            &mut self,
            cut_vertex_slot: usize,
            cut_vertex_entry_side: usize,
            root_copy_slot: usize,
        ) -> Result<(), EmbeddingMutationError> {
            let dfs_child_primary_slot = self.dfs_child_from_root(root_copy_slot);
            let mut current_arc = self.slots[root_copy_slot].first_arc;
            while let Some(arc) = current_arc {
                current_arc = self.arcs[arc].next;
                self.arcs[arc].source_slot = cut_vertex_slot;
                let twin = self.arcs[arc].twin;
                self.arcs[twin].target_slot = cut_vertex_slot;
            }

            let cut_vertex_arc = self.slot_arc(cut_vertex_slot, cut_vertex_entry_side);
            let root_inner_arc = self.slot_arc(root_copy_slot, 1 ^ cut_vertex_entry_side).ok_or(
                EmbeddingMutationError::MissingSlotArc {
                    slot: root_copy_slot,
                    side: 1 ^ cut_vertex_entry_side,
                },
            )?;
            let root_outer_arc = self.slot_arc(root_copy_slot, cut_vertex_entry_side).ok_or(
                EmbeddingMutationError::MissingSlotArc {
                    slot: root_copy_slot,
                    side: cut_vertex_entry_side,
                },
            )?;

            if let Some(cut_vertex_arc) = cut_vertex_arc {
                self.set_adjacent_arc(
                    cut_vertex_arc,
                    1 ^ cut_vertex_entry_side,
                    Some(root_inner_arc),
                );
                self.set_adjacent_arc(root_inner_arc, cut_vertex_entry_side, Some(cut_vertex_arc));
                self.set_slot_arc(cut_vertex_slot, cut_vertex_entry_side, Some(root_outer_arc));
                self.set_adjacent_arc(root_outer_arc, 1 ^ cut_vertex_entry_side, None);
            } else {
                self.set_slot_arc(cut_vertex_slot, 1 ^ cut_vertex_entry_side, Some(root_inner_arc));
                self.set_adjacent_arc(root_inner_arc, cut_vertex_entry_side, None);
                self.set_slot_arc(cut_vertex_slot, cut_vertex_entry_side, Some(root_outer_arc));
                self.set_adjacent_arc(root_outer_arc, 1 ^ cut_vertex_entry_side, None);
            }

            self.redirect_merged_root_references(root_copy_slot, cut_vertex_slot);
            self.remove_separated_dfs_child(cut_vertex_slot, dfs_child_primary_slot);

            self.slots[root_copy_slot].first_arc = None;
            self.slots[root_copy_slot].last_arc = None;
            self.slots[root_copy_slot].ext_face = [None, None];
            self.slots[root_copy_slot].pertinent_roots.clear();
            self.slots[root_copy_slot].future_pertinent_child = None;
            self.slots[root_copy_slot].pertinent_edge = None;

            Ok(())
        }

        fn merge_trace_frames(
            &mut self,
            current_primary_slot: usize,
            frames: &[WalkDownFrame],
        ) -> Result<(), EmbeddingMutationError> {
            for frame in frames.iter().rev() {
                let cut_vertex_slot = frame.cut_vertex_slot;
                let cut_vertex_entry_side = frame.cut_vertex_entry_side;
                let root_copy_slot = frame.root_copy_slot;
                let ext_face_vertex = self.ext_face_vertex(root_copy_slot, 1 ^ frame.root_side);
                self.slots[cut_vertex_slot].ext_face[cut_vertex_entry_side] = Some(ext_face_vertex);

                if self.slots[ext_face_vertex].ext_face[0]
                    == self.slots[ext_face_vertex].ext_face[1]
                {
                    self.slots[ext_face_vertex].ext_face[frame.root_side] = Some(cut_vertex_slot);
                } else {
                    let ext_face_side =
                        usize::from(self.ext_face_vertex(ext_face_vertex, 0) != root_copy_slot);
                    self.slots[ext_face_vertex].ext_face[ext_face_side] = Some(cut_vertex_slot);
                }

                let mut root_side = frame.root_side;
                if cut_vertex_entry_side == frame.root_side {
                    root_side = 1 ^ cut_vertex_entry_side;
                    if !self.is_singleton_slot(root_copy_slot) {
                        // Boyer skips the root-copy inversion only for singleton
                        // bicomps, but it still xors the DFS-child sign below.
                        self.invert_vertex(root_copy_slot);
                    }
                    if let Some(child_arc) = self.root_copy_child_arc(root_copy_slot) {
                        self.arcs[child_arc].inverted = !self.arcs[child_arc].inverted;
                    }
                }

                self.slots[cut_vertex_slot]
                    .pertinent_roots
                    .retain(|&candidate| candidate != root_copy_slot);

                if self.slots[cut_vertex_slot].future_pertinent_child
                    == Some(self.dfs_child_from_root(root_copy_slot))
                {
                    self.slots[cut_vertex_slot].future_pertinent_child = self
                        .next_dfs_child(cut_vertex_slot, self.dfs_child_from_root(root_copy_slot));
                }

                self.merge_vertex(cut_vertex_slot, cut_vertex_entry_side, root_copy_slot)?;

                if root_side != frame.root_side {
                    self.update_future_pertinent_child(cut_vertex_slot, current_primary_slot);
                }
            }
            Ok(())
        }

        fn embed_back_edge_to_descendant(
            &mut self,
            root_copy_slot: usize,
            root_side: usize,
            descendant_slot: usize,
            descendant_entry_side: usize,
        ) -> Result<(), EmbeddingMutationError> {
            let forward_arc = self.slots[descendant_slot]
                .pertinent_edge
                .take()
                .ok_or(EmbeddingMutationError::MissingPertinentEdge { slot: descendant_slot })?;
            let back_arc = self.arcs[forward_arc].twin;

            let root_arc = self.slot_arc(root_copy_slot, root_side).ok_or(
                EmbeddingMutationError::MissingSlotArc { slot: root_copy_slot, side: root_side },
            )?;
            let descendant_arc = self.slot_arc(descendant_slot, descendant_entry_side).ok_or(
                EmbeddingMutationError::MissingSlotArc {
                    slot: descendant_slot,
                    side: descendant_entry_side,
                },
            )?;

            self.arcs[forward_arc].source_slot = root_copy_slot;
            self.arcs[forward_arc].target_slot = descendant_slot;
            self.arcs[forward_arc].embedded = true;
            self.set_adjacent_arc(forward_arc, 1 ^ root_side, None);
            self.set_adjacent_arc(forward_arc, root_side, Some(root_arc));
            self.set_adjacent_arc(root_arc, 1 ^ root_side, Some(forward_arc));
            self.set_slot_arc(root_copy_slot, root_side, Some(forward_arc));

            self.arcs[back_arc].source_slot = descendant_slot;
            self.arcs[back_arc].target_slot = root_copy_slot;
            self.arcs[back_arc].embedded = true;
            self.set_adjacent_arc(back_arc, 1 ^ descendant_entry_side, None);
            self.set_adjacent_arc(back_arc, descendant_entry_side, Some(descendant_arc));
            self.set_adjacent_arc(descendant_arc, 1 ^ descendant_entry_side, Some(back_arc));
            self.set_slot_arc(descendant_slot, descendant_entry_side, Some(back_arc));

            self.slots[root_copy_slot].ext_face[root_side] = Some(descendant_slot);
            self.slots[descendant_slot].ext_face[descendant_entry_side] = Some(root_copy_slot);

            Ok(())
        }

        pub(crate) fn apply_walk_down_trace(
            &mut self,
            current_primary_slot: usize,
            trace: &WalkDownTrace,
        ) -> Result<(), EmbeddingMutationError> {
            let (descendant_slot, descendant_entry_side) = match trace.outcome {
                WalkDownOutcome::DescendantFound { slot, entry_side } => (slot, entry_side),
                WalkDownOutcome::StoppingVertex { .. }
                | WalkDownOutcome::BlockedBicomp { .. }
                | WalkDownOutcome::CompletedToRoot => {
                    return Err(EmbeddingMutationError::TraceDidNotReachDescendant);
                }
            };

            self.merge_trace_frames(current_primary_slot, &trace.frames)?;
            self.embed_back_edge_to_descendant(
                trace.root_copy_slot,
                trace.root_side,
                descendant_slot,
                descendant_entry_side,
            )
        }

        pub(crate) fn apply_stopping_short_circuit(
            &mut self,
            root_copy_slot: usize,
            root_side: usize,
            mut stopping_slot: usize,
            mut stopping_entry_side: usize,
        ) {
            if self.ext_face_vertex(root_copy_slot, 1 ^ root_side) == stopping_slot {
                let previous_stopping_slot = stopping_slot;
                stopping_slot = self.ext_face_vertex(stopping_slot, stopping_entry_side);
                stopping_entry_side =
                    usize::from(self.ext_face_vertex(stopping_slot, 0) == previous_stopping_slot);
            }

            self.slots[root_copy_slot].ext_face[root_side] = Some(stopping_slot);
            self.slots[stopping_slot].ext_face[stopping_entry_side] = Some(root_copy_slot);
        }

        #[allow(clippy::too_many_lines, clippy::uninlined_format_args)]
        pub(crate) fn walk_down_child(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            root_copy_slot: usize,
            mode: super::EmbeddingRunMode,
        ) -> Result<WalkDownChildOutcome, WalkDownExecutionError> {
            let mut frames = Vec::new();
            let child_primary_slot = self.dfs_child_from_root(root_copy_slot);
            let next_child_primary_slot =
                self.next_dfs_child(current_primary_slot, child_primary_slot);
            for root_side in 0..2 {
                let mut current_entry_side = 1 ^ root_side;
                let mut current_slot =
                    self.walk_ext_face_neighbor(mode, root_copy_slot, &mut current_entry_side);

                'walkdown: while current_slot != root_copy_slot {
                    if self.slots[current_slot].pertinent_edge.is_some() {
                        if mode == super::EmbeddingRunMode::K33Search {
                            if let Some((merge_blocker_slot, u_max, merge_root_copy_slot)) = self
                                .find_k33_merge_blocker(
                                    current_primary_slot,
                                    current_slot,
                                    root_copy_slot,
                                    &frames,
                                )
                            {
                                if self.probe_k33_merge_blocker(
                                    preprocessing,
                                    merge_blocker_slot,
                                    u_max,
                                    merge_root_copy_slot,
                                )? {
                                    return Ok(WalkDownChildOutcome::K33Found);
                                }
                            }
                        }
                        if !frames.is_empty() {
                            self.merge_trace_frames(current_primary_slot, &frames)?;
                            frames.clear();
                        }
                        self.embed_back_edge_to_descendant(
                            root_copy_slot,
                            root_side,
                            current_slot,
                            current_entry_side,
                        )?;
                        continue;
                    }

                    if let Some(root_to_descend) =
                        self.slots[current_slot].pertinent_roots.first().copied()
                    {
                        match self.resolve_pertinent_root_walk_action(
                            preprocessing,
                            current_primary_slot,
                            root_copy_slot,
                            root_side,
                            current_slot,
                            current_entry_side,
                            root_to_descend,
                            &frames,
                            mode,
                        )? {
                            PertinentRootWalkAction::Descend {
                                next_slot,
                                next_entry_side,
                                chosen_root_side,
                            } => {
                                frames.push(WalkDownFrame {
                                    cut_vertex_slot: current_slot,
                                    cut_vertex_entry_side: current_entry_side,
                                    root_copy_slot: root_to_descend,
                                    root_side: chosen_root_side,
                                });
                                current_slot = next_slot;
                                current_entry_side = next_entry_side;
                                continue;
                            }
                            PertinentRootWalkAction::ContinueWalkdown => continue 'walkdown,
                            PertinentRootWalkAction::Return(outcome) => return Ok(outcome),
                        }
                    }

                    self.update_future_pertinent_child(current_slot, current_primary_slot);
                    if self.is_future_pertinent(current_slot, current_primary_slot)
                        || matches!(
                            mode,
                            super::EmbeddingRunMode::Outerplanarity
                                | super::EmbeddingRunMode::K23Search
                                | super::EmbeddingRunMode::K4Search
                        )
                    {
                        self.apply_stopping_short_circuit(
                            root_copy_slot,
                            root_side,
                            current_slot,
                            current_entry_side,
                        );
                        break;
                    }

                    current_slot =
                        self.walk_ext_face_neighbor(mode, current_slot, &mut current_entry_side);
                }
            }

            let child_forward_arc_head = self.child_subtree_forward_arc_head(
                preprocessing,
                current_primary_slot,
                child_primary_slot,
                next_child_primary_slot,
            );
            if let Some(forward_arc) = child_forward_arc_head {
                let action = self.resolve_child_subtree_action(
                    preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    child_primary_slot,
                    next_child_primary_slot,
                    mode,
                )?;
                return self.finalize_child_subtree_action(
                    preprocessing,
                    current_primary_slot,
                    child_primary_slot,
                    next_child_primary_slot,
                    forward_arc,
                    action,
                );
            }
            self.advance_forward_arc_list(
                preprocessing,
                current_primary_slot,
                child_primary_slot,
                next_child_primary_slot,
            );
            Ok(WalkDownChildOutcome::Completed)
        }

        fn mark_external_face_primary_vertices(
            &self,
            start_slot: usize,
            visited_primaries: &mut [bool],
            visited_slots: &mut [bool],
        ) {
            let mut stack = vec![start_slot];

            while let Some(slot) = stack.pop() {
                if visited_slots[slot] {
                    continue;
                }
                visited_slots[slot] = true;

                match self.slots[slot].kind {
                    EmbeddingSlotKind::Primary { .. } => {
                        visited_primaries[slot] = true;
                        for &child_primary_slot in &self.slots[slot].sorted_dfs_children {
                            if let Some(root_copy_slot) =
                                self.root_copy_by_primary_dfi[child_primary_slot]
                            {
                                if !visited_slots[root_copy_slot] {
                                    stack.push(root_copy_slot);
                                }
                            }
                        }
                    }
                    EmbeddingSlotKind::RootCopy { parent_primary_slot, .. } => {
                        visited_primaries[parent_primary_slot] = true;
                        if !visited_slots[parent_primary_slot] {
                            stack.push(parent_primary_slot);
                        }
                    }
                }

                for next_slot in self.slots[slot].ext_face.into_iter().flatten() {
                    if !visited_slots[next_slot] {
                        stack.push(next_slot);
                    }
                }
            }
        }

        pub(crate) fn all_primary_vertices_on_external_face(
            &self,
            preprocessing: &DfsPreprocessing,
        ) -> bool {
            let mut visited_primaries = vec![false; preprocessing.vertices.len()];
            let mut visited_slots = vec![false; self.slots.len()];

            for &root_original_vertex in &preprocessing.dfs_roots {
                let root_primary_slot = self.primary_slot_by_original_vertex[root_original_vertex];
                self.mark_external_face_primary_vertices(
                    root_primary_slot,
                    &mut visited_primaries,
                    &mut visited_slots,
                );
            }

            visited_primaries.into_iter().all(core::convert::identity)
        }
    }

    #[cfg(test)]
    mod tests {
        #![allow(clippy::pedantic)]
        use alloc::vec::Vec;

        use super::{
            EmbeddingArcRecord, EmbeddingMutationError, EmbeddingSlot, EmbeddingSlotKind,
            EmbeddingState,
        };
        use crate::traits::algorithms::planarity_detection::preprocessing::{
            DfsArcType, LocalSimpleGraph,
        };

        #[allow(clippy::too_many_lines)]
        fn simple_reduction_embedding() -> EmbeddingState {
            EmbeddingState {
                slots: vec![
                    EmbeddingSlot {
                        kind: EmbeddingSlotKind::Primary { original_vertex: 0 },
                        first_arc: Some(0),
                        last_arc: Some(0),
                        ext_face: [Some(1), Some(1)],
                        visited: false,
                        visited_info: 0,
                        pertinent_roots: Vec::new(),
                        future_pertinent_child: None,
                        pertinent_edge: None,
                        k33_merge_blocker: None,
                        k33_minor_e_reduced: false,
                        sorted_dfs_children: Vec::new(),
                        separated_dfs_children: Vec::new(),
                    },
                    EmbeddingSlot {
                        kind: EmbeddingSlotKind::Primary { original_vertex: 1 },
                        first_arc: Some(1),
                        last_arc: Some(2),
                        ext_face: [Some(0), Some(2)],
                        visited: false,
                        visited_info: 0,
                        pertinent_roots: Vec::new(),
                        future_pertinent_child: None,
                        pertinent_edge: None,
                        k33_merge_blocker: None,
                        k33_minor_e_reduced: false,
                        sorted_dfs_children: Vec::new(),
                        separated_dfs_children: Vec::new(),
                    },
                    EmbeddingSlot {
                        kind: EmbeddingSlotKind::Primary { original_vertex: 2 },
                        first_arc: Some(3),
                        last_arc: Some(3),
                        ext_face: [Some(1), Some(1)],
                        visited: false,
                        visited_info: 0,
                        pertinent_roots: Vec::new(),
                        future_pertinent_child: None,
                        pertinent_edge: None,
                        k33_merge_blocker: None,
                        k33_minor_e_reduced: false,
                        sorted_dfs_children: Vec::new(),
                        separated_dfs_children: Vec::new(),
                    },
                ],
                arcs: vec![
                    EmbeddingArcRecord {
                        original_arc: Some(0),
                        source_slot: 0,
                        target_slot: 1,
                        twin: 1,
                        next: None,
                        prev: None,
                        visited: false,
                        kind: DfsArcType::Child,
                        embedded: true,
                        inverted: false,
                        reduction_endpoint_arc: None,
                    },
                    EmbeddingArcRecord {
                        original_arc: Some(1),
                        source_slot: 1,
                        target_slot: 0,
                        twin: 0,
                        next: Some(2),
                        prev: None,
                        visited: false,
                        kind: DfsArcType::Parent,
                        embedded: true,
                        inverted: false,
                        reduction_endpoint_arc: None,
                    },
                    EmbeddingArcRecord {
                        original_arc: Some(2),
                        source_slot: 1,
                        target_slot: 2,
                        twin: 3,
                        next: None,
                        prev: Some(1),
                        visited: false,
                        kind: DfsArcType::Child,
                        embedded: true,
                        inverted: false,
                        reduction_endpoint_arc: None,
                    },
                    EmbeddingArcRecord {
                        original_arc: Some(3),
                        source_slot: 2,
                        target_slot: 1,
                        twin: 2,
                        next: None,
                        prev: None,
                        visited: false,
                        kind: DfsArcType::Parent,
                        embedded: true,
                        inverted: false,
                        reduction_endpoint_arc: None,
                    },
                ],
                primary_slot_by_original_vertex: vec![0, 1, 2],
                root_copy_by_primary_dfi: vec![None, None, None],
                least_ancestor_by_primary_slot: vec![0, 1, 2],
                lowpoint_by_primary_slot: vec![0, 1, 2],
                forward_arc_head_index_by_primary_slot: vec![None, None, None],
                handling_k4_blocked_bicomp: false,
                k4_reblocked_same_root: false,
            }
        }

        #[test]
        fn test_reduce_external_face_path_to_edge_round_trips() {
            let mut embedding = simple_reduction_embedding();

            let reduction_arc = embedding
                .reduce_external_face_path_to_edge(
                    0,
                    0,
                    2,
                    1,
                    DfsArcType::Child,
                    DfsArcType::Parent,
                )
                .unwrap()
                .expect("path 0-1-2 should reduce to a synthetic edge");
            let reduction_twin = embedding.arcs[reduction_arc].twin;

            assert_eq!(embedding.slots[0].first_arc, Some(reduction_arc));
            assert_eq!(embedding.slots[0].last_arc, Some(reduction_arc));
            assert_eq!(embedding.slots[2].first_arc, Some(reduction_twin));
            assert_eq!(embedding.slots[2].last_arc, Some(reduction_twin));
            assert_eq!(embedding.slots[1].first_arc, None);
            assert_eq!(embedding.slots[1].last_arc, None);
            assert_eq!(embedding.arcs[reduction_arc].target_slot, 2);
            assert_eq!(embedding.arcs[reduction_arc].reduction_endpoint_arc, Some(0));
            assert_eq!(embedding.arcs[reduction_twin].reduction_endpoint_arc, Some(3));

            assert!(embedding.restore_reduced_path_edge(reduction_arc));

            assert_eq!(embedding.slots[0].first_arc, Some(0));
            assert_eq!(embedding.slots[0].last_arc, Some(0));
            assert_eq!(embedding.slots[1].first_arc, Some(1));
            assert_eq!(embedding.slots[1].last_arc, Some(2));
            assert_eq!(embedding.slots[2].first_arc, Some(3));
            assert_eq!(embedding.slots[2].last_arc, Some(3));
            assert_eq!(embedding.arcs[1].next, Some(2));
            assert_eq!(embedding.arcs[2].prev, Some(1));
        }

        #[test]
        fn test_clear_bicomp_search_state_resets_slot_and_arc_flags() {
            let mut embedding = simple_reduction_embedding();
            let reset_value = embedding.primary_slot_by_original_vertex.len();

            for slot in &mut embedding.slots {
                slot.visited = true;
                slot.visited_info = usize::MAX;
            }
            for arc in &mut embedding.arcs {
                arc.visited = true;
            }

            embedding.clear_bicomp_search_state(0);

            for slot in &embedding.slots {
                assert!(!slot.visited);
                assert_eq!(slot.visited_info, reset_value);
            }
            for arc in &embedding.arcs {
                assert!(!arc.visited);
            }
        }

        #[test]
        fn test_clear_all_visited_flags_in_full_bicomp_resets_arc_twins() {
            let mut embedding = simple_reduction_embedding();

            for slot in &mut embedding.slots {
                slot.visited = true;
            }
            for arc in &mut embedding.arcs {
                arc.visited = true;
            }

            embedding.clear_all_visited_flags_in_full_bicomp(0);

            for slot in &embedding.slots {
                assert!(!slot.visited);
            }
            for arc in &embedding.arcs {
                assert!(!arc.visited);
            }
        }

        #[test]
        fn test_normalize_slot_boundary_arcs_recovers_last_arc_from_first_only() {
            let mut embedding = simple_reduction_embedding();
            embedding.slots[1].last_arc = None;

            embedding.normalize_slot_boundary_arcs(1);

            assert_eq!(embedding.slots[1].first_arc, Some(1));
            assert_eq!(embedding.slots[1].last_arc, Some(2));
        }

        #[test]
        fn test_normalize_slot_boundary_arcs_recovers_first_arc_from_last_only() {
            let mut embedding = simple_reduction_embedding();
            embedding.slots[1].first_arc = None;

            embedding.normalize_slot_boundary_arcs(1);

            assert_eq!(embedding.slots[1].first_arc, Some(1));
            assert_eq!(embedding.slots[1].last_arc, Some(2));
        }

        #[test]
        fn test_find_pertinent_vertex_between_active_sides_parallel_finds_middle_slot() {
            let mut embedding = simple_reduction_embedding();
            embedding.slots[1].pertinent_edge = Some(0);

            assert_eq!(
                embedding.find_pertinent_vertex_between_active_sides_parallel(0, 0, 2, 1),
                Some(1)
            );
        }

        #[test]
        fn test_reduce_path_to_edge_by_endpoint_arcs_handles_direct_edge() {
            let mut embedding = simple_reduction_embedding();
            embedding.slots[0].ext_face = [None, None];
            embedding.slots[1].ext_face = [None, None];

            let reduction = embedding.reduce_path_to_edge_by_endpoint_arcs(0, 0, 1, 1).unwrap();

            assert_eq!(reduction, None);
            assert_eq!(embedding.arcs.len(), 4);
            assert_eq!(embedding.slots[0].ext_face, [Some(1), Some(1)]);
            assert!(embedding.slots[1].ext_face.contains(&Some(0)));
        }

        #[test]
        fn test_reduce_path_to_edge_by_endpoint_arcs_creates_synthetic_edge() {
            let mut embedding = simple_reduction_embedding();

            let reduction_arc = embedding
                .reduce_path_to_edge_by_endpoint_arcs(0, 0, 2, 3)
                .unwrap()
                .expect("path 0-1-2 should reduce to a synthetic edge");
            let reduction_twin = embedding.arcs[reduction_arc].twin;

            assert_eq!(embedding.slots[0].first_arc, Some(reduction_arc));
            assert_eq!(embedding.slots[0].last_arc, Some(reduction_arc));
            assert_eq!(embedding.slots[2].first_arc, Some(reduction_twin));
            assert_eq!(embedding.slots[2].last_arc, Some(reduction_twin));
            assert_eq!(embedding.slots[1].first_arc, None);
            assert_eq!(embedding.slots[1].last_arc, None);
            assert_eq!(embedding.arcs[reduction_arc].target_slot, 2);
            assert_eq!(embedding.arcs[reduction_arc].reduction_endpoint_arc, Some(0));
            assert_eq!(embedding.arcs[reduction_twin].reduction_endpoint_arc, Some(3));
        }

        #[test]
        fn test_mark_tree_path_slots_visited_errors_on_missing_step() {
            let mut embedding = simple_reduction_embedding();

            assert_eq!(
                embedding.mark_tree_path_slots_visited(&[0, 2]),
                Err(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: 0,
                    start_side: 0,
                    end_slot: 2,
                })
            );
        }

        #[test]
        fn test_mark_current_dfs_path_in_bicomp_errors_without_parent_arc() {
            let mut embedding = simple_reduction_embedding();
            embedding.slots[2].first_arc = None;
            embedding.slots[2].last_arc = None;

            assert_eq!(
                embedding.mark_current_dfs_path_in_bicomp(0, 2),
                Err(EmbeddingMutationError::MissingExternalFacePath {
                    start_slot: 0,
                    start_side: 0,
                    end_slot: 2,
                })
            );
        }

        #[test]
        fn test_cumulative_orientation_on_tree_path_accumulates_child_inversions() {
            let mut embedding = simple_reduction_embedding();
            embedding.arcs[2].inverted = true;

            assert_eq!(embedding.cumulative_orientation_on_tree_path(0, 2), Ok(true));
        }

        #[test]
        fn test_choose_root_descent_can_prefer_y_side() {
            let preprocessing =
                LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
            let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

            embedding.slots[3].ext_face = [Some(1), Some(2)];
            embedding.slots[1].ext_face = [Some(0), Some(3)];
            embedding.slots[2].ext_face = [Some(3), Some(0)];
            embedding.slots[2].pertinent_edge = Some(0);

            assert_eq!(embedding.choose_root_descent(0, 3), Some((2, 0, 1)));
        }

        #[test]
        fn test_choose_root_descent_can_fall_back_to_future_pertinent_y_side() {
            let preprocessing =
                LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
            let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

            embedding.slots[3].ext_face = [Some(1), Some(2)];
            embedding.slots[1].ext_face = [Some(0), Some(3)];
            embedding.slots[2].ext_face = [Some(3), Some(0)];
            embedding.slots[2].pertinent_edge = Some(0);
            embedding.slots[2].future_pertinent_child = Some(2);

            assert_eq!(embedding.choose_root_descent(3, 3), Some((2, 0, 1)));
        }

        #[test]
        fn test_merge_vertex_handles_cut_vertex_without_incident_entry_arc() {
            let preprocessing = LocalSimpleGraph::from_edges(2, &[[0, 1]]).unwrap().preprocess();
            let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

            assert!(embedding.slots[0].first_arc.is_none());
            assert!(embedding.slots[0].last_arc.is_none());

            embedding.merge_vertex(0, 0, 2).unwrap();

            assert!(embedding.slots[0].first_arc.is_some());
            assert!(embedding.slots[0].last_arc.is_some());
            assert_eq!(embedding.slots[2].first_arc, None);
            assert_eq!(embedding.slots[2].last_arc, None);
            assert_eq!(embedding.slots[2].ext_face, [None, None]);
        }

        #[test]
        fn test_mark_external_face_primary_vertices_from_root_copy_marks_parent() {
            let preprocessing = LocalSimpleGraph::from_edges(2, &[[0, 1]]).unwrap().preprocess();
            let embedding = EmbeddingState::from_preprocessing(&preprocessing);
            let mut visited_primaries = vec![false; preprocessing.vertices.len()];
            let mut visited_slots = vec![false; embedding.slots.len()];

            embedding.mark_external_face_primary_vertices(
                2,
                &mut visited_primaries,
                &mut visited_slots,
            );

            assert!(visited_slots[2]);
            assert!(visited_slots[0]);
            assert!(visited_primaries[0]);
            assert!(visited_primaries[1]);
        }

        #[test]
        fn test_find_nonplanarity_descendant_bicomp_root_returns_trace_frame_root() {
            let preprocessing =
                LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2], [0, 2]]).unwrap().preprocess();
            let forward_arc_to_two = preprocessing.vertices[0].sorted_forward_arcs[0];
            let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

            embedding.walk_up(0, forward_arc_to_two);

            assert_eq!(embedding.find_nonplanarity_descendant_bicomp_root(0, 3, &[0]).unwrap(), 4);
        }

        #[test]
        fn test_find_nonplanarity_descendant_bicomp_root_errors_without_descendant() {
            let preprocessing =
                LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
            let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

            assert_eq!(
                embedding.find_nonplanarity_descendant_bicomp_root(0, 3, &[0, 1]),
                Err(super::K33ContextInitFailure::NoDescendantBicompRoot)
            );
        }

        #[test]
        fn test_parent_primary_slot_returns_parent_for_non_root_primary() {
            let preprocessing =
                LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
            let embedding = EmbeddingState::from_preprocessing(&preprocessing);

            assert_eq!(embedding.parent_primary_slot(1), Some(0));
            assert_eq!(embedding.parent_primary_slot(2), Some(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        child_subtree_has_blocking_forward_arc_head,
        embedding::{
            EmbeddingMutationError, EmbeddingSlotKind, EmbeddingState, WalkDownChildOutcome,
            WalkDownExecutionError, WalkDownOutcome,
        },
        preprocessing::{DfsArcType, LocalSimpleGraph, LocalSimpleGraphError},
        run_k4_homeomorph_engine, run_k33_homeomorph_engine, run_planarity_engine,
    };
    use crate::{
        impls::{CSR2D, SortedVec, SymmetricCSR2D},
        prelude::*,
        traits::{EdgesBuilder, PlanarityError, VocabularyBuilder},
    };

    fn arc_kind(
        graph: &super::preprocessing::DfsPreprocessing,
        source: usize,
        target: usize,
    ) -> DfsArcType {
        graph.adjacency_arcs[source]
            .iter()
            .map(|&arc_id| &graph.arcs[arc_id])
            .find(|arc| arc.target == target)
            .unwrap()
            .kind
    }

    fn run_engine_on_edges(node_count: usize, edges: &[[usize; 2]]) -> bool {
        let preprocessing = LocalSimpleGraph::from_edges(node_count, edges).unwrap().preprocess();
        run_planarity_engine(&preprocessing)
    }

    fn run_k33_engine_on_edges(node_count: usize, edges: &[[usize; 2]]) -> bool {
        let preprocessing = LocalSimpleGraph::from_edges(node_count, edges).unwrap().preprocess();
        run_k33_homeomorph_engine(&preprocessing)
    }

    fn run_k23_engine_on_edges(node_count: usize, edges: &[[usize; 2]]) -> bool {
        let preprocessing = LocalSimpleGraph::from_edges(node_count, edges).unwrap().preprocess();
        super::run_k23_homeomorph_engine(&preprocessing)
    }

    #[allow(clippy::too_many_lines)]
    fn run_engine_stepwise(
        node_count: usize,
        edges: &[[usize; 2]],
    ) -> Result<(), alloc::string::String> {
        let preprocessing = LocalSimpleGraph::from_edges(node_count, edges).unwrap().preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => {
                    return Err(alloc::string::String::from(
                        "encountered a root copy in the primary slot range",
                    ));
                }
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::Planarity,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => {
                        return Err(alloc::format!(
                            "walk_down_child unexpectedly reported K23Found at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                        ));
                    }
                    Ok(WalkDownChildOutcome::K33Found) => {
                        return Err(alloc::format!(
                            "walk_down_child unexpectedly reported K33Found at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                        ));
                    }
                    Ok(WalkDownChildOutcome::K4Found) => {
                        return Err(alloc::format!(
                            "walk_down_child unexpectedly reported K4Found at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                        ));
                    }
                    Err(error) => {
                        return match error {
                            WalkDownExecutionError::BlockedBicomp { context } => {
                                let blocked_root = context.blocked_root_copy_slot;
                                let left = embedding.slots[blocked_root].ext_face[0].map_or_else(
                                || alloc::string::String::from("none"),
                                |slot| {
                                    alloc::format!(
                                        "slot={slot} kind={:?} pertinent_edge={:?} pertinent_roots={:?} future_child={:?}",
                                        embedding.slots[slot].kind,
                                        embedding.slots[slot].pertinent_edge,
                                        embedding.slots[slot].pertinent_roots,
                                        embedding.slots[slot].future_pertinent_child,
                                    )
                                },
                            );
                                let right = embedding.slots[blocked_root].ext_face[1].map_or_else(
                                || alloc::string::String::from("none"),
                                |slot| {
                                    alloc::format!(
                                        "slot={slot} kind={:?} pertinent_edge={:?} pertinent_roots={:?} future_child={:?}",
                                        embedding.slots[slot].kind,
                                        embedding.slots[slot].pertinent_edge,
                                        embedding.slots[slot].pertinent_roots,
                                        embedding.slots[slot].future_pertinent_child,
                                    )
                                },
                            );
                                Err(alloc::format!(
                                    "blocked bicomp at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}, blocked_root={blocked_root}, blocked_root_kind={:?}, walk_root_copy={}, walk_root_side={}, cut_vertex_slot={}, cut_vertex_entry_side={}, left={left}, right={right}",
                                    embedding.slots[blocked_root].kind,
                                    context.walk_root_copy_slot,
                                    context.walk_root_side,
                                    context.cut_vertex_slot,
                                    context.cut_vertex_entry_side,
                                ))
                            }
                            WalkDownExecutionError::Mutation(error) => {
                                Err(alloc::format!(
                                    "walk_down_child mutation failed at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}: {error}"
                                ))
                            }
                            WalkDownExecutionError::InvalidK23Context => {
                                Err(alloc::format!(
                                    "walk_down_child unexpectedly failed K23 context init at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                                ))
                            }
                            WalkDownExecutionError::InvalidK4Context => {
                                Err(alloc::format!(
                                    "walk_down_child unexpectedly failed K4 context init at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                                ))
                            }
                            WalkDownExecutionError::InvalidK33Context => {
                                Err(alloc::format!(
                                    "walk_down_child unexpectedly failed K33 context init at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                                ))
                            }
                            WalkDownExecutionError::UnembeddedForwardArcInChildSubtree {
                                forward_arc,
                            } => {
                                Err(alloc::format!(
                                    "walk_down_child left forward_arc={forward_arc} unembedded at current={current_primary_slot}, child={child_primary_slot}, root_copy={root_copy_slot}"
                                ))
                            }
                        };
                    }
                }

                let next_child_primary_slot =
                    embedding.next_dfs_child(current_primary_slot, child_primary_slot);
                if child_subtree_has_blocking_forward_arc_head(
                    &preprocessing,
                    &embedding,
                    original_vertex,
                    child_primary_slot,
                    next_child_primary_slot,
                ) {
                    let remaining_targets = preprocessing.vertices[original_vertex]
                        .sorted_forward_arcs
                        .iter()
                        .copied()
                        .filter(|&forward_arc| !embedding.arcs[forward_arc].embedded)
                        .map(|forward_arc| {
                            alloc::format!(
                                "arc={forward_arc} target_slot={} target_kind={:?}",
                                embedding.arcs[forward_arc].target_slot,
                                embedding.slots[embedding.arcs[forward_arc].target_slot].kind,
                            )
                        })
                        .collect::<alloc::vec::Vec<_>>();
                    return Err(alloc::format!(
                        "unembedded forward arc remains at current={current_primary_slot}, child={child_primary_slot}, next_child={next_child_primary_slot:?}, remaining_targets={remaining_targets:?}"
                    ));
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_preprocessing_classifies_tree_and_back_arcs() {
        let graph = LocalSimpleGraph::from_edges(4, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3]])
            .unwrap()
            .preprocess();

        assert_eq!(arc_kind(&graph, 0, 1), DfsArcType::Child);
        assert_eq!(arc_kind(&graph, 1, 0), DfsArcType::Parent);
        assert_eq!(arc_kind(&graph, 1, 2), DfsArcType::Child);
        assert_eq!(arc_kind(&graph, 2, 1), DfsArcType::Parent);
        assert_eq!(arc_kind(&graph, 2, 3), DfsArcType::Child);
        assert_eq!(arc_kind(&graph, 3, 2), DfsArcType::Parent);
        assert_eq!(arc_kind(&graph, 2, 0), DfsArcType::Back);
        assert_eq!(arc_kind(&graph, 0, 2), DfsArcType::Forward);
        assert_eq!(arc_kind(&graph, 3, 0), DfsArcType::Back);
        assert_eq!(arc_kind(&graph, 0, 3), DfsArcType::Forward);
    }

    #[test]
    fn test_preprocessing_assigns_dfi_roots_children_and_forward_arc_order() {
        let graph =
            LocalSimpleGraph::from_edges(6, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [4, 5]])
                .unwrap()
                .preprocess();

        assert_eq!(graph.dfs_roots, vec![0, 4]);
        assert_eq!(graph.vertices[0].dfi, 0);
        assert_eq!(graph.vertices[1].dfi, 1);
        assert_eq!(graph.vertices[2].dfi, 2);
        assert_eq!(graph.vertices[3].dfi, 3);
        assert_eq!(graph.vertices[4].dfi, 4);
        assert_eq!(graph.vertices[5].dfi, 5);

        assert_eq!(graph.vertices[0].sorted_dfs_children, vec![1]);
        assert_eq!(graph.vertices[1].sorted_dfs_children, vec![2]);
        assert_eq!(graph.vertices[2].sorted_dfs_children, vec![3]);
        assert!(graph.vertices[3].sorted_dfs_children.is_empty());
        assert_eq!(graph.vertices[4].sorted_dfs_children, vec![5]);

        assert_eq!(graph.vertex_by_dfi, vec![0, 1, 2, 3, 4, 5]);

        let forward_arc_targets = graph.vertices[0]
            .sorted_forward_arcs
            .iter()
            .map(|&arc_id| graph.arcs[arc_id].target)
            .collect::<alloc::vec::Vec<_>>();
        assert_eq!(forward_arc_targets, vec![2, 3]);
        assert!(graph.vertices[1].sorted_forward_arcs.is_empty());
    }

    #[test]
    fn test_preprocessing_computes_least_ancestor_and_lowpoint() {
        let graph =
            LocalSimpleGraph::from_edges(6, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [4, 5]])
                .unwrap()
                .preprocess();

        assert_eq!(graph.vertices[0].least_ancestor, 0);
        assert_eq!(graph.vertices[0].lowpoint, 0);
        assert_eq!(graph.vertices[1].least_ancestor, 1);
        assert_eq!(graph.vertices[1].lowpoint, 0);
        assert_eq!(graph.vertices[2].least_ancestor, 0);
        assert_eq!(graph.vertices[2].lowpoint, 0);
        assert_eq!(graph.vertices[3].least_ancestor, 0);
        assert_eq!(graph.vertices[3].lowpoint, 0);
        assert_eq!(graph.vertices[4].least_ancestor, 4);
        assert_eq!(graph.vertices[4].lowpoint, 4);
        assert_eq!(graph.vertices[5].least_ancestor, 5);
        assert_eq!(graph.vertices[5].lowpoint, 5);

        assert_eq!(graph.vertices[0].parent_arc, None);
        assert_eq!(graph.vertices[1].parent_arc.map(|arc_id| graph.arcs[arc_id].target), Some(0));
        assert_eq!(graph.vertices[2].parent_arc.map(|arc_id| graph.arcs[arc_id].target), Some(1));
        assert_eq!(graph.vertices[3].parent_arc.map(|arc_id| graph.arcs[arc_id].target), Some(2));
    }

    #[test]
    fn test_preprocessing_rejects_self_loops_and_parallel_edges() {
        let self_loop = LocalSimpleGraph::from_edges(3, &[[0, 0], [0, 1]]);
        assert!(self_loop.is_err());

        let parallel_edge = LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 0]]);
        assert!(parallel_edge.is_err());
    }

    #[test]
    fn test_preprocessing_rejects_out_of_range_endpoints() {
        assert_eq!(
            LocalSimpleGraph::from_edges(3, &[[3, 1]]),
            Err(LocalSimpleGraphError::OutOfRange { endpoint: 3, node_count: 3 })
        );
        assert_eq!(
            LocalSimpleGraph::from_edges(3, &[[1, 3]]),
            Err(LocalSimpleGraphError::OutOfRange { endpoint: 3, node_count: 3 })
        );
    }

    #[test]
    fn test_preprocessing_maps_out_of_range_endpoints_to_public_error() {
        assert_eq!(
            LocalSimpleGraph::map_local_simple_graph_error(LocalSimpleGraphError::OutOfRange {
                endpoint: 3,
                node_count: 3,
            }),
            PlanarityError::InvalidEdgeEndpoint { endpoint: 3, node_count: 3 }
        );
    }

    #[test]
    fn test_local_simple_graph_builds_from_undirected_graph() {
        let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(5)
            .symbols((0..5).enumerate())
            .build()
            .unwrap();
        let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
            .expected_number_of_edges(4)
            .expected_shape(5)
            .edges([(0usize, 1usize), (1, 2), (2, 3), (3, 4)].into_iter())
            .build()
            .unwrap();
        let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));

        let preprocessing =
            LocalSimpleGraph::try_from_undirected_graph(&graph).unwrap().preprocess();

        assert_eq!(preprocessing.dfs_roots, vec![0]);
        assert_eq!(preprocessing.vertices[0].sorted_dfs_children, vec![1]);
        assert_eq!(preprocessing.vertices[1].sorted_dfs_children, vec![2]);
        assert_eq!(preprocessing.vertices[2].sorted_dfs_children, vec![3]);
        assert_eq!(preprocessing.vertices[3].sorted_dfs_children, vec![4]);
        assert!(preprocessing.vertices[4].sorted_dfs_children.is_empty());
    }

    #[test]
    fn test_embedding_initializes_singleton_bicomps_from_tree_edges() {
        let preprocessing = LocalSimpleGraph::from_edges(5, &[[0, 1], [1, 2], [2, 3], [3, 4]])
            .unwrap()
            .preprocess();
        let embedding = EmbeddingState::from_preprocessing(&preprocessing);

        assert_eq!(embedding.primary_slot_by_original_vertex, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            embedding.root_copy_by_primary_dfi,
            vec![None, Some(5), Some(6), Some(7), Some(8)]
        );
        assert_eq!(embedding.least_ancestor_by_primary_slot, vec![0, 1, 2, 3, 4]);
        assert_eq!(embedding.lowpoint_by_primary_slot, vec![0, 1, 2, 3, 4]);
        assert_eq!(embedding.slots.len(), 9);

        assert!(matches!(
            embedding.slots[0].kind,
            EmbeddingSlotKind::Primary { original_vertex: 0 }
        ));
        assert_eq!(embedding.slots[0].first_arc, None);
        assert_eq!(embedding.slots[0].last_arc, None);
        assert_eq!(embedding.slots[0].ext_face, [None, None]);
        assert_eq!(embedding.slots[0].visited_info, 5);
        assert_eq!(embedding.slots[0].future_pertinent_child, Some(1));
        assert_eq!(embedding.slots[0].sorted_dfs_children, vec![1]);
        assert!(embedding.slots[0].pertinent_roots.is_empty());
        assert_eq!(embedding.slots[0].pertinent_edge, None);

        for primary_slot in 1..5 {
            let root_copy_slot = embedding.root_copy_by_primary_dfi[primary_slot].unwrap();
            let parent_arc = embedding.slots[primary_slot].first_arc.unwrap();
            let child_arc = embedding.slots[root_copy_slot].first_arc.unwrap();

            assert_eq!(embedding.slots[primary_slot].last_arc, Some(parent_arc));
            assert_eq!(embedding.slots[root_copy_slot].last_arc, Some(child_arc));
            assert_eq!(
                embedding.slots[primary_slot].ext_face,
                [Some(root_copy_slot), Some(root_copy_slot)]
            );
            assert_eq!(
                embedding.slots[root_copy_slot].ext_face,
                [Some(primary_slot), Some(primary_slot)]
            );

            assert!(embedding.arcs[parent_arc].embedded);
            assert!(embedding.arcs[child_arc].embedded);
            assert_eq!(embedding.arcs[parent_arc].twin, child_arc);
            assert_eq!(embedding.arcs[child_arc].twin, parent_arc);
            assert_eq!(embedding.arcs[parent_arc].source_slot, primary_slot);
            assert_eq!(embedding.arcs[parent_arc].target_slot, root_copy_slot);
            assert_eq!(embedding.arcs[child_arc].source_slot, root_copy_slot);
            assert_eq!(embedding.arcs[child_arc].target_slot, primary_slot);
            assert_eq!(embedding.slots[primary_slot].visited_info, 5);
            assert!(embedding.slots[primary_slot].pertinent_roots.is_empty());
            assert_eq!(embedding.slots[primary_slot].pertinent_edge, None);

            assert!(matches!(
                embedding.slots[root_copy_slot].kind,
                EmbeddingSlotKind::RootCopy {
                    dfs_child_primary_slot,
                    ..
                } if dfs_child_primary_slot == primary_slot
            ));
        }
    }

    #[test]
    fn test_walk_up_marks_pertinent_roots_along_descendant_path() {
        let preprocessing =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3]])
                .unwrap()
                .preprocess();
        let forward_arc_to_three = preprocessing.vertices[0].sorted_forward_arcs[1];
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.walk_up(0, forward_arc_to_three);

        assert_eq!(embedding.slots[3].pertinent_edge, Some(forward_arc_to_three));
        assert_eq!(embedding.slots[3].visited_info, 0);
        assert_eq!(embedding.slots[2].visited_info, 0);
        assert_eq!(embedding.slots[1].visited_info, 0);
        assert_eq!(embedding.slots[0].visited_info, 4);

        assert_eq!(embedding.slots[2].pertinent_roots, vec![6]);
        assert_eq!(embedding.slots[1].pertinent_roots, vec![5]);
        assert_eq!(embedding.slots[0].pertinent_roots, vec![4]);
    }

    #[test]
    fn test_walk_up_reuses_existing_pertinent_root_paths_without_duplicates() {
        let preprocessing =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3]])
                .unwrap()
                .preprocess();
        let forward_arc_to_two = preprocessing.vertices[0].sorted_forward_arcs[0];
        let forward_arc_to_three = preprocessing.vertices[0].sorted_forward_arcs[1];
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.walk_up(0, forward_arc_to_three);
        embedding.walk_up(0, forward_arc_to_two);

        assert_eq!(embedding.slots[3].pertinent_edge, Some(forward_arc_to_three));
        assert_eq!(embedding.slots[2].pertinent_edge, Some(forward_arc_to_two));
        assert_eq!(embedding.slots[2].pertinent_roots, vec![6]);
        assert_eq!(embedding.slots[1].pertinent_roots, vec![5]);
        assert_eq!(embedding.slots[0].pertinent_roots, vec![4]);
    }

    #[test]
    fn test_future_pertinent_includes_least_ancestor_condition() {
        let preprocessing = LocalSimpleGraph::from_edges(4, &[[0, 1], [1, 2], [2, 3], [0, 3]])
            .unwrap()
            .preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        assert_eq!(embedding.least_ancestor_by_primary_slot, vec![0, 1, 2, 0]);
        assert!(!embedding.is_future_pertinent(3, 0));
        assert!(embedding.is_future_pertinent(3, 1));

        embedding.update_future_pertinent_child(2, 1);
        assert!(embedding.is_future_pertinent(2, 1));
    }

    #[test]
    fn test_future_pertinent_is_false_for_root_copy_slots() {
        let preprocessing =
            LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
        let embedding = EmbeddingState::from_preprocessing(&preprocessing);

        assert!(!embedding.is_future_pertinent(3, 0));
    }

    #[test]
    fn test_walk_down_trace_descends_through_pertinent_child_bicomps() {
        let preprocessing =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [1, 2], [2, 3], [0, 2], [0, 3]])
                .unwrap()
                .preprocess();
        let forward_arc_to_three = preprocessing.vertices[0].sorted_forward_arcs[1];
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.walk_up(0, forward_arc_to_three);
        let trace = embedding.walk_down_trace(0, 4, 0, false);

        assert_eq!(trace.root_copy_slot, 4);
        assert_eq!(trace.root_side, 0);
        assert_eq!(trace.visited_slots, vec![1, 2, 3]);
        assert_eq!(trace.frames.len(), 2);
        assert_eq!(trace.frames[0].cut_vertex_slot, 1);
        assert_eq!(trace.frames[0].root_copy_slot, 5);
        assert_eq!(trace.frames[0].root_side, 0);
        assert_eq!(trace.frames[1].cut_vertex_slot, 2);
        assert_eq!(trace.frames[1].root_copy_slot, 6);
        assert_eq!(trace.frames[1].root_side, 0);
        assert_eq!(trace.outcome, WalkDownOutcome::DescendantFound { slot: 3, entry_side: 1 });
    }

    #[test]
    fn test_walk_down_trace_distinguishes_inactive_vs_outerplanar_stopping() {
        let preprocessing = LocalSimpleGraph::from_edges(2, &[[0, 1]]).unwrap().preprocess();
        let mut planar_embedding = EmbeddingState::from_preprocessing(&preprocessing);
        let planar_trace = planar_embedding.walk_down_trace(0, 2, 0, false);
        assert_eq!(planar_trace.visited_slots, vec![1]);
        assert!(planar_trace.frames.is_empty());
        assert_eq!(planar_trace.outcome, WalkDownOutcome::CompletedToRoot);

        let mut outerplanar_embedding = EmbeddingState::from_preprocessing(&preprocessing);
        let outerplanar_trace = outerplanar_embedding.walk_down_trace(0, 2, 0, true);
        assert_eq!(outerplanar_trace.visited_slots, vec![1]);
        assert!(outerplanar_trace.frames.is_empty());
        assert_eq!(
            outerplanar_trace.outcome,
            WalkDownOutcome::StoppingVertex { slot: 1, entry_side: 1 }
        );
    }

    #[test]
    fn test_walk_down_trace_reports_blocked_bicomp_without_root_descent() {
        let preprocessing =
            LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.slots[1].pertinent_roots.push(4);
        let trace = embedding.walk_down_trace(0, 3, 0, false);

        assert_eq!(trace.visited_slots, vec![1]);
        assert!(trace.frames.is_empty());
        assert_eq!(trace.outcome, WalkDownOutcome::BlockedBicomp { root_copy_slot: 4 });
    }

    #[test]
    fn test_walk_down_child_reports_blocked_bicomp_context() {
        let preprocessing =
            LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2]]).unwrap().preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.slots[1].pertinent_roots.push(4);

        let context = match embedding.walk_down_child(
            &preprocessing,
            0,
            3,
            super::EmbeddingRunMode::Planarity,
        ) {
            Err(WalkDownExecutionError::BlockedBicomp { context }) => context,
            Ok(WalkDownChildOutcome::Completed) => {
                panic!("walk_down_child should have been blocked")
            }
            Ok(WalkDownChildOutcome::K23Found) => {
                panic!("walk_down_child should not report K23Found in planarity mode")
            }
            Ok(WalkDownChildOutcome::K33Found) => {
                panic!("walk_down_child should not report K33Found in planarity mode")
            }
            Ok(WalkDownChildOutcome::K4Found) => {
                panic!("walk_down_child should not report K4Found in planarity mode")
            }
            Err(WalkDownExecutionError::Mutation(error)) => {
                panic!("walk_down_child should not reach mutation path: {error}")
            }
            Err(WalkDownExecutionError::InvalidK23Context) => {
                panic!("walk_down_child should not fail K23 context init in planarity mode")
            }
            Err(WalkDownExecutionError::InvalidK4Context) => {
                panic!("walk_down_child should not fail K4 context init in planarity mode")
            }
            Err(WalkDownExecutionError::InvalidK33Context) => {
                panic!("walk_down_child should not fail K33 context init in planarity mode")
            }
            Err(WalkDownExecutionError::UnembeddedForwardArcInChildSubtree { forward_arc }) => {
                panic!(
                    "walk_down_child should not leave residual forward arc in blocked-bicomp test: {forward_arc}"
                )
            }
        };

        assert_eq!(context.current_primary_slot, 0);
        assert_eq!(context.walk_root_copy_slot, 3);
        assert_eq!(context.walk_root_side, 0);
        assert_eq!(context.cut_vertex_slot, 1);
        assert_eq!(context.cut_vertex_entry_side, 1);
        assert_eq!(context.blocked_root_copy_slot, 4);
    }

    #[test]
    fn test_apply_walk_down_trace_merges_singleton_bicomp_and_embeds_triangle_back_edge() {
        let preprocessing =
            LocalSimpleGraph::from_edges(3, &[[0, 1], [1, 2], [0, 2]]).unwrap().preprocess();
        let forward_arc_to_two = preprocessing.vertices[0].sorted_forward_arcs[0];
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        embedding.walk_up(0, forward_arc_to_two);
        let trace = embedding.walk_down_trace(0, 3, 0, false);
        assert_eq!(trace.outcome, WalkDownOutcome::DescendantFound { slot: 2, entry_side: 1 });
        assert_eq!(trace.frames.len(), 1);
        assert_eq!(trace.frames[0].cut_vertex_slot, 1);
        assert_eq!(trace.frames[0].root_copy_slot, 4);

        embedding.apply_walk_down_trace(0, &trace).unwrap();

        assert!(embedding.slots[1].pertinent_roots.is_empty());
        assert_eq!(embedding.slots[2].pertinent_edge, None);

        assert_eq!(embedding.slots[4].first_arc, None);
        assert_eq!(embedding.slots[4].last_arc, None);
        assert_eq!(embedding.slots[4].ext_face, [None, None]);

        let parent_arc_into_root = preprocessing.vertices[1].parent_arc.unwrap();
        let root_arc_to_cut_vertex = preprocessing.arcs[parent_arc_into_root].twin;
        let parent_arc_into_merged_child = preprocessing.vertices[2].parent_arc.unwrap();
        let merged_child_arc = preprocessing.arcs[parent_arc_into_merged_child].twin;
        let back_arc = preprocessing.arcs[forward_arc_to_two].twin;

        assert_eq!(embedding.slots[1].first_arc, Some(parent_arc_into_root));
        assert_eq!(embedding.slots[1].last_arc, Some(merged_child_arc));
        assert_eq!(embedding.arcs[merged_child_arc].source_slot, 1);
        assert_eq!(embedding.arcs[merged_child_arc].target_slot, 2);

        assert_eq!(embedding.slots[3].first_arc, Some(forward_arc_to_two));
        assert_eq!(embedding.slots[3].last_arc, Some(root_arc_to_cut_vertex));
        assert_eq!(embedding.arcs[forward_arc_to_two].source_slot, 3);
        assert_eq!(embedding.arcs[forward_arc_to_two].target_slot, 2);
        assert!(
            embedding.arcs[forward_arc_to_two].next == Some(root_arc_to_cut_vertex)
                || embedding.arcs[forward_arc_to_two].prev == Some(root_arc_to_cut_vertex)
        );
        assert!(
            embedding.arcs[root_arc_to_cut_vertex].next == Some(forward_arc_to_two)
                || embedding.arcs[root_arc_to_cut_vertex].prev == Some(forward_arc_to_two)
        );

        assert_eq!(embedding.slots[2].first_arc, Some(parent_arc_into_merged_child));
        assert_eq!(embedding.slots[2].last_arc, Some(back_arc));
        assert_eq!(embedding.arcs[back_arc].source_slot, 2);
        assert_eq!(embedding.arcs[back_arc].target_slot, 3);
        assert!(
            embedding.arcs[back_arc].next == Some(parent_arc_into_merged_child)
                || embedding.arcs[back_arc].prev == Some(parent_arc_into_merged_child)
        );
        assert!(
            embedding.arcs[parent_arc_into_merged_child].next == Some(back_arc)
                || embedding.arcs[parent_arc_into_merged_child].prev == Some(back_arc)
        );

        assert_eq!(embedding.slots[3].ext_face, [Some(2), Some(1)]);
        assert_eq!(embedding.slots[2].ext_face, [Some(1), Some(3)]);
    }

    #[test]
    fn test_apply_walk_down_trace_rejects_non_descendant_outcome() {
        let preprocessing = LocalSimpleGraph::from_edges(2, &[[0, 1]]).unwrap().preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);
        let trace = embedding.walk_down_trace(0, 2, 0, false);

        assert_eq!(
            embedding.apply_walk_down_trace(0, &trace),
            Err(EmbeddingMutationError::TraceDidNotReachDescendant)
        );
    }

    #[test]
    fn test_engine_accepts_diamond_k4_minus_edge_regression() {
        assert!(run_engine_on_edges(4, &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]));
    }

    #[test]
    fn test_engine_accepts_outerplanar_cycle_chords_regression() {
        let edges = [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]];
        if let Err(reason) = run_engine_stepwise(6, &edges) {
            panic!("{reason}");
        }
        assert!(run_engine_on_edges(6, &edges));
    }

    #[test]
    fn test_engine_accepts_random_planar_regression() {
        let edges = [
            [0, 8],
            [0, 11],
            [1, 2],
            [2, 3],
            [2, 5],
            [3, 6],
            [3, 7],
            [3, 8],
            [3, 10],
            [3, 11],
            [4, 6],
            [5, 10],
            [8, 9],
            [9, 11],
            [10, 11],
        ];
        if let Err(reason) = run_engine_stepwise(12, &edges) {
            panic!("{reason}");
        }
        assert!(run_engine_on_edges(12, &edges));
    }

    #[test]
    fn test_engine_accepts_erdos_renyi_corpus_regression() {
        let edges = [
            [0, 7],
            [0, 9],
            [1, 2],
            [1, 7],
            [1, 9],
            [2, 3],
            [2, 7],
            [2, 9],
            [3, 7],
            [4, 9],
            [8, 11],
        ];
        if let Err(reason) = run_engine_stepwise(12, &edges) {
            panic!("{reason}");
        }
        assert!(run_engine_on_edges(12, &edges));
    }

    #[test]
    fn test_engine_accepts_erdos_renyi_080328_corpus_regression() {
        let edges = [
            [0, 6],
            [0, 7],
            [1, 2],
            [1, 4],
            [2, 3],
            [2, 6],
            [2, 8],
            [3, 4],
            [3, 8],
            [4, 5],
            [5, 6],
            [5, 8],
        ];
        if let Err(reason) = run_engine_stepwise(9, &edges) {
            panic!("{reason}");
        }
        assert!(run_engine_on_edges(9, &edges));
    }

    #[test]
    fn test_engine_accepts_k4_subdivision_corpus_regression() {
        let edges = [
            [0, 2],
            [0, 3],
            [0, 7],
            [1, 5],
            [1, 7],
            [1, 8],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 8],
            [4, 6],
        ];
        if let Err(reason) = run_engine_stepwise(9, &edges) {
            panic!("{reason}");
        }
        assert!(run_engine_on_edges(9, &edges));
    }

    #[test]
    fn test_engine_rejects_k33_subdivision_corpus_regression() {
        let edges = [
            [0, 3],
            [0, 4],
            [0, 6],
            [1, 4],
            [1, 5],
            [1, 7],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 7],
            [5, 6],
        ];
        assert!(!run_engine_on_edges(8, &edges));
    }

    #[test]
    fn test_engine_rejects_k5_subdivision_corpus_regression() {
        let edges = [
            [0, 1],
            [0, 3],
            [0, 4],
            [0, 12],
            [1, 3],
            [1, 7],
            [1, 9],
            [2, 8],
            [2, 10],
            [2, 11],
            [2, 12],
            [3, 5],
            [3, 8],
            [4, 5],
            [4, 6],
            [4, 9],
            [6, 11],
            [7, 10],
        ];
        assert!(!run_engine_on_edges(13, &edges));
    }

    #[test]
    fn test_diamond_two_stage_merge_clears_merged_root_copy_from_cut_vertex_lists() {
        let preprocessing =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
                .unwrap()
                .preprocess();
        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        let forward_arc_from_one_to_three = preprocessing.vertices[1].sorted_forward_arcs[0];
        let root_copy_of_two = embedding.root_copy_by_primary_dfi[2].unwrap();
        embedding.walk_up(1, forward_arc_from_one_to_three);
        embedding.slots[1].pertinent_roots.clear();
        let trace_from_one = embedding.walk_down_trace(1, root_copy_of_two, 0, false);
        assert_eq!(
            trace_from_one.outcome,
            WalkDownOutcome::DescendantFound { slot: 3, entry_side: 1 }
        );
        embedding.apply_walk_down_trace(1, &trace_from_one).unwrap();

        assert!(embedding.slots[root_copy_of_two].first_arc.is_some());
        assert!(embedding.slots[1].pertinent_roots.is_empty());

        let forward_arc_from_zero_to_two = preprocessing.vertices[0].sorted_forward_arcs[0];
        let root_copy_of_one = embedding.root_copy_by_primary_dfi[1].unwrap();
        embedding.walk_up(0, forward_arc_from_zero_to_two);
        assert_eq!(embedding.slots[1].pertinent_roots, vec![root_copy_of_two]);

        let trace_from_zero = embedding.walk_down_trace(0, root_copy_of_one, 0, false);
        assert!(matches!(
            trace_from_zero.outcome,
            WalkDownOutcome::DescendantFound { slot: 2, .. }
        ));
        assert_eq!(trace_from_zero.frames.len(), 1);
        assert_eq!(trace_from_zero.frames[0].cut_vertex_slot, 1);
        assert_eq!(trace_from_zero.frames[0].root_copy_slot, root_copy_of_two);

        embedding.apply_walk_down_trace(0, &trace_from_zero).unwrap();

        assert_eq!(embedding.slots[root_copy_of_two].first_arc, None);
        assert!(embedding.slots[1].pertinent_roots.is_empty());
        for (slot_index, slot) in embedding.slots.iter().enumerate() {
            assert!(
                !slot.pertinent_roots.contains(&root_copy_of_two),
                "slot {slot_index} still retains merged root copy {root_copy_of_two} in pertinent_roots"
            );
            assert_ne!(
                slot.ext_face[0],
                Some(root_copy_of_two),
                "slot {slot_index} still points to merged root copy {root_copy_of_two} on ext_face[0]"
            );
            assert_ne!(
                slot.ext_face[1],
                Some(root_copy_of_two),
                "slot {slot_index} still points to merged root copy {root_copy_of_two} on ext_face[1]"
            );
        }

        let follow_up_trace_left = embedding.walk_down_trace(0, root_copy_of_one, 0, false);
        assert!(matches!(
            follow_up_trace_left.outcome,
            WalkDownOutcome::CompletedToRoot
                | WalkDownOutcome::StoppingVertex { .. }
                | WalkDownOutcome::DescendantFound { .. }
                | WalkDownOutcome::BlockedBicomp { .. }
        ));
        let follow_up_trace_right = embedding.walk_down_trace(0, root_copy_of_one, 1, false);
        assert!(matches!(
            follow_up_trace_right.outcome,
            WalkDownOutcome::CompletedToRoot
                | WalkDownOutcome::StoppingVertex { .. }
                | WalkDownOutcome::DescendantFound { .. }
                | WalkDownOutcome::BlockedBicomp { .. }
        ));
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_000000_regression() {
        assert!(run_k33_engine_on_edges(
            13,
            &[
                [0, 1],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
                [1, 7],
                [1, 8],
                [1, 9],
                [2, 6],
                [2, 10],
                [3, 4],
                [4, 6],
                [4, 8],
                [4, 10],
                [4, 12],
                [5, 12],
                [6, 9],
                [7, 8],
                [7, 12],
                [8, 10],
                [9, 10],
                [9, 12],
                [10, 11],
                [10, 12],
                [11, 12],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_accepts_k6_complete_regression() {
        assert!(run_k33_engine_on_edges(
            6,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5],
                [4, 5],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_accepts_k33_subdivision_000024_regression() {
        assert!(run_k33_engine_on_edges(
            16,
            &[
                [0, 3],
                [0, 6],
                [0, 7],
                [1, 8],
                [1, 10],
                [1, 11],
                [2, 4],
                [2, 5],
                [2, 13],
                [3, 9],
                [3, 15],
                [4, 6],
                [4, 10],
                [5, 7],
                [5, 12],
                [8, 9],
                [11, 12],
                [13, 14],
                [14, 15],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_001692_regression() {
        assert!(run_k33_engine_on_edges(
            14,
            &[
                [0, 2],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 10],
                [0, 13],
                [1, 6],
                [1, 10],
                [1, 12],
                [2, 3],
                [2, 6],
                [2, 7],
                [2, 10],
                [3, 11],
                [3, 13],
                [5, 7],
                [5, 9],
                [5, 12],
                [5, 13],
                [6, 8],
                [7, 10],
                [8, 12],
                [11, 12],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_007290_regression() {
        assert!(run_k33_engine_on_edges(
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [0, 8],
                [1, 2],
                [1, 4],
                [1, 7],
                [2, 5],
                [2, 6],
                [3, 4],
                [3, 5],
                [3, 8],
                [4, 5],
                [5, 7],
                [5, 8],
                [7, 8],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_017523_regression() {
        assert!(run_k33_engine_on_edges(
            7,
            &[
                [0, 1],
                [0, 2],
                [0, 6],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 6],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 6],
                [4, 6],
                [5, 6],
            ]
        ));
    }

    #[test]
    fn test_k33_engine_stepwise_rejects_k5_subdivision_000008_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 3],
                [0, 5],
                [0, 7],
                [0, 10],
                [1, 3],
                [1, 6],
                [1, 11],
                [1, 12],
                [2, 9],
                [2, 11],
                [2, 13],
                [2, 14],
                [3, 13],
                [3, 15],
                [4, 10],
                [4, 12],
                [4, 14],
                [4, 15],
                [5, 6],
                [7, 8],
                [8, 9],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => {
                        panic!(
                            "unexpected K33Found at step={current_primary_slot}, child={child_primary_slot}, root={root_copy_slot}"
                        );
                    }
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_001692_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            14,
            &[
                [0, 2],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 10],
                [0, 13],
                [1, 6],
                [1, 10],
                [1, 12],
                [2, 3],
                [2, 6],
                [2, 7],
                [2, 10],
                [3, 11],
                [3, 13],
                [5, 7],
                [5, 9],
                [5, 12],
                [5, 13],
                [6, 8],
                [7, 10],
                [8, 12],
                [11, 12],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_000000_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            13,
            &[
                [0, 1],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
                [1, 7],
                [1, 8],
                [1, 9],
                [2, 6],
                [2, 10],
                [3, 4],
                [4, 6],
                [4, 8],
                [4, 10],
                [4, 12],
                [5, 12],
                [6, 9],
                [7, 8],
                [7, 12],
                [8, 10],
                [9, 10],
                [9, 12],
                [10, 11],
                [10, 12],
                [11, 12],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_000081_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            10,
            &[
                [0, 4],
                [0, 6],
                [0, 7],
                [1, 3],
                [1, 7],
                [1, 9],
                [2, 6],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 4],
                [3, 6],
                [3, 7],
                [3, 8],
                [4, 6],
                [4, 7],
                [4, 8],
                [5, 9],
                [6, 8],
                [7, 8],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_000081_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            10,
            &[
                [0, 4],
                [0, 6],
                [0, 7],
                [1, 3],
                [1, 7],
                [1, 9],
                [2, 6],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 4],
                [3, 6],
                [3, 7],
                [3, 8],
                [4, 6],
                [4, 7],
                [4, 8],
                [5, 9],
                [6, 8],
                [7, 8],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_000792_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            12,
            &[
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 10],
                [1, 2],
                [1, 3],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [2, 3],
                [2, 4],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 4],
                [3, 7],
                [3, 10],
                [4, 9],
                [4, 10],
                [5, 6],
                [5, 9],
                [6, 7],
                [6, 8],
                [6, 9],
                [6, 10],
                [7, 8],
                [7, 9],
                [7, 10],
                [9, 10],
                [10, 11],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_000792_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            12,
            &[
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 10],
                [1, 2],
                [1, 3],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [2, 3],
                [2, 4],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 4],
                [3, 7],
                [3, 10],
                [4, 9],
                [4, 10],
                [5, 6],
                [5, 9],
                [6, 7],
                [6, 8],
                [6, 9],
                [6, 10],
                [7, 8],
                [7, 9],
                [7, 10],
                [9, 10],
                [10, 11],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_001467_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 6],
                [0, 9],
                [0, 13],
                [1, 5],
                [1, 11],
                [1, 14],
                [2, 13],
                [3, 6],
                [3, 13],
                [4, 7],
                [5, 6],
                [5, 7],
                [5, 9],
                [6, 8],
                [7, 12],
                [7, 14],
                [8, 9],
                [8, 11],
                [8, 15],
                [9, 11],
                [9, 14],
                [10, 11],
                [11, 13],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_001467_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 6],
                [0, 9],
                [0, 13],
                [1, 5],
                [1, 11],
                [1, 14],
                [2, 13],
                [3, 6],
                [3, 13],
                [4, 7],
                [5, 6],
                [5, 7],
                [5, 9],
                [6, 8],
                [7, 12],
                [7, 14],
                [8, 9],
                [8, 11],
                [8, 15],
                [9, 11],
                [9, 14],
                [10, 11],
                [11, 13],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_002340_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [1, 4],
                [3, 4],
                [3, 5],
                [3, 6],
                [3, 7],
                [4, 5],
                [4, 6],
                [4, 7],
                [4, 8],
                [5, 6],
                [5, 7],
                [5, 8],
                [6, 7],
                [6, 8],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_002340_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [1, 4],
                [3, 4],
                [3, 5],
                [3, 6],
                [3, 7],
                [4, 5],
                [4, 6],
                [4, 7],
                [4, 8],
                [5, 6],
                [5, 7],
                [5, 8],
                [6, 7],
                [6, 8],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_007290_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [0, 8],
                [1, 2],
                [1, 4],
                [1, 7],
                [2, 5],
                [2, 6],
                [3, 4],
                [3, 5],
                [3, 8],
                [4, 5],
                [5, 7],
                [5, 8],
                [7, 8],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_rejects_k5_subdivision_000017_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 2],
                [0, 3],
                [0, 5],
                [0, 8],
                [1, 7],
                [1, 9],
                [1, 10],
                [1, 11],
                [2, 9],
                [2, 12],
                [2, 14],
                [3, 10],
                [3, 13],
                [3, 15],
                [4, 8],
                [4, 11],
                [4, 14],
                [4, 15],
                [5, 6],
                [6, 7],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                let outcome = embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                );
                match outcome {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => {
                        panic!(
                            "unexpected K33 at step={current_primary_slot}, child={child_primary_slot}"
                        );
                    }
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_k33_engine_stepwise_rejects_k5_subdivision_000863_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [1, 7],
                [1, 8],
                [1, 9],
                [2, 7],
                [2, 10],
                [2, 11],
                [3, 6],
                [3, 8],
                [3, 10],
                [3, 14],
                [4, 9],
                [4, 13],
                [4, 15],
                [5, 6],
                [11, 12],
                [12, 13],
                [14, 15],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                let outcome = embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                );
                match outcome {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => {
                        panic!(
                            "unexpected K33 at step={current_primary_slot}, child={child_primary_slot}"
                        );
                    }
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_007290_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [0, 8],
                [1, 2],
                [1, 4],
                [1, 7],
                [2, 5],
                [2, 6],
                [3, 4],
                [3, 5],
                [3, 8],
                [4, 5],
                [5, 7],
                [5, 8],
                [7, 8],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_017523_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            7,
            &[
                [0, 1],
                [0, 2],
                [0, 6],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 6],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 6],
                [4, 6],
                [5, 6],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_k33_subdivision_000024_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 3],
                [0, 6],
                [0, 7],
                [1, 8],
                [1, 10],
                [1, 11],
                [2, 4],
                [2, 5],
                [2, 13],
                [3, 9],
                [3, 15],
                [4, 6],
                [4, 10],
                [5, 7],
                [5, 12],
                [8, 9],
                [11, 12],
                [13, 14],
                [14, 15],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                let outcome = embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                );
                match outcome {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_rejects_wheel_seven_regression() {
        assert!(!run_k33_engine_on_edges(
            7,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [1, 6],
            ]
        ));
    }

    #[test]
    fn test_k23_engine_accepts_k5_subdivision_000296_regression() {
        assert!(run_k23_engine_on_edges(
            16,
            &[
                [0, 1],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 3],
                [1, 4],
                [1, 9],
                [2, 4],
                [2, 5],
                [2, 10],
                [2, 11],
                [3, 4],
                [3, 6],
                [3, 15],
                [4, 8],
                [7, 8],
                [9, 10],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
            ]
        ));
    }

    #[test]
    fn test_k23_engine_accepts_erdos_renyi_002961_regression() {
        assert!(run_k23_engine_on_edges(
            13,
            &[
                [0, 5],
                [0, 10],
                [2, 9],
                [3, 4],
                [3, 8],
                [4, 9],
                [6, 7],
                [6, 10],
                [6, 11],
                [6, 12],
                [8, 9],
                [8, 10],
                [8, 11],
                [8, 12],
                [9, 10],
                [10, 11],
            ]
        ));
    }

    #[test]
    fn test_k23_engine_accepts_erdos_renyi_009567_regression() {
        assert!(run_k23_engine_on_edges(
            10,
            &[
                [0, 3],
                [0, 7],
                [1, 2],
                [1, 4],
                [1, 6],
                [1, 7],
                [2, 3],
                [2, 4],
                [2, 5],
                [2, 6],
                [2, 8],
                [3, 6],
                [3, 9],
                [4, 5],
                [5, 7],
                [5, 8],
                [5, 9],
            ]
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_k23_engine_stepwise_accepts_erdos_renyi_000306_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            10,
            &[
                [0, 4],
                [0, 5],
                [0, 7],
                [1, 9],
                [2, 5],
                [2, 6],
                [2, 7],
                [3, 4],
                [3, 6],
                [3, 7],
                [3, 8],
                [5, 9],
                [6, 9],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K23Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => return,
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K23");
    }

    #[test]
    fn test_k23_engine_stepwise_accepts_erdos_renyi_0089600_combined_reference_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            9,
            &[
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 6],
                [0, 7],
                [1, 2],
                [1, 6],
                [1, 8],
                [2, 6],
                [3, 7],
                [3, 8],
                [4, 5],
                [4, 7],
                [5, 6],
                [5, 7],
                [6, 7],
                [7, 8],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K23Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => return,
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K23");
    }

    #[test]
    fn test_k23_engine_stepwise_accepts_erdos_renyi_0293070_combined_reference_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            12,
            &[
                [0, 4],
                [0, 7],
                [0, 9],
                [0, 11],
                [1, 2],
                [1, 4],
                [1, 7],
                [2, 7],
                [3, 6],
                [3, 7],
                [3, 11],
                [4, 10],
                [5, 6],
                [5, 7],
                [5, 8],
                [5, 11],
                [6, 7],
                [7, 8],
                [7, 9],
                [7, 11],
                [9, 10],
                [9, 11],
                [10, 11],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K23Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => return,
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K23");
    }

    #[test]
    fn test_k33_engine_rejects_k5_subdivision_000008_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 3],
                [0, 5],
                [0, 7],
                [0, 10],
                [1, 3],
                [1, 6],
                [1, 11],
                [1, 12],
                [2, 9],
                [2, 11],
                [2, 13],
                [2, 14],
                [3, 13],
                [3, 15],
                [4, 10],
                [4, 12],
                [4, 14],
                [4, 15],
                [5, 6],
                [7, 8],
                [8, 9],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => {
                        panic!(
                            "false positive at step={current_primary_slot}, child={child_primary_slot}"
                        );
                    }
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_k33_engine_rejects_k5_subdivision_000017_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 2],
                [0, 3],
                [0, 5],
                [0, 8],
                [1, 7],
                [1, 9],
                [1, 10],
                [1, 11],
                [2, 9],
                [2, 12],
                [2, 14],
                [3, 10],
                [3, 13],
                [3, 15],
                [4, 8],
                [4, 11],
                [4, 14],
                [4, 15],
                [5, 6],
                [6, 7],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => {
                        panic!(
                            "false positive at step={current_primary_slot}, child={child_primary_slot}"
                        );
                    }
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_0237330_combined_reference_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            7,
            &[
                [0, 1],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 5],
                [3, 4],
                [3, 5],
                [3, 6],
                [4, 5],
                [4, 6],
                [5, 6],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_0199830_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            10,
            &[
                [0, 1],
                [0, 2],
                [0, 5],
                [0, 7],
                [1, 2],
                [1, 5],
                [1, 7],
                [2, 4],
                [2, 6],
                [2, 7],
                [4, 7],
                [4, 9],
                [5, 7],
                [5, 9],
                [6, 9],
                [7, 9],
                [8, 9],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_0199830_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            10,
            &[
                [0, 1],
                [0, 2],
                [0, 5],
                [0, 7],
                [1, 2],
                [1, 5],
                [1, 7],
                [2, 4],
                [2, 6],
                [2, 7],
                [4, 7],
                [4, 9],
                [5, 7],
                [5, 9],
                [6, 9],
                [7, 9],
                [8, 9],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_0300350_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            14,
            &[
                [0, 7],
                [0, 9],
                [0, 10],
                [0, 13],
                [1, 5],
                [1, 13],
                [2, 8],
                [2, 12],
                [3, 5],
                [3, 6],
                [3, 8],
                [4, 7],
                [4, 8],
                [4, 9],
                [4, 13],
                [5, 9],
                [8, 10],
                [8, 12],
                [8, 13],
                [9, 13],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_0300350_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            14,
            &[
                [0, 7],
                [0, 9],
                [0, 10],
                [0, 13],
                [1, 5],
                [1, 13],
                [2, 8],
                [2, 12],
                [3, 5],
                [3, 6],
                [3, 8],
                [4, 7],
                [4, 8],
                [4, 9],
                [4, 13],
                [5, 9],
                [8, 10],
                [8, 12],
                [8, 13],
                [9, 13],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_erdos_renyi_0343220_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            12,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [2, 5],
                [2, 6],
                [2, 7],
                [2, 9],
                [3, 4],
                [3, 7],
                [4, 7],
                [5, 6],
                [5, 7],
                [5, 9],
                [6, 7],
                [6, 8],
                [6, 11],
                [7, 8],
                [7, 9],
                [7, 10],
                [8, 10],
                [9, 10],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_erdos_renyi_0343220_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            12,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [2, 5],
                [2, 6],
                [2, 7],
                [2, 9],
                [3, 4],
                [3, 7],
                [4, 7],
                [5, 6],
                [5, 7],
                [5, 9],
                [6, 7],
                [6, 8],
                [6, 11],
                [7, 8],
                [7, 9],
                [7, 10],
                [8, 10],
                [9, 10],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_fuzzer_regression_20260411_direct() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 6],
                [0, 9],
                [0, 10],
                [0, 12],
                [0, 13],
                [3, 9],
                [3, 10],
                [3, 12],
                [3, 13],
                [5, 9],
                [7, 10],
                [9, 10],
                [9, 13],
                [10, 11],
                [10, 12],
                [11, 13],
                [12, 13],
                [12, 14],
                [13, 14],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_fuzzer_regression_20260411() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 6],
                [0, 9],
                [0, 10],
                [0, 12],
                [0, 13],
                [3, 9],
                [3, 10],
                [3, 12],
                [3, 13],
                [5, 9],
                [7, 10],
                [9, 10],
                [9, 13],
                [10, 11],
                [10, 12],
                [11, 13],
                [12, 13],
                [12, 14],
                [13, 14],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_fuzzer_regression_20260411_b() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 6],
                [0, 9],
                [0, 12],
                [0, 13],
                [3, 9],
                [3, 10],
                [3, 12],
                [3, 13],
                [5, 9],
                [5, 13],
                [6, 14],
                [7, 10],
                [9, 10],
                [9, 13],
                [10, 11],
                [10, 14],
                [11, 13],
                [12, 14],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k33_engine_accepts_fuzzer_regression_20260412_direct() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 1],
                [0, 7],
                [0, 11],
                [0, 15],
                [1, 5],
                [1, 11],
                [2, 4],
                [3, 5],
                [3, 10],
                [3, 15],
                [4, 5],
                [4, 12],
                [4, 13],
                [5, 6],
                [5, 7],
                [5, 10],
                [5, 11],
                [5, 12],
                [5, 15],
                [7, 8],
                [7, 11],
                [7, 15],
                [8, 9],
                [9, 15],
                [11, 12],
                [11, 15],
                [14, 15],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(!run_planarity_engine(&preprocessing));
        assert!(run_k33_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k33_engine_stepwise_accepts_fuzzer_regression_20260412() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 1],
                [0, 7],
                [0, 11],
                [0, 15],
                [1, 5],
                [1, 11],
                [2, 4],
                [3, 5],
                [3, 10],
                [3, 15],
                [4, 5],
                [4, 12],
                [4, 13],
                [5, 6],
                [5, 7],
                [5, 10],
                [5, 11],
                [5, 12],
                [5, 15],
                [7, 8],
                [7, 11],
                [7, 15],
                [8, 9],
                [9, 15],
                [11, 12],
                [11, 15],
                [14, 15],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K33Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => return,
                    Ok(WalkDownChildOutcome::K4Found) => unreachable!(),
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K33");
    }

    #[test]
    fn test_k4_engine_accepts_erdos_renyi_0025860_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 4],
                [0, 6],
                [0, 11],
                [1, 8],
                [1, 10],
                [2, 7],
                [3, 4],
                [3, 11],
                [4, 8],
                [4, 11],
                [6, 7],
                [6, 8],
                [6, 12],
                [7, 10],
                [8, 11],
                [8, 12],
                [9, 12],
                [9, 13],
                [9, 14],
                [11, 13],
                [11, 14],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(run_planarity_engine(&preprocessing));
        assert!(run_k4_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k4_engine_stepwise_accepts_erdos_renyi_0025860_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 4],
                [0, 6],
                [0, 11],
                [1, 8],
                [1, 10],
                [2, 7],
                [3, 4],
                [3, 11],
                [4, 8],
                [4, 11],
                [6, 7],
                [6, 8],
                [6, 12],
                [7, 10],
                [8, 11],
                [8, 12],
                [9, 12],
                [9, 13],
                [9, 14],
                [11, 13],
                [11, 14],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => return,
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K4");
    }

    #[test]
    fn test_k4_engine_accepts_erdos_renyi_0048200_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 5],
                [0, 7],
                [0, 10],
                [1, 7],
                [1, 14],
                [2, 4],
                [2, 14],
                [5, 10],
                [5, 13],
                [5, 15],
                [6, 9],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 9],
                [8, 10],
                [8, 14],
                [9, 11],
                [10, 13],
                [12, 14],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(run_planarity_engine(&preprocessing));
        assert!(run_k4_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k4_engine_accepts_erdos_renyi_0001210_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            7,
            &[[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 5], [2, 6], [3, 4], [4, 6], [5, 6]],
        )
        .unwrap()
        .preprocess();

        assert!(run_planarity_engine(&preprocessing));
        assert!(run_k4_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k4_engine_accepts_erdos_renyi_0397370_direct_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            14,
            &[
                [0, 4],
                [0, 5],
                [1, 5],
                [1, 6],
                [1, 9],
                [2, 4],
                [3, 4],
                [3, 9],
                [3, 10],
                [3, 13],
                [4, 10],
                [6, 7],
                [7, 8],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 11],
                [8, 13],
                [9, 13],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(run_planarity_engine(&preprocessing));
        assert!(run_k4_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k4_engine_accepts_fuzzer_regression_20260411_direct() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 3],
                [0, 10],
                [0, 11],
                [3, 10],
                [3, 11],
                [3, 12],
                [3, 13],
                [4, 12],
                [4, 14],
                [7, 12],
                [7, 14],
                [9, 10],
                [9, 13],
                [9, 14],
                [10, 11],
            ],
        )
        .unwrap()
        .preprocess();

        assert!(run_planarity_engine(&preprocessing));
        assert!(run_k4_homeomorph_engine(&preprocessing));
    }

    #[test]
    fn test_k4_engine_stepwise_accepts_fuzzer_regression_20260411() {
        let preprocessing = LocalSimpleGraph::from_edges(
            15,
            &[
                [0, 3],
                [0, 10],
                [0, 11],
                [3, 10],
                [3, 11],
                [3, 12],
                [3, 13],
                [4, 12],
                [4, 14],
                [7, 12],
                [7, 14],
                [9, 10],
                [9, 13],
                [9, 14],
                [10, 11],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => return,
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K4");
    }

    #[test]
    fn test_k4_engine_stepwise_accepts_erdos_renyi_0397370_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            14,
            &[
                [0, 4],
                [0, 5],
                [1, 5],
                [1, 6],
                [1, 9],
                [2, 4],
                [3, 4],
                [3, 9],
                [3, 10],
                [3, 13],
                [4, 10],
                [6, 7],
                [7, 8],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 11],
                [8, 13],
                [9, 13],
                [12, 13],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => return,
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K4");
    }

    #[test]
    fn test_k4_engine_stepwise_accepts_erdos_renyi_0048200_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            16,
            &[
                [0, 5],
                [0, 7],
                [0, 10],
                [1, 7],
                [1, 14],
                [2, 4],
                [2, 14],
                [5, 10],
                [5, 13],
                [5, 15],
                [6, 9],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 9],
                [8, 10],
                [8, 14],
                [9, 11],
                [10, 13],
                [12, 14],
            ],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => return,
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K4");
    }

    #[test]
    fn test_k4_engine_stepwise_accepts_erdos_renyi_0001210_regression() {
        let preprocessing = LocalSimpleGraph::from_edges(
            7,
            &[[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 5], [2, 6], [3, 4], [4, 6], [5, 6]],
        )
        .unwrap()
        .preprocess();

        let mut embedding = EmbeddingState::from_preprocessing(&preprocessing);

        for current_primary_slot in (0..preprocessing.vertices.len()).rev() {
            for slot in &mut embedding.slots {
                slot.pertinent_edge = None;
            }

            let original_vertex = match embedding.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => unreachable!(),
            };

            for &forward_arc in &preprocessing.vertices[original_vertex].sorted_forward_arcs {
                embedding.walk_up(current_primary_slot, forward_arc);
            }
            embedding.slots[current_primary_slot].pertinent_roots.clear();

            let child_count = embedding.slots[current_primary_slot].sorted_dfs_children.len();
            for child_index in 0..child_count {
                let child_primary_slot =
                    embedding.slots[current_primary_slot].sorted_dfs_children[child_index];
                if embedding.slots[child_primary_slot].pertinent_roots.is_empty() {
                    continue;
                }
                let Some(root_copy_slot) = embedding.root_copy_by_primary_dfi[child_primary_slot]
                else {
                    continue;
                };
                if embedding.slots[root_copy_slot].first_arc.is_none() {
                    continue;
                }

                match embedding.walk_down_child(
                    &preprocessing,
                    current_primary_slot,
                    root_copy_slot,
                    super::EmbeddingRunMode::K4Search,
                ) {
                    Ok(WalkDownChildOutcome::Completed) => {}
                    Ok(WalkDownChildOutcome::K23Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K33Found) => unreachable!(),
                    Ok(WalkDownChildOutcome::K4Found) => return,
                    Err(error) => {
                        panic!(
                            "unexpected error at step={current_primary_slot}, child={child_primary_slot}, error={error:?}"
                        );
                    }
                }
            }
        }

        panic!("engine completed without finding K4");
    }
}
