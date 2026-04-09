//! Submodule declaring planarity detection for simple undirected graphs.
//!
//! The current implementation builds a local simple-graph view, performs DFS
//! preprocessing, and then runs the crate's internal edge-addition embedding
//! engine. The public contract is intentionally limited to simple undirected
//! graphs, so self-loops and parallel edges are rejected.

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

fn run_planarity_engine(preprocessing: &preprocessing::DfsPreprocessing) -> bool {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingRunMode {
    Planarity,
    Outerplanarity,
    K23Search,
}

enum EmbeddingRunOutcome {
    Embedded(embedding::EmbeddingState),
    K23Found,
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
    let forward_arc = preprocessing.vertices[original_vertex]
        .sorted_forward_arcs
        .iter()
        .copied()
        .find(|&forward_arc| !embedding.arcs[forward_arc].embedded)?;
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
        pub(crate) visited_info: usize,
        pub(crate) pertinent_roots: Vec<usize>,
        pub(crate) future_pertinent_child: Option<usize>,
        pub(crate) pertinent_edge: Option<usize>,
        pub(crate) sorted_dfs_children: Vec<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct EmbeddingArcRecord {
        pub(crate) original_arc: usize,
        pub(crate) source_slot: usize,
        pub(crate) target_slot: usize,
        pub(crate) twin: usize,
        pub(crate) next: Option<usize>,
        pub(crate) prev: Option<usize>,
        pub(crate) kind: DfsArcType,
        pub(crate) embedded: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct EmbeddingState {
        pub(crate) slots: Vec<EmbeddingSlot>,
        pub(crate) arcs: Vec<EmbeddingArcRecord>,
        pub(crate) primary_slot_by_original_vertex: Vec<usize>,
        pub(crate) root_copy_by_primary_dfi: Vec<Option<usize>>,
        pub(crate) least_ancestor_by_primary_slot: Vec<usize>,
        pub(crate) lowpoint_by_primary_slot: Vec<usize>,
        pub(crate) next_forward_arc_index_by_primary_slot: Vec<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
    pub(crate) enum EmbeddingMutationError {
        #[error("walk-down trace did not end at a descendant suitable for back-edge embedding")]
        TraceDidNotReachDescendant,
        #[error("expected an embedded arc at slot {slot} side {side}")]
        MissingSlotArc { slot: usize, side: usize },
        #[error("expected slot {slot} to have a pertinent edge before back-edge insertion")]
        MissingPertinentEdge { slot: usize },
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
    struct NonOuterplanarityContext {
        current_primary_slot: usize,
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
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum K23BicompSearchOutcome {
        MinorA,
        MinorB,
        MinorE1OrE2,
        MinorE3OrE4,
        SeparableK4,
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
                    visited_info: number_of_vertices,
                    pertinent_roots: Vec::new(),
                    future_pertinent_child: None,
                    pertinent_edge: None,
                    sorted_dfs_children: Vec::new(),
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
                    visited_info: number_of_vertices,
                    pertinent_roots: Vec::new(),
                    future_pertinent_child: None,
                    pertinent_edge: None,
                    sorted_dfs_children: Vec::new(),
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
                slots[primary_slot].future_pertinent_child = sorted_dfs_children.first().copied();
                slots[primary_slot].sorted_dfs_children = sorted_dfs_children;
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
                        original_arc,
                        source_slot: primary_slot_by_original_vertex[arc.source],
                        target_slot: primary_slot_by_original_vertex[arc.target],
                        twin: arc.twin,
                        next: None,
                        prev: None,
                        kind: arc.kind,
                        embedded: false,
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
                next_forward_arc_index_by_primary_slot: vec![0; number_of_vertices],
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

        fn find_nonouterplanarity_context(
            &self,
            current_primary_slot: usize,
            bicomp_root_copy_slot: usize,
        ) -> Option<NonOuterplanarityContext> {
            let mut x_prev_link = 1usize;
            let mut y_prev_link = 0usize;
            let x_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut x_prev_link);
            let y_slot = self.real_ext_face_neighbor(bicomp_root_copy_slot, &mut y_prev_link);
            let w_slot = self.find_pertinent_vertex_on_lower_face(x_slot, x_prev_link, y_slot)?;

            Some(NonOuterplanarityContext {
                current_primary_slot,
                x_slot,
                y_slot,
                w_slot,
                x_prev_link,
                y_prev_link,
            })
        }

        fn find_pertinent_vertex_on_lower_face(
            &self,
            x_slot: usize,
            x_prev_link: usize,
            y_slot: usize,
        ) -> Option<usize> {
            let mut candidate_slot = x_slot;
            let mut previous_link = x_prev_link;

            candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            while candidate_slot != y_slot {
                if self.is_pertinent(candidate_slot) {
                    return Some(candidate_slot);
                }
                candidate_slot = self.real_ext_face_neighbor(candidate_slot, &mut previous_link);
            }

            None
        }

        pub(crate) fn search_for_k23_in_bicomp(
            &mut self,
            current_primary_slot: usize,
            walk_root_copy_slot: usize,
            blocked_root_copy_slot: usize,
        ) -> Result<K23BicompSearchOutcome, WalkDownExecutionError> {
            if blocked_root_copy_slot != walk_root_copy_slot {
                return Ok(K23BicompSearchOutcome::MinorA);
            }

            let Some(context) =
                self.find_nonouterplanarity_context(current_primary_slot, blocked_root_copy_slot)
            else {
                return Err(WalkDownExecutionError::InvalidK23Context);
            };

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
                Ok(K23BicompSearchOutcome::SeparableK4)
            }
        }

        pub(crate) fn continue_after_separable_k4(&mut self, context: &BlockedBicompContext) {
            self.slots[context.cut_vertex_slot]
                .pertinent_roots
                .retain(|&candidate| candidate != context.blocked_root_copy_slot);
            self.update_future_pertinent_child(
                context.cut_vertex_slot,
                context.current_primary_slot,
            );
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

        fn current_forward_arc_head(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
        ) -> Option<usize> {
            let original_vertex = match self.slots[current_primary_slot].kind {
                EmbeddingSlotKind::Primary { original_vertex } => original_vertex,
                EmbeddingSlotKind::RootCopy { .. } => {
                    panic!("forward-arc heads are tracked only for primary slots")
                }
            };
            let sorted_forward_arcs = &preprocessing.vertices[original_vertex].sorted_forward_arcs;
            let next_index = &mut self.next_forward_arc_index_by_primary_slot[current_primary_slot];

            while *next_index < sorted_forward_arcs.len()
                && self.arcs[sorted_forward_arcs[*next_index]].embedded
            {
                *next_index += 1;
            }

            sorted_forward_arcs.get(*next_index).copied()
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

            (descendant_primary_slot >= child_primary_slot
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
            while let Some(forward_arc) =
                self.current_forward_arc_head(preprocessing, current_primary_slot)
            {
                let descendant_primary_slot = self.arcs[forward_arc].target_slot;
                if descendant_primary_slot < child_primary_slot
                    || next_child_primary_slot
                        .is_some_and(|next_child| descendant_primary_slot >= next_child)
                {
                    break;
                }
                self.next_forward_arc_index_by_primary_slot[current_primary_slot] += 1;
            }
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
            for slot in &mut self.slots {
                for ext_face_vertex in &mut slot.ext_face {
                    if *ext_face_vertex == Some(root_copy_slot) {
                        *ext_face_vertex = Some(cut_vertex_slot);
                    }
                }
                slot.pertinent_roots.retain(|&candidate| candidate != root_copy_slot);
            }
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
        fn choose_root_descent(
            &mut self,
            current_primary_slot: usize,
            root_copy_slot: usize,
        ) -> Option<(usize, usize, usize)> {
            let x_slot = self.ext_face_vertex(root_copy_slot, 0);
            let x_prev_link = usize::from(self.ext_face_vertex(x_slot, 1) == root_copy_slot);
            let y_slot = self.ext_face_vertex(root_copy_slot, 1);
            let y_prev_link = usize::from(self.ext_face_vertex(y_slot, 0) != root_copy_slot);

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

        #[allow(clippy::similar_names)]
        pub(crate) fn walk_up(&mut self, current_primary_slot: usize, forward_arc: usize) {
            let descendant_primary_slot = self.arcs[forward_arc].target_slot;
            self.slots[descendant_primary_slot].pertinent_edge = Some(forward_arc);

            let mut zig = descendant_primary_slot;
            let mut zag = descendant_primary_slot;
            let mut zig_entry_side = 1usize;
            let mut zag_entry_side = 0usize;

            while zig != current_primary_slot {
                let zig_candidate = self.ext_face_vertex(zig, 1 ^ zig_entry_side);
                let mut root_copy_slot = None;
                let (next_zig_vertex, next_zag_vertex) = if self.is_virtual(zig_candidate) {
                    if self.slots[zig].visited_info == current_primary_slot {
                        break;
                    }
                    let root = zig_candidate;
                    root_copy_slot = Some(root);
                    let zag_vertex = self
                        .ext_face_vertex(root, usize::from(self.ext_face_vertex(root, 0) == zig));
                    if self.slots[zag_vertex].visited_info == current_primary_slot {
                        break;
                    }
                    (zig_candidate, zag_vertex)
                } else {
                    let zag_candidate = self.ext_face_vertex(zag, 1 ^ zag_entry_side);
                    if self.is_virtual(zag_candidate) {
                        if self.slots[zag].visited_info == current_primary_slot {
                            break;
                        }
                        let root = zag_candidate;
                        root_copy_slot = Some(root);
                        let zig_vertex = self.ext_face_vertex(
                            root,
                            usize::from(self.ext_face_vertex(root, 0) == zag),
                        );
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
                    zig_entry_side = usize::from(self.ext_face_vertex(next_zig_vertex, 0) != zig);
                    zig = next_zig_vertex;
                    zag_entry_side = usize::from(self.ext_face_vertex(next_zag_vertex, 0) != zag);
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

            let mut current_slot = self.ext_face_vertex(root_copy_slot, root_side);
            let mut current_entry_side =
                usize::from(self.ext_face_vertex(current_slot, 1) == root_copy_slot);

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
                    if let Some((next_slot, next_entry_side, chosen_root_side)) =
                        self.choose_root_descent(current_primary_slot, root_to_descend)
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

                let next_slot = self.ext_face_vertex(current_slot, 1 ^ current_entry_side);
                current_entry_side =
                    usize::from(self.ext_face_vertex(next_slot, 0) != current_slot);
                current_slot = next_slot;
            }

            trace
        }

        fn merge_vertex(
            &mut self,
            cut_vertex_slot: usize,
            cut_vertex_entry_side: usize,
            root_copy_slot: usize,
        ) -> Result<(), EmbeddingMutationError> {
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
                let mut root_side = frame.root_side;

                if cut_vertex_entry_side == root_side {
                    root_side = 1 ^ cut_vertex_entry_side;
                    if !self.is_singleton_slot(root_copy_slot) {
                        // Boyer inverts only the bicomp root copy here and defers
                        // orientation propagation to later bicomp handling.
                        self.invert_vertex(root_copy_slot);
                    }
                }

                let ext_face_vertex = self.ext_face_vertex(root_copy_slot, 1 ^ root_side);
                self.slots[cut_vertex_slot].ext_face[cut_vertex_entry_side] = Some(ext_face_vertex);

                if self.slots[ext_face_vertex].ext_face[0]
                    == self.slots[ext_face_vertex].ext_face[1]
                {
                    self.slots[ext_face_vertex].ext_face[root_side] = Some(cut_vertex_slot);
                } else {
                    let ext_face_side =
                        usize::from(self.ext_face_vertex(ext_face_vertex, 0) != root_copy_slot);
                    self.slots[ext_face_vertex].ext_face[ext_face_side] = Some(cut_vertex_slot);
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

        #[allow(clippy::too_many_lines)]
        pub(crate) fn walk_down_child(
            &mut self,
            preprocessing: &DfsPreprocessing,
            current_primary_slot: usize,
            root_copy_slot: usize,
            mode: super::EmbeddingRunMode,
        ) -> Result<WalkDownChildOutcome, WalkDownExecutionError> {
            loop {
                let mut made_progress = false;
                let mut restart_from_root = false;
                let mut frames = Vec::new();

                for root_side in 0..2 {
                    let mut current_slot = self.ext_face_vertex(root_copy_slot, root_side);
                    let mut current_entry_side =
                        usize::from(self.ext_face_vertex(current_slot, 1) == root_copy_slot);

                    while current_slot != root_copy_slot {
                        if self.slots[current_slot].pertinent_edge.is_some() {
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
                            made_progress = true;
                            restart_from_root = true;
                            break;
                        }

                        if let Some(root_to_descend) =
                            self.slots[current_slot].pertinent_roots.first().copied()
                        {
                            if let Some((next_slot, next_entry_side, chosen_root_side)) =
                                self.choose_root_descent(current_primary_slot, root_to_descend)
                            {
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

                            let context = BlockedBicompContext {
                                current_primary_slot,
                                walk_root_copy_slot: root_copy_slot,
                                walk_root_side: root_side,
                                cut_vertex_slot: current_slot,
                                cut_vertex_entry_side: current_entry_side,
                                blocked_root_copy_slot: root_to_descend,
                            };
                            if mode == super::EmbeddingRunMode::K23Search {
                                match self.search_for_k23_in_bicomp(
                                    current_primary_slot,
                                    root_copy_slot,
                                    root_to_descend,
                                )? {
                                    K23BicompSearchOutcome::SeparableK4 => {
                                        self.continue_after_separable_k4(&context);
                                        made_progress = true;
                                        restart_from_root = true;
                                        break;
                                    }
                                    K23BicompSearchOutcome::MinorA
                                    | K23BicompSearchOutcome::MinorB
                                    | K23BicompSearchOutcome::MinorE1OrE2
                                    | K23BicompSearchOutcome::MinorE3OrE4 => {
                                        return Ok(WalkDownChildOutcome::K23Found);
                                    }
                                }
                            }

                            return Err(WalkDownExecutionError::BlockedBicomp { context });
                        }

                        self.update_future_pertinent_child(current_slot, current_primary_slot);
                        if self.is_future_pertinent(current_slot, current_primary_slot)
                            || mode != super::EmbeddingRunMode::Planarity
                        {
                            let root_copy_ext_face_before = self.slots[root_copy_slot].ext_face;
                            self.apply_stopping_short_circuit(
                                root_copy_slot,
                                root_side,
                                current_slot,
                                current_entry_side,
                            );
                            made_progress |=
                                self.slots[root_copy_slot].ext_face != root_copy_ext_face_before;
                            break;
                        }

                        let next_slot = self.ext_face_vertex(current_slot, 1 ^ current_entry_side);
                        current_entry_side =
                            usize::from(self.ext_face_vertex(next_slot, 0) != current_slot);
                        current_slot = next_slot;
                    }

                    if restart_from_root {
                        break;
                    }
                }

                if !made_progress {
                    break;
                }
            }

            let child_primary_slot = self.dfs_child_from_root(root_copy_slot);
            let next_child_primary_slot =
                self.next_dfs_child(current_primary_slot, child_primary_slot);

            if let Some(forward_arc) = self.child_subtree_forward_arc_head(
                preprocessing,
                current_primary_slot,
                child_primary_slot,
                next_child_primary_slot,
            ) {
                if mode == super::EmbeddingRunMode::K23Search {
                    match self.search_for_k23_in_bicomp(
                        current_primary_slot,
                        root_copy_slot,
                        root_copy_slot,
                    )? {
                        K23BicompSearchOutcome::SeparableK4 => {
                            self.continue_after_same_root_separable_k4(
                                preprocessing,
                                current_primary_slot,
                                root_copy_slot,
                            );
                            return Ok(WalkDownChildOutcome::Completed);
                        }
                        K23BicompSearchOutcome::MinorA
                        | K23BicompSearchOutcome::MinorB
                        | K23BicompSearchOutcome::MinorE1OrE2
                        | K23BicompSearchOutcome::MinorE3OrE4 => {
                            return Ok(WalkDownChildOutcome::K23Found);
                        }
                    }
                }

                return Err(WalkDownExecutionError::UnembeddedForwardArcInChildSubtree {
                    forward_arc,
                });
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
        use super::EmbeddingState;
        use crate::traits::algorithms::planarity_detection::preprocessing::LocalSimpleGraph;

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
        run_planarity_engine,
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
            Err(WalkDownExecutionError::Mutation(error)) => {
                panic!("walk_down_child should not reach mutation path: {error}")
            }
            Err(WalkDownExecutionError::InvalidK23Context) => {
                panic!("walk_down_child should not fail K23 context init in planarity mode")
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
}
