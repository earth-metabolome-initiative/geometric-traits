//! Submodule providing the `BiconnectedComponents` trait and its blanket
//! implementation for undirected monopartite monoplex graphs.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::{
    MonopartiteGraph, PositiveInteger, SparseMatrix2D, UndirectedMonopartiteEdges,
    UndirectedMonopartiteMonoplexGraph,
};

/// Canonical undirected edge representation used by the biconnected-components
/// algorithm.
pub type BiconnectedEdge<NodeId> = [NodeId; 2];

/// Result of the Hopcroft-Tarjan biconnected-components decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiconnectedComponentsResult<NodeId: PositiveInteger> {
    /// The edge-partitioned biconnected components.
    edge_biconnected_components: Vec<Vec<BiconnectedEdge<NodeId>>>,
    /// The corresponding vertex views of the biconnected components.
    vertex_biconnected_components: Vec<Vec<NodeId>>,
    /// The articulation points of the graph.
    articulation_points: Vec<NodeId>,
    /// The bridges of the graph.
    bridges: Vec<BiconnectedEdge<NodeId>>,
    /// Vertices that do not belong to any edge-based biconnected component.
    vertices_without_biconnected_component: Vec<NodeId>,
    /// Indices of the cyclic biconnected components.
    cyclic_biconnected_component_ids: Vec<usize>,
    /// Number of connected components in the graph.
    number_of_connected_components: usize,
    /// Whether the full graph is biconnected.
    is_biconnected: bool,
}

impl<NodeId: PositiveInteger> BiconnectedComponentsResult<NodeId> {
    /// Returns the number of edge-based biconnected components.
    #[inline]
    #[must_use]
    pub fn number_of_biconnected_components(&self) -> usize {
        self.edge_biconnected_components.len()
    }

    /// Returns the number of connected components in the graph.
    #[inline]
    #[must_use]
    pub fn number_of_connected_components(&self) -> usize {
        self.number_of_connected_components
    }

    /// Returns whether the whole graph is biconnected.
    #[inline]
    #[must_use]
    pub fn is_biconnected(&self) -> bool {
        self.is_biconnected
    }

    /// Returns the edge-based biconnected components.
    #[inline]
    pub fn edge_biconnected_components(
        &self,
    ) -> core::slice::Iter<'_, Vec<BiconnectedEdge<NodeId>>> {
        self.edge_biconnected_components.iter()
    }

    /// Returns the vertex-based biconnected components.
    #[inline]
    pub fn vertex_biconnected_components(&self) -> core::slice::Iter<'_, Vec<NodeId>> {
        self.vertex_biconnected_components.iter()
    }

    /// Returns the edge-based biconnected component with the provided index.
    #[inline]
    #[must_use]
    pub fn edge_biconnected_component(&self, component_id: usize) -> &[BiconnectedEdge<NodeId>] {
        &self.edge_biconnected_components[component_id]
    }

    /// Returns the vertex-based biconnected component with the provided index.
    #[inline]
    #[must_use]
    pub fn vertex_biconnected_component(&self, component_id: usize) -> &[NodeId] {
        &self.vertex_biconnected_components[component_id]
    }

    /// Returns the articulation points of the graph.
    #[inline]
    pub fn articulation_points(&self) -> core::iter::Copied<core::slice::Iter<'_, NodeId>> {
        self.articulation_points.iter().copied()
    }

    /// Returns the bridges of the graph.
    #[inline]
    pub fn bridges(&self) -> core::iter::Copied<core::slice::Iter<'_, BiconnectedEdge<NodeId>>> {
        self.bridges.iter().copied()
    }

    /// Returns the vertices that are not part of any edge-based biconnected
    /// component.
    #[inline]
    pub fn vertices_without_biconnected_component(
        &self,
    ) -> core::iter::Copied<core::slice::Iter<'_, NodeId>> {
        self.vertices_without_biconnected_component.iter().copied()
    }

    /// Returns the indices of the cyclic biconnected components.
    #[inline]
    pub fn cyclic_biconnected_component_ids(
        &self,
    ) -> core::iter::Copied<core::slice::Iter<'_, usize>> {
        self.cyclic_biconnected_component_ids.iter().copied()
    }
}

type NeighborIterator<'graph, G> = <<<G as UndirectedMonopartiteMonoplexGraph>::UndirectedMonopartiteEdges as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix as SparseMatrix2D>::SparseRow<'graph>;

struct DfsFrame<'graph, G: UndirectedMonopartiteMonoplexGraph + ?Sized>
where
    <<G as UndirectedMonopartiteMonoplexGraph>::UndirectedMonopartiteEdges as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix:
        'graph,
{
    /// Current node.
    node: G::NodeId,
    /// Parent in the DFS tree.
    parent: Option<G::NodeId>,
    /// Iterator over the node neighbors.
    neighbors: NeighborIterator<'graph, G>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NormalizedComponent<NodeId: PositiveInteger> {
    /// The component edges.
    edges: Vec<BiconnectedEdge<NodeId>>,
    /// The component vertices.
    vertices: Vec<NodeId>,
}

#[inline]
fn normalize_edge<NodeId: PositiveInteger>(left: NodeId, right: NodeId) -> BiconnectedEdge<NodeId> {
    if left <= right { [left, right] } else { [right, left] }
}

#[inline]
fn pop_component_until<NodeId: PositiveInteger>(
    edge_stack: &mut Vec<BiconnectedEdge<NodeId>>,
    stop_edge: Option<BiconnectedEdge<NodeId>>,
) -> Option<Vec<BiconnectedEdge<NodeId>>> {
    let mut component = Vec::new();
    while let Some(edge) = edge_stack.pop() {
        component.push(edge);
        if Some(edge) == stop_edge {
            break;
        }
    }
    if component.is_empty() {
        None
    } else {
        component.sort_unstable();
        component.dedup();
        Some(component)
    }
}

#[inline]
fn normalize_component<NodeId: PositiveInteger>(
    mut edges: Vec<BiconnectedEdge<NodeId>>,
) -> NormalizedComponent<NodeId> {
    edges.sort_unstable();
    edges.dedup();
    let mut vertices = edges.iter().flat_map(|edge| [edge[0], edge[1]]).collect::<Vec<_>>();
    vertices.sort_unstable();
    vertices.dedup();
    NormalizedComponent { edges, vertices }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for biconnected-components decomposition.
pub enum BiconnectedComponentsError {
    /// The graph contains self-loops, which are unsupported by this simple
    /// undirected implementation.
    #[error(
        "The biconnected-components algorithm currently supports only simple undirected graphs and does not accept self-loops."
    )]
    SelfLoopsUnsupported,
    /// Parallel edges are unsupported by the algorithm contract.
    ///
    /// The current matrix-backed graph traits canonicalize duplicate edges
    /// before this algorithm sees them, so this variant exists mainly to make
    /// the unsupported-input contract explicit and forward-compatible.
    #[error(
        "The biconnected-components algorithm currently supports only simple undirected graphs and does not accept parallel edges."
    )]
    ParallelEdgesUnsupported,
}

impl From<BiconnectedComponentsError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: BiconnectedComponentsError) -> Self {
        Self::BiconnectedComponentsError(error)
    }
}

impl<G: MonopartiteGraph> From<BiconnectedComponentsError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: BiconnectedComponentsError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

/// Trait providing the Hopcroft-Tarjan biconnected-components decomposition for
/// undirected graphs.
pub trait BiconnectedComponents: UndirectedMonopartiteMonoplexGraph {
    /// Returns the edge-based biconnected components together with the derived
    /// articulation points and bridges.
    ///
    /// # Errors
    ///
    /// * If the graph contains self-loops.
    /// * Parallel edges are unsupported by the public contract, though the
    ///   current matrix-backed graph traits canonicalize duplicate edges before
    ///   this algorithm sees them.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V + E) space.
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
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// let decomposition = graph.biconnected_components().unwrap();
    /// assert_eq!(decomposition.number_of_biconnected_components(), 3);
    /// assert_eq!(decomposition.articulation_points().collect::<Vec<_>>(), vec![2, 3]);
    /// assert_eq!(decomposition.bridges().collect::<Vec<_>>(), vec![[2, 3], [3, 4]]);
    /// ```
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn biconnected_components(
        &self,
    ) -> Result<BiconnectedComponentsResult<Self::NodeId>, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
    {
        if self.has_self_loops() {
            return Err(BiconnectedComponentsError::SelfLoopsUnsupported.into());
        }

        let number_of_nodes = self.number_of_nodes().as_();
        let mut discovery_time: Vec<Option<usize>> = vec![None; number_of_nodes];
        let mut low: Vec<usize> = vec![0; number_of_nodes];
        let mut connected_component_count = 0usize;
        let mut dfs_time = 0usize;
        let mut edge_stack: Vec<BiconnectedEdge<Self::NodeId>> = Vec::new();
        let mut raw_components: Vec<Vec<BiconnectedEdge<Self::NodeId>>> = Vec::new();
        let mut dfs_stack: Vec<DfsFrame<'_, Self>> = Vec::new();

        for root in self.node_ids() {
            if discovery_time[root.as_()].is_some() {
                continue;
            }

            connected_component_count += 1;
            discovery_time[root.as_()] = Some(dfs_time);
            low[root.as_()] = dfs_time;
            dfs_time += 1;
            dfs_stack.push(DfsFrame { node: root, parent: None, neighbors: self.neighbors(root) });

            while !dfs_stack.is_empty() {
                let maybe_neighbor = {
                    let frame = dfs_stack.last_mut().unwrap();
                    frame.neighbors.next()
                };

                if let Some(neighbor) = maybe_neighbor {
                    let node = dfs_stack.last().unwrap().node;
                    let parent = dfs_stack.last().unwrap().parent;

                    if discovery_time[neighbor.as_()].is_none() {
                        discovery_time[neighbor.as_()] = Some(dfs_time);
                        low[neighbor.as_()] = dfs_time;
                        dfs_time += 1;
                        edge_stack.push(normalize_edge(node, neighbor));
                        dfs_stack.push(DfsFrame {
                            node: neighbor,
                            parent: Some(node),
                            neighbors: self.neighbors(neighbor),
                        });
                        continue;
                    }

                    if Some(neighbor) != parent
                        && discovery_time[neighbor.as_()].unwrap()
                            <= discovery_time[node.as_()].unwrap()
                    {
                        low[node.as_()] =
                            low[node.as_()].min(discovery_time[neighbor.as_()].unwrap());
                        edge_stack.push(normalize_edge(node, neighbor));
                    }
                    continue;
                }

                let frame = dfs_stack.pop().unwrap();
                let node = frame.node;
                if let Some(parent) = frame.parent {
                    let parent_index = parent.as_();
                    let node_index = node.as_();
                    low[parent_index] = low[parent_index].min(low[node_index]);

                    if low[node_index] >= discovery_time[parent_index].unwrap() {
                        if let Some(component) =
                            pop_component_until(&mut edge_stack, Some(normalize_edge(parent, node)))
                        {
                            raw_components.push(component);
                        }
                    }
                }
            }

            if let Some(component) = pop_component_until(&mut edge_stack, None) {
                raw_components.push(component);
            }
        }

        let mut components: Vec<NormalizedComponent<Self::NodeId>> =
            raw_components.into_iter().map(normalize_component).collect();
        components.sort_unstable();

        let mut edge_biconnected_components = Vec::with_capacity(components.len());
        let mut vertex_biconnected_components = Vec::with_capacity(components.len());
        let mut covered_vertices = vec![false; number_of_nodes];
        let mut articulation_memberships = vec![0usize; number_of_nodes];
        let mut bridges = Vec::new();
        let mut cyclic_biconnected_component_ids = Vec::new();

        for (component_id, component) in components.into_iter().enumerate() {
            if component.edges.len() == 1 {
                bridges.push(component.edges[0]);
            }
            if !component.edges.is_empty() && component.edges.len() >= component.vertices.len() {
                cyclic_biconnected_component_ids.push(component_id);
            }
            for &vertex in &component.vertices {
                covered_vertices[vertex.as_()] = true;
                articulation_memberships[vertex.as_()] += 1;
            }
            edge_biconnected_components.push(component.edges);
            vertex_biconnected_components.push(component.vertices);
        }

        let articulation_points: Vec<Self::NodeId> =
            self.node_ids().filter(|&vertex| articulation_memberships[vertex.as_()] > 1).collect();
        let vertices_without_biconnected_component: Vec<Self::NodeId> =
            self.node_ids().filter(|&vertex| !covered_vertices[vertex.as_()]).collect();
        let is_biconnected = number_of_nodes >= 2
            && connected_component_count == 1
            && articulation_points.is_empty();

        Ok(BiconnectedComponentsResult {
            edge_biconnected_components,
            vertex_biconnected_components,
            articulation_points,
            bridges,
            vertices_without_biconnected_component,
            cyclic_biconnected_component_ids,
            number_of_connected_components: connected_component_count,
            is_biconnected,
        })
    }

    /// Returns whether the full graph is biconnected.
    ///
    /// # Errors
    ///
    /// Returns the same errors as
    /// [`biconnected_components`](Self::biconnected_components).
    #[inline]
    fn is_biconnected(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
    {
        Ok(self.biconnected_components()?.is_biconnected())
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> BiconnectedComponents for G {}
