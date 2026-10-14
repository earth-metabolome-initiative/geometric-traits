//! Submodule providing the `CactusDetection` trait and its blanket
//! implementation for undirected graphs.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::{
    PositiveInteger, SparseMatrix2D, UndirectedMonopartiteEdges, UndirectedMonopartiteMonoplexGraph,
};

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

#[inline]
fn normalize_edge<NodeId: PositiveInteger>(left: NodeId, right: NodeId) -> [NodeId; 2] {
    if left <= right { [left, right] } else { [right, left] }
}

#[inline]
fn validate_component<NodeId: PositiveInteger>(
    edge_stack: &mut Vec<[NodeId; 2]>,
    stop_edge: Option<[NodeId; 2]>,
    touched: &mut [bool],
    touched_vertices: &mut Vec<NodeId>,
) -> bool {
    let mut edge_count = 0usize;

    while let Some(edge) = edge_stack.pop() {
        edge_count += 1;

        for vertex in edge {
            let vertex_index = vertex.as_();
            if !touched[vertex_index] {
                touched[vertex_index] = true;
                touched_vertices.push(vertex);
            }
        }

        if Some(edge) == stop_edge {
            break;
        }
    }

    let vertex_count = touched_vertices.len();
    for vertex in touched_vertices.drain(..) {
        touched[vertex.as_()] = false;
    }

    edge_count == 1 || edge_count == vertex_count
}

/// Trait providing cactus-graph recognition for undirected graphs.
///
/// A graph is a cactus when every edge belongs to at most one simple cycle.
/// Equivalently, each edge-based biconnected component is either a single
/// edge or a simple cycle.
///
/// The implementation performs a depth-first traversal with Tarjan low-link
/// bookkeeping and validates each edge-biconnected component as soon as it
/// closes. This avoids materializing the full decomposition and rejects early
/// on the first non-cactus block.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::{SortedVec, SymmetricCSR2D},
///     prelude::*,
///     traits::{CactusDetection, EdgesBuilder, VocabularyBuilder},
/// };
///
/// let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
/// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)];
/// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
///     .expected_number_of_symbols(nodes.len())
///     .symbols(nodes.into_iter().enumerate())
///     .build()
///     .unwrap();
/// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
///     .expected_number_of_edges(edges.len())
///     .expected_shape(nodes.len())
///     .edges(edges.into_iter())
///     .build()
///     .unwrap();
/// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
///
/// assert!(graph.is_cactus());
/// ```
pub trait CactusDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns whether the graph is a cactus graph.
    ///
    /// Disconnected graphs are accepted when each connected component is a
    /// cactus. Graphs with self-loops are rejected.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V + E) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{CactusDetection, EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// assert!(!graph.is_cactus());
    /// ```
    #[inline]
    fn is_cactus(&self) -> bool
    where
        Self: Sized,
    {
        if self.has_self_loops() {
            return false;
        }

        let number_of_nodes = self.number_of_nodes().as_();
        let mut discovery_time = vec![usize::MAX; number_of_nodes];
        let mut low = vec![0usize; number_of_nodes];
        let mut dfs_time = 0usize;
        let mut edge_stack: Vec<[Self::NodeId; 2]> = Vec::new();
        let mut dfs_stack: Vec<DfsFrame<'_, Self>> = Vec::new();
        let mut touched = vec![false; number_of_nodes];
        let mut touched_vertices = Vec::new();

        for root in self.node_ids() {
            if discovery_time[root.as_()] != usize::MAX {
                continue;
            }

            discovery_time[root.as_()] = dfs_time;
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

                    if discovery_time[neighbor.as_()] == usize::MAX {
                        discovery_time[neighbor.as_()] = dfs_time;
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
                        && discovery_time[neighbor.as_()] <= discovery_time[node.as_()]
                    {
                        low[node.as_()] = low[node.as_()].min(discovery_time[neighbor.as_()]);
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

                    if low[node_index] >= discovery_time[parent_index]
                        && !validate_component(
                            &mut edge_stack,
                            Some(normalize_edge(parent, node)),
                            &mut touched,
                            &mut touched_vertices,
                        )
                    {
                        return false;
                    }
                }
            }

            if !validate_component(&mut edge_stack, None, &mut touched, &mut touched_vertices) {
                return false;
            }
        }

        true
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> CactusDetection for G {}
