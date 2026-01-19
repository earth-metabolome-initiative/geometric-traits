//! Submodule providing the `SinkNodes` trait and its blanket
//! implementation, which provides a method to retrieve the sink nodes of the
//! graph, which are the set of nodes with no successors.

use crate::traits::{IntoUsize, MonoplexMonopartiteGraph};
/// Trait providing the `sink_nodes` method, which returns the sink nodes of the
/// graph. A sink node is a node with no successors.
pub trait SinkNodes: MonoplexMonopartiteGraph {
    /// Returns the sink nodes of the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: DiGraph<usize> = DiGraph::from((nodes, edges));
    ///
    /// // Node 1 is a sink node because it has incoming edges but no outgoing edges.
    /// // Node 2 is isolated, so it is not considered a sink node.
    /// assert_eq!(graph.sink_nodes(), vec![1]);
    /// ```
    fn sink_nodes(&self) -> Vec<Self::NodeId> {
        let mut visited = vec![false; self.number_of_nodes().into_usize()];

        // Iterate over all nodes and mark the successors of each node as
        // visited. A node is considered visited if it has a predecessor.
        for node in self.node_ids() {
            // Mark the successors of the node as visited.
            for successor_node_id in self.successors(node) {
                visited[successor_node_id.into_usize()] = true;
            }
        }
        // Finally, we iterate over all nodes and keep the nodes that have not
        // been visited. A node is considered visited if it has a predecessor.
        self.node_ids()
            .zip(visited)
            .filter_map(
                |(node, visited)| {
                    if visited && !self.has_successors(node) { Some(node) } else { None }
                },
            )
            .collect()
    }
}

impl<G: MonoplexMonopartiteGraph + ?Sized> SinkNodes for G {}
