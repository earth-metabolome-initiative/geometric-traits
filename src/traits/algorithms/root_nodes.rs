//! Submodule providing the `RootNodes` trait and its blanket
//! implementation, which provides a method to retrieve the root nodes of the
//! graph, which are the set of nodes with no predecessors.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

/// Trait providing the `root_nodes` method, which returns the root nodes of the
/// graph. A root node is a node with no predecessors.
pub trait RootNodes: MonoplexMonopartiteGraph {
    /// Returns the root nodes of the graph.
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
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (2, 3)];
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
    /// let roots = graph.root_nodes();
    /// assert_eq!(roots, vec![0, 2]);
    /// ```
    fn root_nodes(&self) -> Vec<Self::NodeId> {
        // Create a vector to store whether a node has been visited or not
        // and initialize it to false for all nodes.
        let mut visited = vec![false; self.number_of_nodes().as_()];

        // Iterate over all nodes and mark the successors of each node as
        // visited. A node is considered visited if it has a predecessor.
        for node in self.node_ids() {
            // Mark the successors of the node as visited.
            for successor_node_id in self.successors(node) {
                visited[successor_node_id.as_()] = true;
            }
        }

        // Finally, we iterate over all nodes and keep the nodes that have not
        // been visited. A node is considered visited if it has a predecessor.
        self.node_ids()
            .zip(visited)
            .filter_map(|(node, visited)| if visited { None } else { Some(node) })
            .collect()
    }
}

impl<G: ?Sized + MonoplexMonopartiteGraph> RootNodes for G {}
