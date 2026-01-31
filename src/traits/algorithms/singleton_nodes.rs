//! Submodule providing the `SingletonNodes` trait and its blanket
//! implementation, which provides a method to retrieve the singleton nodes from
//! a graph, which is a node with no predecessor and no successors edges.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::traits::{IntoUsize, MonoplexMonopartiteGraph};
/// Trait providing the `singleton_nodes` method, which returns the singleton
/// nodes of the graph. A singleton node is a node with no predecessors and no
/// successors.
pub trait SingletonNodes: MonoplexMonopartiteGraph {
    /// Returns the singleton nodes of the graph.
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
    /// // Node 2 and 3 are singleton nodes because they have no incoming or outgoing edges.
    /// assert_eq!(graph.singleton_nodes(), vec![2, 3]);
    /// ```
    fn singleton_nodes(&self) -> Vec<Self::NodeId> {
        let mut visited = vec![false; self.number_of_nodes().into_usize()];

        // Iterate over all nodes and mark the successors of each node as
        // visited. A node is considered visited if it has a predecessor.
        for node in self.node_ids() {
            let mut has_successors = false;
            // Mark the successors of the node as visited.
            for successor_node_id in self.successors(node) {
                visited[successor_node_id.into_usize()] = true;
                has_successors = true;
            }
            if has_successors {
                visited[node.into_usize()] = true;
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

impl<G: MonoplexMonopartiteGraph + ?Sized> SingletonNodes for G {}
