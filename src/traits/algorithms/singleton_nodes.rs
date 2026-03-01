//! Submodule providing the `SingletonNodes` trait and its blanket
//! implementation, which provides a method to retrieve the singleton nodes from
//! a graph, which is a node with no predecessor and no successors edges.
use alloc::vec::Vec;

use super::node_classification::predecessor_successor_flags;
use crate::traits::MonoplexMonopartiteGraph;
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
        let (has_predecessor, has_successor) = predecessor_successor_flags(self);

        self.node_ids()
            .zip(has_predecessor.into_iter().zip(has_successor))
            .filter_map(|(node, (has_predecessor, has_successor))| {
                (!has_predecessor && !has_successor).then_some(node)
            })
            .collect()
    }
}

impl<G: MonoplexMonopartiteGraph + ?Sized> SingletonNodes for G {}
