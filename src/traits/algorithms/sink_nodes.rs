//! Submodule providing the `SinkNodes` trait and its blanket
//! implementation, which provides a method to retrieve the sink nodes of the
//! graph, which are the set of nodes with no successors.
use alloc::vec::Vec;

use super::node_classification::predecessor_successor_flags;
use crate::traits::MonoplexMonopartiteGraph;
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
    #[inline]
    fn sink_nodes(&self) -> Vec<Self::NodeId> {
        let (has_predecessor, has_successor) = predecessor_successor_flags(self);

        self.node_ids()
            .zip(has_predecessor.into_iter().zip(has_successor))
            .filter_map(|(node, (has_predecessor, has_successor))| {
                (has_predecessor && !has_successor).then_some(node)
            })
            .collect()
    }
}

impl<G: MonoplexMonopartiteGraph + ?Sized> SinkNodes for G {}
