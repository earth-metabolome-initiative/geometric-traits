//! Submodule providing the `RootNodes` trait and its blanket
//! implementation, which provides a method to retrieve the root nodes of the
//! graph, which are the set of nodes with no predecessors.
use alloc::vec::Vec;

use super::node_classification::predecessor_successor_flags;
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
    #[inline]
    fn root_nodes(&self) -> Vec<Self::NodeId> {
        let (has_predecessor, _) = predecessor_successor_flags(self);

        self.node_ids()
            .zip(has_predecessor)
            .filter_map(|(node, has_predecessor)| (!has_predecessor).then_some(node))
            .collect()
    }
}

impl<G: ?Sized + MonoplexMonopartiteGraph> RootNodes for G {}
