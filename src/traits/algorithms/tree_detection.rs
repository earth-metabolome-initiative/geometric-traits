//! Submodule providing the `TreeDetection` trait and its blanket
//! implementation for undirected graphs.

use num_traits::AsPrimitive;

use crate::traits::{ConnectedComponents, UndirectedMonopartiteMonoplexGraph};

/// Trait providing predicates for detecting trees and forests in undirected
/// graphs.
pub trait TreeDetection: UndirectedMonopartiteMonoplexGraph + ConnectedComponents<usize>
where
    Self::NodeId: AsPrimitive<usize>,
{
    /// Returns true if the graph is a forest.
    ///
    /// A forest is an undirected graph with no cycles. Empty graphs and
    /// disconnected acyclic graphs are forests.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder, algorithms::tree_detection::TreeDetection},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (2, 3)];
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
    /// assert!(graph.is_forest());
    /// assert!(!graph.is_tree());
    /// ```
    #[inline]
    fn is_forest(&self) -> bool {
        if self.has_self_loops() {
            return false;
        }

        let number_of_nodes = self.number_of_nodes().as_();
        let number_of_undirected_edges = self.number_of_edges().as_() / 2;

        self.connected_components().is_ok_and(|connected_components| {
            number_of_undirected_edges + connected_components.number_of_components()
                == number_of_nodes
        })
    }

    /// Returns true if the graph is a tree.
    ///
    /// A tree is a connected forest.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder, algorithms::tree_detection::TreeDetection},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];
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
    /// assert!(graph.is_tree());
    /// ```
    #[inline]
    fn is_tree(&self) -> bool {
        if self.has_self_loops() {
            return false;
        }

        let number_of_nodes = self.number_of_nodes().as_();
        let number_of_undirected_edges = self.number_of_edges().as_() / 2;

        self.connected_components().is_ok_and(|connected_components| {
            connected_components.number_of_components() == 1
                && number_of_undirected_edges + 1 == number_of_nodes
        })
    }
}

impl<G> TreeDetection for G
where
    G: UndirectedMonopartiteMonoplexGraph + ConnectedComponents<usize>,
    G::NodeId: AsPrimitive<usize>,
{
}
