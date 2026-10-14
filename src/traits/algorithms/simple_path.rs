//! Submodule providing the trait `SimplePath`, which checks if the directed
//! graph associated with the provided structure is a simple path, i.e., a
//! linear sequence of nodes where each node (except the first and last) has
//! exactly one predecessor and one successor, with the last node having no
//! successors and the first node having no predecessors.

use num_traits::{One, SaturatingAdd, Zero};

use crate::traits::RootNodes;

/// A simple path is a special case of a directed acyclic graph (DAG) where
/// there is a single linear sequence of nodes. In a simple path, each node
/// (except the first and last) has exactly one predecessor and one successor,
/// with the last node having no successors and the first node having no
/// predecessors.
pub trait SimplePath: RootNodes {
    /// Returns whether the directed graph associated with the structure is a
    /// simple path.
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
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
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
    /// assert!(graph.is_simple_path());
    /// ```
    #[inline]
    fn is_simple_path(&self) -> bool {
        if self.number_of_nodes() == Self::NodeId::zero() {
            // An empty graph is not a simple path.
            return false;
        }

        let root_nodes = self.root_nodes();
        if root_nodes.len() != 1 {
            return false; // A simple path must have exactly one root node.
        }

        let mut current_node = root_nodes[0];
        let mut visited_nodes = Self::NodeId::zero();

        loop {
            visited_nodes = visited_nodes.saturating_add(&Self::NodeId::one());
            if self.out_degree(current_node) > Self::NodeId::one() {
                return false; // More than one outgoing edge means it's not a simple path.
            }

            if let Some(successor) = self.successors(current_node).next() {
                if visited_nodes == self.number_of_nodes() {
                    return false; // More visited nodes than total nodes means a cycle.
                }

                current_node = successor;
            } else {
                break;
            }
        }

        visited_nodes == self.number_of_nodes()
    }
}

impl<T> SimplePath for T where T: RootNodes {}
