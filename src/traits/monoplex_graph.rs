//! Submodule providing the [`MonoplexGraph`] trait.
//!
//! A monoplex graph is a graph where all edges are of the same type,
//! i.e., there is no distinction between different types of edges.

use crate::traits::{SizedRowsSparseMatrix2D, SparseMatrix, SparseMatrix2D};

use super::Edges;

/// Trait for monoplex graphs.
pub trait MonoplexGraph: super::Graph {
    /// The type of the edge in the graph.
    type Edge: super::Edge;
    /// The type of the edges in the graph.
    type Edges: super::Edges<Edge = Self::Edge>;

    /// Returns a reference to the edges of the graph.
    fn edges(&self) -> &Self::Edges;

    /// Returns the successors of the node with the given identifier.
    ///
    /// # Arguments
    ///
    /// * `source_node_id` - The identifier of the source node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
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
    /// let successors: Vec<usize> = graph.successors(0).collect();
    /// assert_eq!(successors, vec![1, 2]);
    /// ```
    fn successors(
        &self,
        source_node_id: <Self::Edges as super::Edges>::SourceNodeId,
    ) -> <<Self::Edges as Edges>::Matrix as SparseMatrix2D>::SparseRow<'_> {
        self.edges().successors(source_node_id)
    }
    /// Returns whether the given source node has successors.
    ///
    /// # Arguments
    ///
    /// * `source_node_id` - The identifier of the source node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
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
    /// assert!(graph.has_successors(0));
    /// assert!(graph.has_successors(1));
    /// assert!(!graph.has_successors(2));
    /// ```
    fn has_successors(&self, source_node_id: <Self::Edges as super::Edges>::SourceNodeId) -> bool {
        self.edges().has_successors(source_node_id)
    }

    /// Returns whether the given source node has a successor with the given
    /// destination node identifier.
    ///
    /// # Arguments
    ///
    /// * `source_node_id` - The identifier of the source node.
    /// * `destination_node_id` - The identifier of the destination node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
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
    /// assert!(graph.has_successor(0, 1));
    /// assert!(graph.has_successor(0, 2));
    /// assert!(!graph.has_successor(0, 3));
    /// ```
    fn has_successor(
        &self,
        source_node_id: <Self::Edges as super::Edges>::SourceNodeId,
        destination_node_id: <Self::Edges as super::Edges>::DestinationNodeId,
    ) -> bool {
        self.edges()
            .has_successor(source_node_id, destination_node_id)
    }

    /// Returns the outbound degree of the node with the given identifier.
    ///
    /// # Arguments
    ///
    /// * `source_node_id` - The identifier of the source node.
    fn out_degree(
        &self,
        source_node_id: <Self::Edges as super::Edges>::SourceNodeId,
    ) -> <Self::Edges as super::Edges>::DestinationNodeId {
        self.edges().out_degree(source_node_id)
    }

    /// Iterates across all out degrees of the graph.
    fn out_degrees(
        &self,
    ) -> <<Self::Edges as Edges>::Matrix as SizedRowsSparseMatrix2D>::SparseRowSizes<'_> {
        self.edges().out_degrees()
    }

    /// Returns the iterator over all sparse coordinates of the matrix.
    fn sparse_coordinates(
        &self,
    ) -> <<Self::Edges as Edges>::Matrix as SparseMatrix>::SparseCoordinates<'_> {
        self.edges().sparse_coordinates()
    }

    /// Returns the number of edges in the graph.
    fn number_of_edges(&self) -> <Self::Edges as super::Edges>::EdgeId {
        self.edges().number_of_edges()
    }
}
