//! Submodule defining a Transposed Monoplex Graph.
//!
//! A transposed monoplex graph is a graph where the edges are of a single type
//! and it is possible to efficiently access the predecessors of a node.

use super::{Edges, MonoplexGraph, TransposedEdges};
use crate::traits::{SizedRowsSparseMatrix2D, SparseBiMatrix2D, SparseMatrix2D};

/// Trait defining a transposed monoplex graph.
///
/// This trait extends `MonoplexGraph` with support for efficiently accessing
/// predecessors of nodes.
pub trait TransposedMonoplexGraph: MonoplexGraph<Edges = Self::TransposedEdges> {
    /// The type of edges in the graph, which must support transposed (column)
    /// access.
    type TransposedEdges: TransposedEdges;

    /// Returns the predecessors of the node with the given identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, GenericGraph},
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
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((nodes.len(), nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let edges: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
    ///     GenericBiMatrix2D::new(edges);
    /// let graph: GenericGraph<
    ///     SortedVec<usize>,
    ///     GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>,
    /// > = GenericGraph::from((nodes, edges));
    ///
    /// let predecessors: Vec<usize> = graph.predecessors(1).collect();
    /// assert_eq!(predecessors, vec![0]);
    /// ```
	fn predecessors(
		&self,
		destination_node_id: <Self::TransposedEdges as Edges>::DestinationNodeId,
    ) -> <<<Self::TransposedEdges as TransposedEdges>::BiMatrix as SparseBiMatrix2D>::SparseTransposedMatrix as SparseMatrix2D>::SparseRow<'_>{
        self.edges().predecessors(destination_node_id)
    }

    /// Returns whether the given destination node has a predecessor with the
    /// given source node identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, GenericGraph},
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
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((nodes.len(), nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let edges: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
    ///     GenericBiMatrix2D::new(edges);
    /// let graph: GenericGraph<
    ///     SortedVec<usize>,
    ///     GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>,
    /// > = GenericGraph::from((nodes, edges));
    ///
    /// assert!(graph.has_predecessor(1, 0));
    /// assert!(!graph.has_predecessor(1, 2));
    /// ```
    fn has_predecessor(
        &self,
        destination_node_id: <Self::TransposedEdges as Edges>::DestinationNodeId,
        source_node_id: <Self::TransposedEdges as Edges>::SourceNodeId,
    ) -> bool {
        self.edges().has_predecessor(destination_node_id, source_node_id)
    }

    /// Returns whether the given destination node has any predecessor.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, GenericGraph},
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
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((nodes.len(), nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let edges: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
    ///     GenericBiMatrix2D::new(edges);
    /// let graph: GenericGraph<
    ///     SortedVec<usize>,
    ///     GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>,
    /// > = GenericGraph::from((nodes, edges));
    ///
    /// assert!(graph.has_predecessors(1));
    /// assert!(!graph.has_predecessors(0));
    /// ```
    fn has_predecessors(
        &self,
        destination_node_id: <Self::TransposedEdges as Edges>::DestinationNodeId,
    ) -> bool {
        self.edges().has_predecessors(destination_node_id)
    }

    /// Returns the inbound degree of the node with the given identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, GenericGraph},
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
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((nodes.len(), nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let edges: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
    ///     GenericBiMatrix2D::new(edges);
    /// let graph: GenericGraph<
    ///     SortedVec<usize>,
    ///     GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>,
    /// > = GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.in_degree(1), 1);
    /// assert_eq!(graph.in_degree(0), 0);
    /// ```
    fn in_degree(
        &self,
        destination_node_id: <Self::TransposedEdges as Edges>::DestinationNodeId,
    ) -> <Self::TransposedEdges as Edges>::SourceNodeId {
        self.edges().in_degree(destination_node_id)
    }

    /// Returns an iterator over the inbound degrees of the nodes.
    ///
    /// # Returns
    ///
    /// An iterator over the inbound degrees of the nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, GenericBiMatrix2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, GenericGraph},
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
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((nodes.len(), nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let edges: GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>> =
    ///     GenericBiMatrix2D::new(edges);
    /// let graph: GenericGraph<
    ///     SortedVec<usize>,
    ///     GenericBiMatrix2D<CSR2D<usize, usize, usize>, CSR2D<usize, usize, usize>>,
    /// > = GenericGraph::from((nodes, edges));
    ///
    /// let in_degrees: Vec<usize> = graph.in_degrees().collect();
    /// assert_eq!(in_degrees, vec![0, 1, 1]);
    /// ```
	fn in_degrees(
		&self
    ) -> <<<Self::TransposedEdges as TransposedEdges>::BiMatrix as SparseBiMatrix2D>::SparseTransposedMatrix as SizedRowsSparseMatrix2D>::SparseRowSizes<'_>{
        self.edges().in_degrees()
    }
}

impl<G> TransposedMonoplexGraph for G
where
    G: MonoplexGraph,
    G::Edges: TransposedEdges,
{
    type TransposedEdges = G::Edges;
}
