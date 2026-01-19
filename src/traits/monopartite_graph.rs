//! Submodule defining the properties of a monopartite graph.
//!
//! A monopartite graph is a graph where the nodes are of the same type, i.e
//! they are not divided into different partitions.

use num_traits::{SaturatingAdd, Zero};

use super::{BidirectionalVocabulary, Edges, Vocabulary};
use crate::traits::{IntoUsize, PositiveInteger, SparseSquareMatrix, Symbol, TryFromUsize};

/// Trait defining the properties of the monopartited edges of a graph.
pub trait MonopartiteEdges:
    Edges<
        SourceNodeId = <Self as MonopartiteEdges>::NodeId,
        DestinationNodeId = <Self as MonopartiteEdges>::NodeId,
        Matrix = <Self as MonopartiteEdges>::MonopartiteMatrix,
    >
{
    /// The monopartited matrix of the graph.
    type MonopartiteMatrix: SparseSquareMatrix<Index = Self::NodeId>;

    /// The identifier of the node.
    type NodeId: PositiveInteger + IntoUsize + TryFromUsize + SaturatingAdd;

    /// Returns whether the graph has self-loops.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, MonopartiteEdges, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (3, 3)];
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
    /// assert!(graph.edges().has_self_loops());
    /// ```
    fn has_self_loops(&self) -> bool {
        self.number_of_self_loops() > Self::NodeId::zero()
    }

    /// Returns the number of self-loops in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, MonopartiteEdges, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (3, 3)];
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
    /// assert_eq!(graph.edges().number_of_self_loops(), 1);
    /// ```
    fn number_of_self_loops(&self) -> Self::NodeId;
}

impl<E> MonopartiteEdges for E
where
    E: Edges<DestinationNodeId = <E as Edges>::SourceNodeId>,
    E::Matrix: SparseSquareMatrix<Index = E::SourceNodeId>,
{
    type MonopartiteMatrix = E::Matrix;
    type NodeId = E::SourceNodeId;

    fn number_of_self_loops(&self) -> Self::NodeId {
        self.matrix().number_of_defined_diagonal_values()
    }
}

/// Trait defining the properties of a monopartited graph.
pub trait MonopartiteGraph: super::Graph {
    /// The dense identifier of the nodes in the graph.
    type NodeId: PositiveInteger + IntoUsize + TryFromUsize;
    /// The symbol of the node.
    type NodeSymbol: Symbol;
    /// The vocabulary holding the symbols of the nodes.
    type Nodes: BidirectionalVocabulary<SourceSymbol = Self::NodeId, DestinationSymbol = Self::NodeSymbol>;

    /// Returns the nodes vocabulary.
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
    /// assert_eq!(graph.nodes_vocabulary().len(), 4);
    /// ```
    fn nodes_vocabulary(&self) -> &Self::Nodes;

    /// Returns the iterator over the node identifiers.
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
    /// let node_ids: Vec<usize> = graph.node_ids().collect();
    /// assert_eq!(node_ids, vec![0, 1, 2, 3]);
    /// ```
    fn node_ids(&self) -> <Self::Nodes as Vocabulary>::Sources<'_> {
        self.nodes_vocabulary().sources()
    }

    /// Returns the iterator over the node symbols.
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
    /// let nodes: Vec<usize> = graph.nodes().collect();
    /// assert_eq!(nodes, vec![0, 1, 2, 3]);
    /// ```
    fn nodes(&self) -> <Self::Nodes as Vocabulary>::Destinations<'_> {
        self.nodes_vocabulary().destinations()
    }

    /// Returns the number of nodes in the graph.
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
    /// assert_eq!(graph.number_of_nodes(), 4);
    /// ```
    fn number_of_nodes(&self) -> Self::NodeId {
        if let Ok(number_of_nodes) = Self::NodeId::try_from(self.nodes_vocabulary().len()) {
            number_of_nodes
        } else {
            panic!("The number of nodes exceeds the capacity of the node identifier.")
        }
    }
}
