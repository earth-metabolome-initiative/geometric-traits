//! Submodule defining a trait for a bipartite graph.
//!
//! A bipartite graph is a graph whose vertices can be divided into two disjoint
//! sets such that no two vertices within the same set are adjacent.

use num_traits::AsPrimitive;

use super::{BidirectionalVocabulary, Graph, Vocabulary};
use crate::traits::{PositiveInteger, Symbol, TryFromUsize};

/// Trait defining the properties of a bipartite graph.
pub trait BipartiteGraph: Graph {
    /// The dense identifiers of the left nodes in the graph.
    type LeftNodeId: PositiveInteger + AsPrimitive<usize> + TryFromUsize;
    /// The dense identifiers of the right nodes in the graph.
    type RightNodeId: PositiveInteger + AsPrimitive<usize> + TryFromUsize;
    /// The symbol of the left node.
    type LeftNodeSymbol: Symbol;
    /// The symbol of the right node.
    type RightNodeSymbol: Symbol;
    /// The vocabulary holding the symbols of the left nodes.
    type LeftNodes: BidirectionalVocabulary<
            SourceSymbol = Self::LeftNodeId,
            DestinationSymbol = Self::LeftNodeSymbol,
        >;
    /// The vocabulary holding the symbols of the right nodes.
    type RightNodes: BidirectionalVocabulary<
            SourceSymbol = Self::RightNodeId,
            DestinationSymbol = Self::RightNodeSymbol,
        >;

    /// Returns a reference to the vocabulary of the left nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.left_nodes_vocabulary().len(), 2);
    /// ```
    fn left_nodes_vocabulary(&self) -> &Self::LeftNodes;

    /// Returns an iterator over the left node IDs in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// let left_node_ids: Vec<usize> = graph.left_node_ids().collect();
    /// assert_eq!(left_node_ids, vec![0, 1]);
    /// ```
    #[inline]
    fn left_node_ids(&self) -> <Self::LeftNodes as Vocabulary>::Sources<'_> {
        self.left_nodes_vocabulary().sources()
    }

    /// Returns an iterator over the left node symbols in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// let left_nodes: Vec<usize> = graph.left_nodes().collect();
    /// assert_eq!(left_nodes, vec![0, 1]);
    /// ```
    #[inline]
    fn left_nodes(&self) -> <Self::LeftNodes as Vocabulary>::Destinations<'_> {
        self.left_nodes_vocabulary().destinations()
    }

    /// Returns the Symbol of the node with the given ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.left_node(&0), Some(0));
    /// assert_eq!(graph.left_node(&1), Some(1));
    /// assert_eq!(graph.left_node(&2), None);
    /// ```
    #[inline]
    fn left_node(&self, left_node_id: &Self::LeftNodeId) -> Option<Self::LeftNodeSymbol> {
        self.left_nodes_vocabulary().convert(left_node_id)
    }

    /// Returns the ID of the node with the given symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.left_node_id(&0), Some(0));
    /// assert_eq!(graph.left_node_id(&1), Some(1));
    /// assert_eq!(graph.left_node_id(&2), None);
    /// ```
    #[inline]
    fn left_node_id(&self, symbol: &Self::LeftNodeSymbol) -> Option<Self::LeftNodeId> {
        self.left_nodes_vocabulary().invert(symbol)
    }

    /// Returns the number of left nodes in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.number_of_left_nodes(), 2);
    /// ```
    #[inline]
    fn number_of_left_nodes(&self) -> usize {
        self.left_nodes_vocabulary().len()
    }

    /// Returns a reference to the vocabulary of the right nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.right_nodes_vocabulary().len(), 2);
    /// ```
    fn right_nodes_vocabulary(&self) -> &Self::RightNodes;

    /// Returns an iterator over the right node IDs in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// let right_node_ids: Vec<usize> = graph.right_node_ids().collect();
    /// assert_eq!(right_node_ids, vec![0, 1]);
    /// ```
    #[inline]
    fn right_node_ids(&self) -> <Self::RightNodes as Vocabulary>::Sources<'_> {
        self.right_nodes_vocabulary().sources()
    }

    /// Returns an iterator over the node symbols in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// let right_nodes: Vec<usize> = graph.right_nodes().collect();
    /// assert_eq!(right_nodes, vec![0, 1]);
    /// ```
    #[inline]
    fn right_nodes(&self) -> <Self::RightNodes as Vocabulary>::Destinations<'_> {
        self.right_nodes_vocabulary().destinations()
    }

    /// Returns the Symbol of the node with the given ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.right_node(&0), Some(0));
    /// assert_eq!(graph.right_node(&1), Some(1));
    /// assert_eq!(graph.right_node(&2), None);
    /// ```
    #[inline]
    fn right_node(&self, right_node_id: &Self::RightNodeId) -> Option<Self::RightNodeSymbol> {
        self.right_nodes_vocabulary().convert(right_node_id)
    }

    /// Returns the ID of the node with the given symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.right_node_id(&0), Some(0));
    /// assert_eq!(graph.right_node_id(&1), Some(1));
    /// assert_eq!(graph.right_node_id(&2), None);
    /// ```
    #[inline]
    fn right_node_id(&self, symbol: &Self::RightNodeSymbol) -> Option<Self::RightNodeId> {
        self.right_nodes_vocabulary().invert(symbol)
    }

    /// Returns the number of right nodes in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::{GenericEdgesBuilder, named_types::BiGraph},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> =
    ///     GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///         .expected_number_of_edges(edges.len())
    ///         .expected_shape((left_nodes.len(), right_nodes.len()))
    ///         .edges(edges.into_iter())
    ///         .build()
    ///         .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.number_of_right_nodes(), 2);
    /// ```
    #[inline]
    fn number_of_right_nodes(&self) -> usize {
        self.right_nodes_vocabulary().len()
    }
}
