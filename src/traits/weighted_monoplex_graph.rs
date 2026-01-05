//! Submodule providing the traits for a generic graph that has weighted edges.

use crate::traits::Number;
use crate::traits::{SparseValuedMatrix, SparseValuedMatrix2D};

use super::{AttributedEdge, Edges, MonoplexGraph};

/// Trait defining a weighted edge.
pub trait WeightedEdge: AttributedEdge<Attribute = Self::Weight> {
    /// Type of the weight.
    type Weight: Number;

    /// Returns the weight of the edge.
    fn weight(&self) -> Self::Weight;
}

impl<E> WeightedEdge for E
where
    E: AttributedEdge,
    E::Attribute: Number,
{
    type Weight = E::Attribute;

    fn weight(&self) -> Self::Weight {
        self.attribute()
    }
}

/// Trait defining an edge data structure that has weighted edges.
pub trait WeightedEdges:
    Edges<
        Edge = <Self as WeightedEdges>::WeightedEdge,
        Matrix = <Self as WeightedEdges>::WeightedMatrix,
    >
{
    /// The type of the weight.
    type Weight: Number;
    /// The type of the weighted edge.
    type WeightedEdge: WeightedEdge<
            Weight = Self::Weight,
            SourceNodeId = Self::SourceNodeId,
            DestinationNodeId = Self::DestinationNodeId,
        >;
    /// The type of the underlying matrix.
    type WeightedMatrix: SparseValuedMatrix2D<
            Value = Self::Weight,
            RowIndex = Self::SourceNodeId,
            ColumnIndex = Self::DestinationNodeId,
            SparseIndex = Self::EdgeId,
        >;

    /// Returns the weights of the successors of a node.
    ///
    /// # Arguments
    ///
    /// * `source_node_id`: The node identifier.
    ///
    /// # Returns
    ///
    /// The weights of the successors of the node.
    fn successor_weights(
        &self,
        source_node_id: Self::SourceNodeId,
    ) -> <Self::WeightedMatrix as SparseValuedMatrix2D>::SparseRowValues<'_> {
        self.matrix().sparse_row_values(source_node_id)
    }

    /// Returns the largest weight of the successors of a node.
    ///
    /// # Arguments
    ///
    /// * `source_node_id`: The node identifier.
    ///
    /// # Returns
    ///
    /// The largest weight of the successors of the node, if any.
    fn max_successor_weight(&self, source_node_id: Self::SourceNodeId) -> Option<Self::Weight>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.matrix().sparse_row_max_value(source_node_id)
    }

    /// Returns the largest weight of the successors of a node and the
    /// corresponding successor node identifier.
    ///
    /// # Arguments
    ///
    /// * `source_node_id`: The node identifier.
    ///
    /// # Returns
    ///
    /// The largest weight of the successors of the node and the corresponding
    /// successor node identifier, if any.
    fn max_successor_weight_and_id(
        &self,
        source_node_id: Self::SourceNodeId,
    ) -> Option<(Self::Weight, Self::DestinationNodeId)>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.matrix()
            .sparse_row_max_value_and_column(source_node_id)
    }

    /// Returns the smallest weight of the successors of a node.
    ///
    /// # Arguments
    ///
    /// * `source_node_id`: The node identifier.
    ///
    /// # Returns
    ///
    /// The smallest weight of the successors of the node, if any.
    fn min_successor_weight(&self, source_node_id: Self::SourceNodeId) -> Option<Self::Weight>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.matrix().sparse_row_min_value(source_node_id)
    }

    /// Returns the smallest weight of the successors of a node and the
    /// corresponding successor node identifier.
    ///
    /// # Arguments
    ///
    /// * `source_node_id`: The node identifier.
    ///
    /// # Returns
    ///
    /// The smallest weight of the successors of the node and the corresponding
    /// successor node identifier, if any.
    fn min_successor_weight_and_id(
        &self,
        source_node_id: Self::SourceNodeId,
    ) -> Option<(Self::Weight, Self::DestinationNodeId)>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.matrix()
            .sparse_row_min_value_and_column(source_node_id)
    }

    /// Returns the sparse weights of the edges.
    fn sparse_weights(&self) -> <Self::WeightedMatrix as SparseValuedMatrix>::SparseValues<'_> {
        self.matrix().sparse_values()
    }
}

impl<E> WeightedEdges for E
where
    E: Edges,
    E::Edge: WeightedEdge,
    E::Matrix: SparseValuedMatrix2D<Value = <E::Edge as WeightedEdge>::Weight>,
{
    type Weight = <E::Edge as WeightedEdge>::Weight;
    type WeightedEdge = E::Edge;
    type WeightedMatrix = E::Matrix;
}

/// Trait defining a graph that has weighted edges.
pub trait WeightedMonoplexGraph:
    MonoplexGraph<
        Edges = <Self as WeightedMonoplexGraph>::WeightedEdges,
        Edge = <Self as WeightedMonoplexGraph>::WeightedEdge,
    >
{
    /// The type of the weight.
    type Weight: Number;
    /// The type of the weighted edge.
    type WeightedEdge: WeightedEdge<Weight = Self::Weight>;
    /// The type of the weighted edges.
    type WeightedEdges: WeightedEdges<Weight = Self::Weight, WeightedEdge = Self::WeightedEdge>;

    /// Returns the weights of the successors of a node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::ValuedCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::GenericGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 1.0), (0, 2, 2.0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> = GenericGraph::from((nodes, edges));
    ///
    /// let weights: Vec<f64> = graph.successor_weights(0).collect();
    /// assert_eq!(weights, vec![1.0, 2.0]);
    /// ```
    fn successor_weights(
        &self,
        source_node_id: <Self::WeightedEdges as Edges>::SourceNodeId,
    ) -> <<Self::WeightedEdges as WeightedEdges>::WeightedMatrix as SparseValuedMatrix2D>::SparseRowValues<'_>
    {
        self.edges().successor_weights(source_node_id)
    }

    /// Returns the largest weight of the successors of a node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::ValuedCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::GenericGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 1.0), (0, 2, 2.0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> = GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.max_successor_weight(0), Some(2.0));
    /// ```
    fn max_successor_weight(
        &self,
        source_node_id: <Self::WeightedEdges as Edges>::SourceNodeId,
    ) -> Option<Self::Weight>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.edges().max_successor_weight(source_node_id)
    }

    /// Returns the largest weight of the successors of a node and the
    /// corresponding successor node identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::ValuedCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::GenericGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 1.0), (0, 2, 2.0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> = GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.max_successor_weight_and_id(0), Some((2.0, 2)));
    /// ```
    fn max_successor_weight_and_id(
        &self,
        source_node_id: <Self::WeightedEdges as Edges>::SourceNodeId,
    ) -> Option<(
        Self::Weight,
        <Self::WeightedEdges as Edges>::DestinationNodeId,
    )>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.edges().max_successor_weight_and_id(source_node_id)
    }

    /// Returns the smallest weight of the successors of a node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::ValuedCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::GenericGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 1.0), (0, 2, 2.0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> = GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.min_successor_weight(0), Some(1.0));
    /// ```
    fn min_successor_weight(
        &self,
        source_node_id: <Self::WeightedEdges as Edges>::SourceNodeId,
    ) -> Option<Self::Weight>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.edges().min_successor_weight(source_node_id)
    }

    /// Returns the smallest weight of the successors of a node and the
    /// corresponding successor node identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::ValuedCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::GenericGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize, f64)> = vec![(0, 1, 1.0), (0, 2, 2.0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, ValuedCSR2D<usize, usize, usize, f64>> = GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.min_successor_weight_and_id(0), Some((1.0, 1)));
    /// ```
    fn min_successor_weight_and_id(
        &self,
        source_node_id: <Self::WeightedEdges as Edges>::SourceNodeId,
    ) -> Option<(
        Self::Weight,
        <Self::WeightedEdges as Edges>::DestinationNodeId,
    )>
    where
        Self::Weight: crate::traits::total_ord::TotalOrd,
    {
        self.edges().min_successor_weight_and_id(source_node_id)
    }

    /// Returns the sparse weights of the edges.
    fn sparse_weights(&self) -> <<Self::WeightedEdges as WeightedEdges>::WeightedMatrix as SparseValuedMatrix>::SparseValues<'_>{
        self.edges().sparse_weights()
    }
}

impl<G> WeightedMonoplexGraph for G
where
    G: MonoplexGraph,
    G::Edges: WeightedEdges<WeightedEdge = G::Edge, Weight = <G::Edge as WeightedEdge>::Weight>,
    G::Edge: WeightedEdge,
{
    type Weight = <G::Edges as WeightedEdges>::Weight;
    type WeightedEdges = G::Edges;
    type WeightedEdge = G::Edge;
}
