//! Submodule defining traits for node-typed bipartite graphs.
//!
//! These graphs are characterized by the fact that:
//!
//! * Left nodes carry a node type.
//! * Right nodes carry a node type.

use super::{BipartiteGraph, TypedNode, Vocabulary};

/// Trait implemented by bipartite graphs whose left and right node symbols are
/// typed.
pub trait NodeTypedBipartiteGraph: BipartiteGraph
where
    Self::LeftNodeSymbol: TypedNode,
    Self::RightNodeSymbol: TypedNode,
{
    /// Returns the node type of the left node with the provided identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     naive_structs::named_types::BiGraph,
    ///     prelude::*,
    ///     traits::{EdgesBuilder, NodeTypedBipartiteGraph, TypedNode, VocabularyBuilder},
    /// };
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// enum LeftKind {
    ///     Entity,
    ///     Action,
    /// }
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// enum RightKind {
    ///     Concept,
    /// }
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct LeftNodeSymbol {
    ///     id: usize,
    ///     kind: LeftKind,
    /// }
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct RightNodeSymbol {
    ///     id: usize,
    ///     kind: RightKind,
    /// }
    ///
    /// impl TypedNode for LeftNodeSymbol {
    ///     type NodeType = LeftKind;
    ///
    ///     fn node_type(&self) -> Self::NodeType {
    ///         self.kind
    ///     }
    /// }
    ///
    /// impl TypedNode for RightNodeSymbol {
    ///     type NodeType = RightKind;
    ///
    ///     fn node_type(&self) -> Self::NodeType {
    ///         self.kind
    ///     }
    /// }
    ///
    /// let left_nodes = vec![
    ///     LeftNodeSymbol { id: 0, kind: LeftKind::Entity },
    ///     LeftNodeSymbol { id: 1, kind: LeftKind::Action },
    /// ];
    /// let right_nodes = vec![
    ///     RightNodeSymbol { id: 0, kind: RightKind::Concept },
    ///     RightNodeSymbol { id: 1, kind: RightKind::Concept },
    /// ];
    /// let edges: Vec<(usize, usize)> = vec![(0, 0), (1, 1)];
    ///
    /// let left_nodes: SortedVec<LeftNodeSymbol> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<RightNodeSymbol> = GenericVocabularyBuilder::default()
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
    /// let graph: BiGraph<LeftNodeSymbol, RightNodeSymbol> =
    ///     BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// assert_eq!(graph.left_node_type(&0), Some(LeftKind::Entity));
    /// assert_eq!(graph.left_node_type(&1), Some(LeftKind::Action));
    /// assert_eq!(graph.left_node_type(&2), None);
    /// ```
    fn left_node_type(
        &self,
        left_node_id: &Self::LeftNodeId,
    ) -> Option<<Self::LeftNodeSymbol as TypedNode>::NodeType> {
        self.left_nodes_vocabulary().convert(left_node_id).map(|node| node.node_type())
    }

    /// Returns the node type of the right node with the provided identifier.
    fn right_node_type(
        &self,
        right_node_id: &Self::RightNodeId,
    ) -> Option<<Self::RightNodeSymbol as TypedNode>::NodeType> {
        self.right_nodes_vocabulary().convert(right_node_id).map(|node| node.node_type())
    }

    /// Returns whether the left node with the provided identifier has the
    /// provided node type.
    fn is_left_node_of_type(
        &self,
        left_node_id: &Self::LeftNodeId,
        node_type: &<Self::LeftNodeSymbol as TypedNode>::NodeType,
    ) -> bool {
        match self.left_node_type(left_node_id) {
            Some(current_type) => current_type == *node_type,
            None => false,
        }
    }

    /// Returns whether the right node with the provided identifier has the
    /// provided node type.
    fn is_right_node_of_type(
        &self,
        right_node_id: &Self::RightNodeId,
        node_type: &<Self::RightNodeSymbol as TypedNode>::NodeType,
    ) -> bool {
        match self.right_node_type(right_node_id) {
            Some(current_type) => current_type == *node_type,
            None => false,
        }
    }

    /// Returns whether the graph contains at least one left node of the
    /// provided node type.
    fn has_left_node_type(
        &self,
        node_type: &<Self::LeftNodeSymbol as TypedNode>::NodeType,
    ) -> bool {
        self.left_nodes().any(|node| node.node_type() == *node_type)
    }

    /// Returns whether the graph contains at least one right node of the
    /// provided node type.
    fn has_right_node_type(
        &self,
        node_type: &<Self::RightNodeSymbol as TypedNode>::NodeType,
    ) -> bool {
        self.right_nodes().any(|node| node.node_type() == *node_type)
    }
}

impl<G> NodeTypedBipartiteGraph for G
where
    G: BipartiteGraph,
    G::LeftNodeSymbol: TypedNode,
    G::RightNodeSymbol: TypedNode,
{
}
