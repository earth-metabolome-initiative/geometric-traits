//! Submodule defining traits for node-typed monopartite graphs.
//!
//! These graphs are characterized by the fact that:
//!
//! * Nodes carry a node type.
//! * Node types are symbols.

use super::{MonopartiteGraph, Symbol, Vocabulary};

/// Marker trait defining a node type.
pub trait NodeType: Symbol {}

impl<T> NodeType for T where T: Symbol {}

/// Trait defining a node carrying an associated node type.
pub trait TypedNode: Symbol {
    /// Type of the node type.
    type NodeType: NodeType;

    /// Returns the node type.
    fn node_type(&self) -> Self::NodeType;
}

/// Trait implemented by monopartite graphs whose node symbols are typed.
pub trait NodeTypedMonopartiteGraph: MonopartiteGraph
where
    Self::NodeSymbol: TypedNode,
{
    /// Returns the node type of the node with the provided node identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, NodeTypedMonopartiteGraph, TypedNode, VocabularyBuilder},
    /// };
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// enum Kind {
    ///     Entity,
    ///     Action,
    /// }
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct TypedNodeSymbol {
    ///     id: usize,
    ///     kind: Kind,
    /// }
    ///
    /// impl TypedNode for TypedNodeSymbol {
    ///     type NodeType = Kind;
    ///
    ///     fn node_type(&self) -> Self::NodeType {
    ///         self.kind
    ///     }
    /// }
    ///
    /// let nodes: Vec<TypedNodeSymbol> = vec![
    ///     TypedNodeSymbol { id: 0, kind: Kind::Entity },
    ///     TypedNodeSymbol { id: 1, kind: Kind::Action },
    ///     TypedNodeSymbol { id: 2, kind: Kind::Entity },
    /// ];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<TypedNodeSymbol> = GenericVocabularyBuilder::default()
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
    /// let graph: DiGraph<TypedNodeSymbol> = DiGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.node_type(&0), Some(Kind::Entity));
    /// assert_eq!(graph.node_type(&1), Some(Kind::Action));
    /// assert_eq!(graph.node_type(&3), None);
    /// ```
    fn node_type(
        &self,
        node_id: &Self::NodeId,
    ) -> Option<<Self::NodeSymbol as TypedNode>::NodeType> {
        self.nodes_vocabulary().convert(node_id).map(|node| node.node_type())
    }

    /// Returns whether the node with the provided identifier has the provided
    /// node type.
    fn is_node_of_type(
        &self,
        node_id: &Self::NodeId,
        node_type: &<Self::NodeSymbol as TypedNode>::NodeType,
    ) -> bool {
        match self.node_type(node_id) {
            Some(current_type) => current_type == *node_type,
            None => false,
        }
    }

    /// Returns whether the graph contains at least one node of the provided
    /// node type.
    fn has_node_type(&self, node_type: &<Self::NodeSymbol as TypedNode>::NodeType) -> bool {
        self.nodes().any(|node| node.node_type() == *node_type)
    }
}

impl<G> NodeTypedMonopartiteGraph for G
where
    G: MonopartiteGraph,
    G::NodeSymbol: TypedNode,
{
}
