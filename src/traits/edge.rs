//! Submodule defining the properties of an edge.

use core::fmt::Debug;

use crate::traits::{PositiveInteger, Symbol};

/// Trait defining the properties of an edge.
pub trait Edge: Debug + Clone {
    /// Type of the source node identifier.
    type SourceNodeId: PositiveInteger;
    /// Type of the destination node identifier.
    type DestinationNodeId: PositiveInteger;

    /// Returns the identifier of the source node.
    fn source(&self) -> Self::SourceNodeId;

    /// Returns the identifier of the destination node.
    fn destination(&self) -> Self::DestinationNodeId;

    /// Returns whether the edge is a self-loop.
    #[inline]
    fn is_self_loop(&self) -> bool
    where
        Self::SourceNodeId: PartialEq<Self::DestinationNodeId>,
    {
        self.source() == self.destination()
    }
}

/// Trait defining an attributed edge.
pub trait AttributedEdge: Edge {
    /// Type of the attribute.
    type Attribute;

    /// Returns the attribute of the edge.
    fn attribute(&self) -> Self::Attribute;
}

/// Marker trait defining an edge type.
pub trait EdgeType: Symbol {}

impl<T> EdgeType for T where T: Symbol {}

/// Trait defining an attributed edge whose attribute is an edge type.
pub trait TypedEdge: AttributedEdge {
    /// Returns the edge type.
    #[inline]
    fn edge_type(&self) -> Self::Attribute
    where
        Self::Attribute: EdgeType,
    {
        self.attribute()
    }
}

impl<E> TypedEdge for E
where
    E: AttributedEdge,
    E::Attribute: EdgeType,
{
}
