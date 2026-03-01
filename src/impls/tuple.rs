//! Submodule implementing graph-related traits for tuples.
use crate::traits::{AttributedEdge, Edge, Number, PositiveInteger};

impl<SourceNodeId: PositiveInteger, DestinationNodeId: PositiveInteger> Edge
    for (SourceNodeId, DestinationNodeId)
{
    type SourceNodeId = SourceNodeId;
    type DestinationNodeId = DestinationNodeId;

    #[inline]
    fn source(&self) -> Self::SourceNodeId {
        self.0
    }

    #[inline]
    fn destination(&self) -> Self::DestinationNodeId {
        self.1
    }
}

impl<SourceNodeId: PositiveInteger, DestinationNodeId: PositiveInteger, Weight: Number> Edge
    for (SourceNodeId, DestinationNodeId, Weight)
{
    type SourceNodeId = SourceNodeId;
    type DestinationNodeId = DestinationNodeId;

    #[inline]
    fn source(&self) -> Self::SourceNodeId {
        self.0
    }

    #[inline]
    fn destination(&self) -> Self::DestinationNodeId {
        self.1
    }
}

impl<SourceNodeId: PositiveInteger, DestinationNodeId: PositiveInteger, Weight: Number>
    AttributedEdge for (SourceNodeId, DestinationNodeId, Weight)
{
    type Attribute = Weight;

    #[inline]
    fn attribute(&self) -> Self::Attribute {
        self.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_2_tuple_source() {
        let edge: (usize, usize) = (5, 10);
        assert_eq!(edge.source(), 5);
    }

    #[test]
    fn test_edge_2_tuple_destination() {
        let edge: (usize, usize) = (5, 10);
        assert_eq!(edge.destination(), 10);
    }

    #[test]
    fn test_edge_3_tuple_source() {
        let edge: (usize, usize, f64) = (1, 2, 3.5);
        assert_eq!(edge.source(), 1);
    }

    #[test]
    fn test_edge_3_tuple_destination() {
        let edge: (usize, usize, f64) = (1, 2, 3.5);
        assert_eq!(edge.destination(), 2);
    }

    #[test]
    fn test_attributed_edge_3_tuple() {
        let edge: (usize, usize, f64) = (1, 2, 3.5);
        assert!((edge.attribute() - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_is_self_loop_true() {
        let edge: (usize, usize) = (5, 5);
        assert!(edge.is_self_loop());
    }

    #[test]
    fn test_edge_is_self_loop_false() {
        let edge: (usize, usize) = (5, 10);
        assert!(!edge.is_self_loop());
    }
}
