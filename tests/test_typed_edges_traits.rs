//! Tests for EdgeType, TypedEdge, and TypedEdges traits.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    traits::{
        AttributedEdge, BipartiteGraph, Edge, EdgeType, Graph, MatrixMut, MonoplexGraph,
        SparseMatrixMut, TypedEdge, TypedEdges,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum RelationType {
    DependsOn,
    Uses,
}

#[derive(Clone, Debug)]
struct RelationEdge {
    source: usize,
    destination: usize,
    relation: RelationType,
}

impl Edge for RelationEdge {
    type SourceNodeId = usize;
    type DestinationNodeId = usize;

    fn source(&self) -> Self::SourceNodeId {
        self.source
    }

    fn destination(&self) -> Self::DestinationNodeId {
        self.destination
    }
}

impl AttributedEdge for RelationEdge {
    type Attribute = RelationType;

    fn attribute(&self) -> Self::Attribute {
        self.relation
    }
}

fn assert_edge_type<T: EdgeType>(_edge_type: T) {}

fn assert_typed_edge<E: TypedEdge>(_edge: &E) {}

fn assert_typed_edges<E>(_edges: &E)
where
    E: TypedEdges,
    E::Edge: AttributedEdge,
    <E::Edge as AttributedEdge>::Attribute: EdgeType,
{
}

#[test]
fn test_edge_type_marker() {
    assert_edge_type(RelationType::DependsOn);
}

#[test]
fn test_typed_edge_for_custom_edge() {
    let edge = RelationEdge { source: 0, destination: 1, relation: RelationType::Uses };

    assert_typed_edge(&edge);
    assert_eq!(edge.edge_type(), RelationType::Uses);
}

#[test]
fn test_typed_edge_for_tuple_with_symbolic_attribute() {
    let edge: (usize, usize, usize) = (0, 1, 7);

    assert_typed_edge(&edge);
    assert_eq!(edge.edge_type(), 7);
}

#[test]
fn test_symbolic_valued_csr2d_exposes_graph_and_typed_edge_traits() {
    let mut edges: ValuedCSR2D<usize, usize, usize, RelationType> =
        SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut edges, (0, 1, RelationType::DependsOn)).unwrap();

    assert_typed_edges(&edges);
    assert!(edges.has_nodes());
    assert!(edges.has_edges());
    assert_eq!(edges.number_of_left_nodes(), 2);
    assert_eq!(edges.number_of_right_nodes(), 2);
    assert_eq!(edges.successors(0).collect::<Vec<_>>(), vec![1]);
}
