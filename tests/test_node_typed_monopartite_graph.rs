//! Tests for node-typed monopartite graph traits.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SquareCSR2D},
    naive_structs::{
        GenericVocabularyBuilder,
        named_types::{DiEdgesBuilder, DiGraph},
    },
    traits::{EdgesBuilder, NodeType, NodeTypedMonopartiteGraph, TypedNode, VocabularyBuilder},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Kind {
    Entity,
    Action,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TypedNodeSymbol {
    id: usize,
    kind: Kind,
}

impl TypedNode for TypedNodeSymbol {
    type NodeType = Kind;

    fn node_type(&self) -> Self::NodeType {
        self.kind
    }
}

fn assert_node_type<T: NodeType>(_node_type: T) {}

fn build_graph() -> DiGraph<TypedNodeSymbol> {
    let nodes = vec![
        TypedNodeSymbol { id: 0, kind: Kind::Entity },
        TypedNodeSymbol { id: 1, kind: Kind::Action },
        TypedNodeSymbol { id: 2, kind: Kind::Entity },
    ];
    let edges = vec![(0, 1), (1, 2)];

    let nodes: SortedVec<TypedNodeSymbol> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(nodes.len())
        .edges(edges.into_iter())
        .build()
        .unwrap();

    DiGraph::from((nodes, edges))
}

#[test]
fn test_node_type_is_a_symbol_marker() {
    assert_node_type(Kind::Entity);
}

#[test]
fn test_node_type_lookup() {
    let graph = build_graph();

    assert_eq!(graph.node_type(&0), Some(Kind::Entity));
    assert_eq!(graph.node_type(&1), Some(Kind::Action));
    assert_eq!(graph.node_type(&2), Some(Kind::Entity));
    assert_eq!(graph.node_type(&3), None);
}

#[test]
fn test_is_node_of_type() {
    let graph = build_graph();

    assert!(graph.is_node_of_type(&0, &Kind::Entity));
    assert!(graph.is_node_of_type(&1, &Kind::Action));
    assert!(!graph.is_node_of_type(&1, &Kind::Entity));
    assert!(!graph.is_node_of_type(&3, &Kind::Entity));
}

#[test]
fn test_has_node_type() {
    let graph = build_graph();

    assert!(graph.has_node_type(&Kind::Entity));
    assert!(graph.has_node_type(&Kind::Action));
}
