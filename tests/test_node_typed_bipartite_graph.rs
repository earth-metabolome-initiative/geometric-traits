//! Tests for node-typed bipartite graph traits.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    naive_structs::{GenericVocabularyBuilder, named_types::BiGraph},
    prelude::*,
    traits::{EdgesBuilder, NodeType, NodeTypedBipartiteGraph, TypedNode, VocabularyBuilder},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum LeftKind {
    Entity,
    Action,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum RightKind {
    Concept,
    Token,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct LeftSymbol {
    id: usize,
    kind: LeftKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RightSymbol {
    id: usize,
    kind: RightKind,
}

impl TypedNode for LeftSymbol {
    type NodeType = LeftKind;

    fn node_type(&self) -> Self::NodeType {
        self.kind
    }
}

impl TypedNode for RightSymbol {
    type NodeType = RightKind;

    fn node_type(&self) -> Self::NodeType {
        self.kind
    }
}

fn assert_node_type<T: NodeType>(_node_type: T) {}

fn build_graph() -> BiGraph<LeftSymbol, RightSymbol> {
    let left_nodes = vec![
        LeftSymbol { id: 0, kind: LeftKind::Entity },
        LeftSymbol { id: 1, kind: LeftKind::Action },
    ];
    let right_nodes = vec![
        RightSymbol { id: 0, kind: RightKind::Concept },
        RightSymbol { id: 1, kind: RightKind::Token },
    ];
    let edges = vec![(0, 0), (1, 1)];

    let left_nodes: SortedVec<LeftSymbol> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(left_nodes.len())
        .symbols(left_nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let right_nodes: SortedVec<RightSymbol> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(right_nodes.len())
        .symbols(right_nodes.into_iter().enumerate())
        .build()
        .unwrap();

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((left_nodes.len(), right_nodes.len()))
            .edges(edges.into_iter())
            .build()
            .unwrap();

    BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap()
}

#[test]
fn test_left_and_right_node_type_markers() {
    assert_node_type(LeftKind::Entity);
    assert_node_type(RightKind::Concept);
}

#[test]
fn test_left_and_right_node_type_lookup() {
    let graph = build_graph();

    assert_eq!(graph.left_node_type(&0), Some(LeftKind::Entity));
    assert_eq!(graph.left_node_type(&1), Some(LeftKind::Action));
    assert_eq!(graph.left_node_type(&2), None);

    assert_eq!(graph.right_node_type(&0), Some(RightKind::Concept));
    assert_eq!(graph.right_node_type(&1), Some(RightKind::Token));
    assert_eq!(graph.right_node_type(&2), None);
}

#[test]
fn test_is_left_and_right_node_of_type() {
    let graph = build_graph();

    assert!(graph.is_left_node_of_type(&0, &LeftKind::Entity));
    assert!(!graph.is_left_node_of_type(&0, &LeftKind::Action));
    assert!(!graph.is_left_node_of_type(&2, &LeftKind::Entity));

    assert!(graph.is_right_node_of_type(&1, &RightKind::Token));
    assert!(!graph.is_right_node_of_type(&1, &RightKind::Concept));
    assert!(!graph.is_right_node_of_type(&2, &RightKind::Concept));
}

#[test]
fn test_has_left_and_right_node_type() {
    let graph = build_graph();

    assert!(graph.has_left_node_type(&LeftKind::Entity));
    assert!(graph.has_left_node_type(&LeftKind::Action));
    assert!(graph.has_right_node_type(&RightKind::Concept));
    assert!(graph.has_right_node_type(&RightKind::Token));
}
