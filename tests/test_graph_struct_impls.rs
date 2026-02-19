//! Tests for GenericGraph and GenericBiGraph: Debug, From/TryFrom, Graph trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::*,
    traits::{
        BipartiteGraph, EdgesBuilder, Graph, MonopartiteGraph, MonoplexGraph, VocabularyBuilder,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestBiMatrix = geometric_traits::impls::GenericBiMatrix2D<TestSquareCSR, TestSquareCSR>;
type TestGraph = geometric_traits::naive_structs::GenericGraph<SortedVec<usize>, TestBiMatrix>;

fn build_graph(n: usize, edges: Vec<(usize, usize)>) -> TestGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).map(|i| (i, i)))
        .build()
        .unwrap();

    let inner: TestSquareCSR = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    let bimatrix = geometric_traits::impls::GenericBiMatrix2D::new(inner);

    TestGraph::from((nodes, bimatrix))
}

// ============================================================================
// GenericGraph Debug impl
// ============================================================================

#[test]
fn test_generic_graph_debug() {
    let graph = build_graph(3, vec![(0, 1), (1, 2)]);
    let debug = format!("{graph:?}");
    assert!(debug.contains("GenericGraph"));
}

// ============================================================================
// GenericGraph From impl
// ============================================================================

#[test]
fn test_generic_graph_from_tuple() {
    let graph = build_graph(3, vec![(0, 1), (1, 2)]);
    assert!(graph.has_nodes());
    assert!(graph.has_edges());
}

// ============================================================================
// GenericGraph Graph trait
// ============================================================================

#[test]
fn test_generic_graph_has_edges_empty() {
    let graph = build_graph(3, vec![]);
    assert!(graph.has_nodes());
    assert!(!graph.has_edges());
}

#[test]
fn test_generic_graph_nodes_vocabulary() {
    let graph = build_graph(3, vec![(0, 1)]);
    let vocab = graph.nodes_vocabulary();
    assert_eq!(vocab.len(), 3);
}

#[test]
fn test_generic_graph_edges_accessor() {
    let graph = build_graph(3, vec![(0, 1), (1, 2)]);
    let edges = graph.edges();
    assert_eq!(edges.number_of_defined_values(), 2);
}

// ============================================================================
// GenericBiGraph tests
// ============================================================================

type TestBiGraph =
    geometric_traits::naive_structs::GenericBiGraph<SortedVec<usize>, SortedVec<usize>, TestCSR>;

fn build_bigraph(left_n: usize, right_n: usize, edges: Vec<(usize, usize)>) -> TestBiGraph {
    let left: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(left_n)
        .symbols((0..left_n).map(|i| (i, i)))
        .build()
        .unwrap();

    let right: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(right_n)
        .symbols((0..right_n).map(|i| (i, i + 100)))
        .build()
        .unwrap();

    let csr: TestCSR = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((left_n, right_n))
        .edges(edges.into_iter())
        .build()
        .unwrap();

    TestBiGraph::try_from((left, right, csr)).unwrap()
}

#[test]
fn test_generic_bigraph_debug() {
    let graph = build_bigraph(2, 2, vec![(0, 1)]);
    let debug = format!("{graph:?}");
    assert!(debug.contains("GenericBiGraph"));
    assert!(debug.contains("left_nodes"));
    assert!(debug.contains("right_nodes"));
}

#[test]
fn test_generic_bigraph_try_from() {
    let graph = build_bigraph(2, 3, vec![(0, 0), (1, 2)]);
    assert!(graph.has_nodes());
    assert!(graph.has_edges());
}

#[test]
fn test_generic_bigraph_has_edges_empty() {
    let graph = build_bigraph(2, 2, vec![]);
    assert!(graph.has_nodes());
    assert!(!graph.has_edges());
}

#[test]
fn test_generic_bigraph_bipartite_graph_trait() {
    let graph = build_bigraph(2, 3, vec![(0, 0), (1, 1)]);

    let left = graph.left_nodes_vocabulary();
    assert_eq!(left.len(), 2);

    let right = graph.right_nodes_vocabulary();
    assert_eq!(right.len(), 3);
}

#[test]
fn test_generic_bigraph_monoplex_graph_trait() {
    let graph = build_bigraph(2, 3, vec![(0, 0), (0, 1), (1, 2)]);
    let edges = graph.edges();
    assert_eq!(edges.number_of_defined_values(), 3);
}

#[test]
fn test_generic_bigraph_clone() {
    let graph = build_bigraph(2, 2, vec![(0, 1)]);
    let cloned = graph.clone();
    assert!(cloned.has_edges());
}
