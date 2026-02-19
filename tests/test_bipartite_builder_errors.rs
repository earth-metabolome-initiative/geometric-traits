//! Tests for GenericMonoplexBipartiteGraphBuilder error paths and building.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    prelude::*,
    traits::{
        BipartiteGraph, BipartiteGraphBuilder, EdgesBuilder, MonoplexGraphBuilder,
        VocabularyBuilder,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestBiGraph =
    geometric_traits::naive_structs::GenericBiGraph<SortedVec<usize>, SortedVec<usize>, TestCSR>;
type BiGraphBuilder =
    geometric_traits::naive_structs::GenericMonoplexBipartiteGraphBuilder<TestBiGraph>;

fn build_nodes(n: usize) -> SortedVec<usize> {
    GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).map(|i| (i, i)))
        .build()
        .unwrap()
}

fn build_edges(rows: usize, cols: usize, edges: Vec<(usize, usize)>) -> TestCSR {
    GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// BipartiteGraphBuilder error paths
// ============================================================================

#[test]
fn test_bipartite_builder_missing_left_nodes() {
    let right = build_nodes(2);
    let edges = build_edges(2, 2, vec![(0, 1)]);

    let result = BiGraphBuilder::default().right_nodes(right).edges(edges).build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Missing attribute"));
}

#[test]
fn test_bipartite_builder_missing_right_nodes() {
    let left = build_nodes(2);
    let edges = build_edges(2, 2, vec![(0, 1)]);

    let result = BiGraphBuilder::default().left_nodes(left).edges(edges).build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Missing attribute"));
}

#[test]
fn test_bipartite_builder_missing_edges() {
    let left = build_nodes(2);
    let right = build_nodes(2);

    let result = BiGraphBuilder::default().left_nodes(left).right_nodes(right).build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Missing attribute"));
}

#[test]
fn test_bipartite_builder_missing_all() {
    let result = BiGraphBuilder::default().build();
    assert!(result.is_err());
}

#[test]
fn test_bipartite_builder_success() {
    let left = build_nodes(2);
    let right = build_nodes(3);
    let edges = build_edges(2, 3, vec![(0, 0), (1, 2)]);

    let result = BiGraphBuilder::default().left_nodes(left).right_nodes(right).edges(edges).build();
    assert!(result.is_ok());

    let graph = result.unwrap();
    assert_eq!(graph.left_nodes_vocabulary().len(), 2);
    assert_eq!(graph.right_nodes_vocabulary().len(), 3);
}

// ============================================================================
// Builder clone test
// ============================================================================

#[test]
fn test_bipartite_builder_clone() {
    let builder = BiGraphBuilder::default();
    let _cloned = builder.clone();
}
