//! Test submodule for the `CycleDetection` trait.

use geometric_traits::impls::SortedVec;
use geometric_traits::impls::SquareCSR2D;
use geometric_traits::{
    prelude::{
        CycleDetection, DiEdgesBuilder, DiGraph, GenericVocabularyBuilder, MonopartiteGraph,
        MonoplexGraph,
    },
    traits::{EdgesBuilder, VocabularyBuilder},
};

#[test]
fn test_no_cycle_detection() -> Result<(), Box<dyn std::error::Error>> {
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
    let edges: Vec<(usize, usize)> = vec![(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)];
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()?;
    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(nodes.len())
        .edges(edges.into_iter())
        .build()?;
    let graph: DiGraph<usize> = DiGraph::from((nodes, edges));

    assert_eq!(graph.number_of_nodes(), 6);
    assert_eq!(graph.number_of_edges(), 5);

    assert!(!graph.has_cycle());

    Ok(())
}

#[test]
fn test_cycle_detection() -> Result<(), Box<dyn std::error::Error>> {
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
    let edges: Vec<(usize, usize)> = vec![(1, 2), (1, 3), (2, 3), (3, 2), (3, 4), (4, 5)];
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()?;
    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(nodes.len())
        .edges(edges.into_iter())
        .build()?;
    let graph: DiGraph<usize> = DiGraph::from((nodes, edges));

    assert_eq!(graph.number_of_nodes(), 6);
    assert_eq!(graph.number_of_edges(), 6);

    assert!(graph.has_cycle());

    Ok(())
}
