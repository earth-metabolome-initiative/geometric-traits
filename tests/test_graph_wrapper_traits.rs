//! Tests for newly added `Graph`/`MonoplexGraph` impls on CSR wrapper types.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D},
    prelude::*,
    traits::{Graph, MonoplexGraph},
};

#[test]
fn test_square_csr2d_graph_has_nodes() {
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(3)
        .edges(vec![(0, 1), (1, 2), (2, 0)].into_iter())
        .build()
        .unwrap();

    assert!(Graph::has_nodes(&edges));
    assert!(Graph::has_edges(&edges));
}

#[test]
fn test_square_csr2d_graph_empty() {
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(0)
        .edges(core::iter::empty())
        .build()
        .unwrap();

    assert!(!Graph::has_nodes(&edges));
    assert!(!Graph::has_edges(&edges));
}

#[test]
fn test_square_csr2d_monoplex_graph_edges() {
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(2)
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build()
        .unwrap();

    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 0).collect();
    assert_eq!(succs, vec![1]);
    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 1).collect();
    assert_eq!(succs, vec![2]);
}

#[test]
fn test_upper_triangular_csr2d_graph_has_nodes() {
    let edges: UpperTriangularCSR2D<CSR2D<usize, usize, usize>> = GenericEdgesBuilder::default()
        .expected_number_of_edges(3)
        .edges(vec![(0, 1), (1, 2), (2, 3)].into_iter())
        .build()
        .unwrap();

    assert!(Graph::has_nodes(&edges));
    assert!(Graph::has_edges(&edges));
}

#[test]
fn test_upper_triangular_csr2d_graph_empty() {
    let edges: UpperTriangularCSR2D<CSR2D<usize, usize, usize>> = GenericEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(0)
        .edges(core::iter::empty())
        .build()
        .unwrap();

    assert!(!Graph::has_nodes(&edges));
    assert!(!Graph::has_edges(&edges));
}

#[test]
fn test_upper_triangular_csr2d_monoplex_graph_edges() {
    let edges: UpperTriangularCSR2D<CSR2D<usize, usize, usize>> = GenericEdgesBuilder::default()
        .expected_number_of_edges(2)
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build()
        .unwrap();

    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 0).collect();
    assert_eq!(succs, vec![1]);
    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 1).collect();
    assert_eq!(succs, vec![2]);
}

#[test]
fn test_symmetric_csr2d_graph_has_nodes() {
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(2)
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build()
        .unwrap();

    assert!(Graph::has_nodes(&edges));
    assert!(Graph::has_edges(&edges));
}

#[test]
fn test_symmetric_csr2d_graph_empty() {
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(0)
        .edges(core::iter::empty())
        .build()
        .unwrap();

    assert!(!Graph::has_nodes(&edges));
    assert!(!Graph::has_edges(&edges));
}

#[test]
fn test_symmetric_csr2d_monoplex_graph_edges() {
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(2)
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build()
        .unwrap();

    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 0).collect();
    assert_eq!(succs, vec![1]);
    let succs: Vec<usize> = MonoplexGraph::successors(&edges, 1).collect();
    assert_eq!(succs, vec![0, 2]);
}
