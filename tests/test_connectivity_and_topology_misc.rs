//! Additional coverage for connectivity and graph traversal algorithms.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, VocabularyBuilder, algorithms::connected_components::ConnectedComponents,
    },
};

mod common;

use common::build_square_csr;

fn build_undigraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let num_nodes = nodes.len();
    let num_edges = edges.len();
    let node_vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_mat: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(node_vocab.len())
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((node_vocab, edge_mat))
}

#[test]
fn test_connected_components_error_conversion() {
    use geometric_traits::traits::algorithms::connected_components::ConnectedComponentsError;

    let err = ConnectedComponentsError::TooManyComponents;
    assert!(!format!("{err}").is_empty());

    let algo_err: geometric_traits::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError = err.into();
    assert!(!format!("{algo_err}").is_empty());
}

#[test]
fn test_cc_single_node() {
    let graph = build_undigraph(vec![0], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
}

#[test]
fn test_cc_all_isolated() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 4);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
}

#[test]
fn test_kahn_diamond_dag() {
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let ordering = m.kahn().unwrap();
    assert!(ordering[0] < ordering[1]);
    assert!(ordering[0] < ordering[2]);
    assert!(ordering[1] < ordering[3]);
    assert!(ordering[2] < ordering[3]);
}

#[test]
fn test_kahn_cycle_fails() {
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);
    assert!(m.kahn().is_err());
}

#[test]
fn test_kahn_wide_dag() {
    let m = build_square_csr(6, vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
    let ordering = m.kahn().unwrap();
    for i in 1..6 {
        assert!(ordering[0] < ordering[i]);
    }
}

#[test]
fn test_tarjan_disconnected_sccs() {
    let m = build_square_csr(6, vec![(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 3);
    for scc in &sccs {
        assert_eq!(scc.len(), 2);
    }
}

#[test]
fn test_tarjan_single_large_scc() {
    let m = build_square_csr(5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 5);
}

#[test]
fn test_tarjan_dag_no_nontrivial_sccs() {
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    for scc in &sccs {
        assert_eq!(scc.len(), 1, "DAG should only have singleton SCCs");
    }
}
