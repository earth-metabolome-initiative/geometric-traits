//! Tests for exact Karp-Sipser preprocessing and recovery.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::{barbell_graph, cycle_graph, grid_graph},
};

fn build_graph(n: usize, edges: &[(usize, usize)]) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let mut sorted_edges: Vec<(usize, usize)> = edges.to_vec();
    sorted_edges.sort_unstable();
    UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(n)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap()
}

fn validate_matching(matrix: &impl SparseSquareMatrix<Index = usize>, matching: &[(usize, usize)]) {
    let n: usize = matrix.order();
    let mut used = vec![false; n];
    for &(u, v) in matching {
        assert!(u < v, "pair must have u < v, got ({u}, {v})");
        assert!(matrix.has_entry(u, v), "edge ({u}, {v}) not in graph");
        assert!(!used[u], "vertex {u} matched twice");
        assert!(!used[v], "vertex {v} matched twice");
        used[u] = true;
        used[v] = true;
    }
}

#[test]
fn test_empty_graph_kernel_is_empty() {
    let g = SquareCSR2D::<CSR2D<usize, usize, usize>>::default();
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    assert_eq!(ks.graph().order(), 0);
    let recovered = ks.recover(vec![]);
    assert!(recovered.is_empty());
}

#[test]
fn test_degree1_single_edge_fully_recovers() {
    let g = build_graph(2, &[(0, 1)]);
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1);
    assert_eq!(ks.graph().order(), 0);
    let recovered = ks.recover(vec![]);
    assert_eq!(recovered, vec![(0, 1)]);
}

#[test]
fn test_degree1_path_collapses_completely() {
    let g = build_graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
    let matching = g.blossom_with_karp_sipser(KarpSipserRules::Degree1);
    validate_matching(&g, &matching);
    assert_eq!(matching.len(), 3);
}

#[test]
fn test_degree1and2_even_cycle_recovers_exact_size() {
    let g = build_graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]);
    let matching = g.blossom_with_karp_sipser(KarpSipserRules::Degree1And2);
    validate_matching(&g, &matching);
    assert_eq!(matching.len(), 3);
}

#[test]
fn test_degree2_shared_neighbor_recovers_deterministically() {
    let g = build_graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let matching = g.blossom_with_karp_sipser(KarpSipserRules::Degree1And2);
    validate_matching(&g, &matching);
    assert_eq!(matching, vec![(0, 1), (2, 3)]);
}

#[test]
fn test_degree2_unmatched_merged_vertex_recovers_exactly() {
    let g = build_graph(3, &[(0, 1), (1, 2), (0, 2)]);
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    assert_eq!(ks.graph().order(), 0);
    let recovered = ks.recover(vec![]);
    validate_matching(&g, &recovered);
    assert_eq!(recovered.len(), g.blossom().len());
}

#[test]
fn test_degree2_then_degree1_pipeline() {
    let g = build_graph(5, &[(0, 1), (1, 2), (2, 3), (1, 4)]);
    let matching = g.blossom_with_karp_sipser(KarpSipserRules::Degree1And2);
    validate_matching(&g, &matching);
    assert_eq!(matching.len(), g.blossom().len());
}

#[test]
fn test_degree1_priority_graph_still_reduces_exactly() {
    let g = build_graph(4, &[(0, 1), (1, 2), (1, 3), (2, 3)]);
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    let recovered = ks.solve_with(Blossom::blossom);
    validate_matching(&g, &recovered);
    assert_eq!(recovered.len(), 2);
}

#[test]
fn test_degree2_large_cycle_kernel_shrinks_and_recovers_exactly() {
    let g = cycle_graph(64);
    let expected = g.blossom().len();
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    assert!(ks.graph().order() < g.order());
    let recovered = ks.solve_with(Blossom::blossom);
    validate_matching(&g, &recovered);
    assert_eq!(recovered.len(), expected);
}

#[test]
fn test_degree2_ladder_kernel_shrinks_and_recovers_exactly() {
    let g = grid_graph(2, 24);
    let expected = g.blossom().len();
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    assert!(ks.graph().order() < g.order());
    let recovered = ks.solve_with(Blossom::blossom);
    validate_matching(&g, &recovered);
    assert_eq!(recovered.len(), expected);
}

#[test]
fn test_degree2_barbell_bridge_recovers_exactly() {
    let g = barbell_graph(4, 18);
    let expected = g.blossom().len();
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1And2);
    assert!(ks.graph().order() < g.order());
    let recovered = ks.solve_with(Blossom::blossom);
    validate_matching(&g, &recovered);
    assert_eq!(recovered.len(), expected);
}

#[test]
fn test_recover_panics_on_invalid_kernel_matching() {
    let g = build_graph(3, &[(0, 1), (1, 2), (0, 2)]);
    let ks = g.karp_sipser_kernel(KarpSipserRules::Degree1);
    let result = std::panic::catch_unwind(|| {
        let _ = ks.recover(vec![(0, 1), (0, 2)]);
    });
    assert!(result.is_err());
}
