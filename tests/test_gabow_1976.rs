//! Tests for Gabow's 1976 maximum matching algorithm.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
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

fn validate_matching(
    matrix: &impl SparseSquareMatrix<Index = usize>,
    matching: &[(usize, usize)],
    expected_size: usize,
) {
    assert_eq!(matching.len(), expected_size, "matching size mismatch");
    let n: usize = matrix.order();
    let mut used = vec![false; n];
    for &(u, v) in matching {
        assert!(u < v, "pairs must be ordered u < v, got ({u}, {v})");
        assert!(matrix.has_entry(u, v), "matched edge ({u}, {v}) not in graph");
        assert!(!used[u], "vertex {u} used twice");
        assert!(!used[v], "vertex {v} used twice");
        used[u] = true;
        used[v] = true;
    }
}

fn validate_gabow(
    matrix: &(impl Blossom<Index = usize> + Blum + Gabow1976 + MicaliVazirani),
    expected_size: usize,
) {
    let gabow = matrix.gabow_1976();
    let blossom = matrix.blossom();
    let blum = matrix.blum();
    let mv = matrix.micali_vazirani();
    validate_matching(matrix, &gabow, expected_size);
    assert_eq!(gabow.len(), blossom.len(), "Gabow and Blossom disagree on size");
    assert_eq!(gabow.len(), blum.len(), "Gabow and Blum disagree on size");
    assert_eq!(gabow.len(), mv.len(), "Gabow and Micali-Vazirani disagree on size");
}

#[test]
fn test_empty_graph() {
    let matrix = SquareCSR2D::<CSR2D<usize, usize, usize>>::default();
    validate_gabow(&matrix, 0);
}

#[test]
fn test_blossom_required() {
    let g = build_graph(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
    validate_gabow(&g, 3);
}

#[test]
fn test_nested_blossom() {
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]);
    validate_gabow(&g, 3);
}

#[test]
fn test_petersen_graph() {
    let g = build_graph(
        10,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (7, 9),
        ],
    );
    validate_gabow(&g, 5);
}

#[test]
fn test_complete_bipartite_k33() {
    let g =
        build_graph(6, &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]);
    validate_gabow(&g, 3);
}

#[test]
fn test_exhaustive_small_graphs_up_to_six_vertices() {
    for n in 0..=6 {
        let mut all_edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                all_edges.push((u, v));
            }
        }

        let graph_count = 1usize << all_edges.len();
        for mask in 0..graph_count {
            let edges: Vec<(usize, usize)> = all_edges
                .iter()
                .enumerate()
                .filter_map(|(bit, &edge)| ((mask >> bit) & 1 == 1).then_some(edge))
                .collect();
            let g = build_graph(n, &edges);
            let gabow = g.gabow_1976();
            let blossom = g.blossom();
            validate_matching(&g, &gabow, blossom.len());
        }
    }
}
