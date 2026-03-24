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

fn assert_karp_sipser_kernel_irreducible_usize(
    graph: &impl SparseSquareMatrix<Index = usize>,
    rules: KarpSipserRules,
) {
    for row in graph.row_indices() {
        let degree = graph.sparse_row(row).filter(|&column| column != row).count();
        match rules {
            KarpSipserRules::Degree1 => {
                assert_ne!(degree, 1, "degree-1 kernel still contains a degree-1 vertex");
            }
            KarpSipserRules::Degree1And2 => {
                assert!(
                    degree == 0 || degree >= 3,
                    "degree-1/2 kernel still contains a reducible vertex of degree {degree}",
                );
            }
        }
    }
}

fn check_karp_sipser_invariants_usize(
    graph: &(
         impl Blossom<Index = usize>
         + Blum<Index = usize>
         + KarpSipser<Index = usize>
         + MicaliVazirani<Index = usize>
     ),
) {
    let blossom_matching = graph.blossom();
    validate_matching(graph, &blossom_matching);
    let expected_size = blossom_matching.len();

    let plain_blum_matching = graph.blum();
    validate_matching(graph, &plain_blum_matching);
    assert_eq!(
        plain_blum_matching.len(),
        expected_size,
        "plain Blum disagrees with Blossom before Karp-Sipser is applied",
    );

    for rules in [KarpSipserRules::Degree1, KarpSipserRules::Degree1And2] {
        let kernel = graph.karp_sipser_kernel(rules);
        assert_karp_sipser_kernel_irreducible_usize(kernel.graph(), rules);

        let recovered_blossom = kernel.solve_with(Blossom::blossom);
        validate_matching(graph, &recovered_blossom);
        assert_eq!(
            recovered_blossom.len(),
            expected_size,
            "Karp-Sipser blossom wrapper changed the matching size",
        );

        let explicit_recover = {
            let kernel = graph.karp_sipser_kernel(rules);
            let kernel_matching = kernel.graph().blossom();
            kernel.recover(kernel_matching)
        };
        validate_matching(graph, &explicit_recover);
        assert_eq!(
            explicit_recover.len(),
            expected_size,
            "explicit kernel recover changed the matching size",
        );

        let mv_matching = graph.micali_vazirani_with_karp_sipser(rules);
        validate_matching(graph, &mv_matching);
        assert_eq!(
            mv_matching.len(),
            expected_size,
            "Karp-Sipser Micali-Vazirani wrapper changed the matching size",
        );

        let blum_matching = graph.blum_with_karp_sipser(rules);
        validate_matching(graph, &blum_matching);
        assert_eq!(
            blum_matching.len(),
            expected_size,
            "Karp-Sipser Blum wrapper changed the matching size",
        );
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

#[test]
fn test_regression_degree1_blum_wrapper_replays_invalid_kernel_fixture() {
    // Former honggfuzz corpus entry
    // `07865582057452d55166215d7d564267.00000209.honggfuzz.cov`,
    // converted into a stable graph fixture so the regression no longer
    // depends on a local honggfuzz workspace.
    let g = build_graph(
        58,
        &[
            (0, 5),
            (0, 37),
            (0, 48),
            (5, 36),
            (5, 49),
            (36, 54),
            (37, 48),
            (37, 49),
            (45, 50),
            (45, 57),
            (48, 54),
            (49, 53),
            (50, 53),
            (50, 57),
            (52, 53),
            (52, 54),
        ],
    );

    let expected = g.blossom().len();
    let matching = g.blum_with_karp_sipser(KarpSipserRules::Degree1);
    let mut used = vec![false; g.order()];
    for &(u, v) in &matching {
        assert!(u < v, "pair must have u < v, got ({u}, {v})");
        assert!(g.has_entry(u, v), "edge ({u}, {v}) not in graph");
        assert!(!used[u], "vertex {u} matched twice");
        assert!(!used[v], "vertex {v} matched twice");
        used[u] = true;
        used[v] = true;
    }
    assert_eq!(matching.len(), expected);
}

#[test]
fn test_regression_large_karp_sipser_fixture_replays_blum_invalid_matching() {
    // Bug 1 counterexample: non-strongly-simple path reconstruction.
    // Minimized from n=58 by renumbering the 12 active vertices.
    // Original mapping: 0→0, 5→1, 36→2, 37→3, 45→4, 48→5,
    //                    49→6, 50→7, 52→8, 53→9, 54→10, 57→11.
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 6),
            (2, 3),
            (2, 6),
            (3, 6),
            (3, 7),
            (4, 7),
            (4, 11),
            (5, 10),
            (6, 9),
            (7, 10),
            (7, 11),
            (8, 9),
            (8, 10),
        ],
    );

    check_karp_sipser_invariants_usize(&g);
}

#[test]
fn test_regression_small_blum_wrapper_size_mismatch() {
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 6),
            (0, 7),
            (1, 3),
            (2, 3),
            (2, 6),
            (4, 7),
            (4, 11),
            (5, 8),
            (5, 10),
            (7, 11),
            (8, 9),
            (8, 10),
            (9, 11),
        ],
    );

    let expected = g.blossom().len();
    for rules in [KarpSipserRules::Degree1, KarpSipserRules::Degree1And2] {
        let matching = g.blum_with_karp_sipser(rules);
        let mut used = vec![false; g.order()];
        for &(u, v) in &matching {
            assert!(u < v, "pair must have u < v, got ({u}, {v})");
            assert!(g.has_entry(u, v), "edge ({u}, {v}) not in graph");
            assert!(!used[u], "vertex {u} matched twice");
            assert!(!used[v], "vertex {v} matched twice");
            used[u] = true;
            used[v] = true;
        }
        assert_eq!(
            matching.len(),
            expected,
            "Karp-Sipser Blum wrapper changed the matching size for {rules:?}",
        );
    }
}
