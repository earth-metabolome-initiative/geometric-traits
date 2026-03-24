//! Tests for Blum's maximum matching algorithm.
//! Mirrors tests/test_micali_vazirani.rs with cross-validation against Blossom.
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

/// Validates Blum matching and cross-checks size against Blossom.
fn validate_blum(matrix: &(impl Blossom<Index = usize> + Blum), expected_size: usize) {
    let blum = matrix.blum();
    let bl = matrix.blossom();
    validate_matching(matrix, &blum, expected_size);
    assert_eq!(blum.len(), bl.len(), "Blum and Blossom disagree on matching size");
}

// ============================================================================
// Basic tests
// ============================================================================

#[test]
fn test_empty_graph() {
    let matrix = SquareCSR2D::<CSR2D<usize, usize, usize>>::default();
    validate_blum(&matrix, 0);
}

#[test]
fn test_single_edge() {
    let g = build_graph(2, &[(0, 1)]);
    validate_blum(&g, 1);
}

#[test]
fn test_path_p3() {
    let g = build_graph(3, &[(0, 1), (1, 2)]);
    validate_blum(&g, 1);
}

#[test]
fn test_triangle() {
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    validate_blum(&g, 1);
}

#[test]
fn test_square_c4() {
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
    validate_blum(&g, 2);
}

#[test]
fn test_pentagon_c5() {
    let g = build_graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
    validate_blum(&g, 2);
}

#[test]
fn test_complete_k4() {
    let g = build_graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    validate_blum(&g, 2);
}

#[test]
fn test_path_p4() {
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3)]);
    validate_blum(&g, 2);
}

#[test]
fn test_star_graph() {
    let g = build_graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
    validate_blum(&g, 1);
}

#[test]
fn test_isolated_nodes() {
    let g = build_graph(4, &[]);
    validate_blum(&g, 0);
}

#[test]
fn test_disconnected() {
    let g = build_graph(5, &[(0, 1), (0, 2), (1, 2), (3, 4)]);
    validate_blum(&g, 2);
}

// ============================================================================
// Blossom-contraction tests
// ============================================================================

#[test]
fn test_blossom_required() {
    let g = build_graph(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
    validate_blum(&g, 3);
}

#[test]
fn test_nested_blossom() {
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]);
    validate_blum(&g, 3);
}

#[test]
fn test_nested_blossom_deep() {
    let g = build_graph(
        10,
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (7, 8),
            (8, 9),
        ],
    );
    validate_blum(&g, 5);
}

#[test]
fn test_single_vertex() {
    let g = build_graph(1, &[]);
    validate_blum(&g, 0);
}

#[test]
fn test_many_disjoint_triangles() {
    let g = build_graph(
        15,
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 5),
            (6, 7),
            (6, 8),
            (7, 8),
            (9, 10),
            (9, 11),
            (10, 11),
            (12, 13),
            (12, 14),
            (13, 14),
        ],
    );
    validate_blum(&g, 5);
}

#[test]
fn test_long_path() {
    let edges: Vec<(usize, usize)> = (0..19).map(|i| (i, i + 1)).collect();
    let g = build_graph(20, &edges);
    validate_blum(&g, 10);
}

#[test]
fn test_many_components() {
    let g = build_graph(12, &[(0, 1), (2, 3), (2, 4), (3, 4), (6, 7), (8, 9), (8, 10), (9, 10)]);
    validate_blum(&g, 4);
}

#[test]
fn test_regression_invalid_non_edge_from_degree1_kernel() {
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 6),
            (2, 10),
            (3, 5),
            (3, 6),
            (4, 7),
            (4, 11),
            (5, 10),
            (6, 9),
            (7, 9),
            (7, 11),
            (8, 9),
            (8, 10),
        ],
    );
    validate_blum(&g, 6);
}

#[test]
fn test_regression_small_plain_blum_size_mismatch() {
    // Bug 3 counterexample (only): MBFS DOM exclusion gap.
    // MBFS leaves level[t] = INF because the DOM node's twin is
    // excluded from level assignment and no other bridge provides it.
    // Requires: per-vertex MDFS fallback when level[t] = INF.
    let g = build_graph(
        15,
        &[
            (0, 9),
            (0, 14),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 7),
            (4, 8),
            (4, 11),
            (5, 7),
            (6, 10),
            (6, 11),
            (8, 11),
            (9, 13),
            (10, 14),
        ],
    );
    let blossom = g.blossom();
    let blum = g.blum();
    let mut used = vec![false; g.order()];
    for &(u, v) in &blum {
        assert!(u < v, "pair must have u < v, got ({u}, {v})");
        assert!(g.has_entry(u, v), "edge ({u}, {v}) not in graph");
        assert!(!used[u], "vertex {u} matched twice");
        assert!(!used[v], "vertex {v} matched twice");
        used[u] = true;
        used[v] = true;
    }
    assert_eq!(blum.len(), blossom.len(), "Blum and Blossom disagree on matching size");
}

#[test]
fn test_regression_reused_vertex_from_degree1_kernel_corpus() {
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 6),
            (2, 3),
            (3, 5),
            (3, 6),
            (4, 7),
            (4, 9),
            (4, 11),
            (5, 10),
            (6, 9),
            (7, 9),
            (7, 11),
            (8, 9),
            (8, 10),
        ],
    );
    validate_blum(&g, 6);
}

#[test]
fn test_regression_non_edge_from_degree1_kernel_corpus_two() {
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 6),
            (2, 3),
            (2, 10),
            (3, 5),
            (3, 6),
            (4, 7),
            (4, 9),
            (4, 11),
            (5, 10),
            (6, 9),
            (7, 9),
            (7, 11),
            (8, 9),
            (8, 10),
        ],
    );
    validate_blum(&g, 6);
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
    validate_blum(&g, 5);
}

// ============================================================================
// Additional graph families
// ============================================================================

#[test]
fn test_self_loops_only() {
    let matrix = build_graph(3, &[]);
    validate_blum(&matrix, 0);
}

#[test]
fn test_self_loops_with_edges() {
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    validate_blum(&g, 1);
}

#[test]
fn test_complete_bipartite_k33() {
    let g =
        build_graph(6, &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]);
    validate_blum(&g, 3);
}

#[test]
fn test_complete_bipartite_k27() {
    let mut edges = Vec::new();
    for j in 2..9 {
        edges.push((0, j));
        edges.push((1, j));
    }
    let g = build_graph(9, &edges);
    validate_blum(&g, 2);
}

#[test]
fn test_wheel_w5() {
    let g = build_graph(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
    );
    validate_blum(&g, 3);
}

#[test]
fn test_wheel_w7() {
    let g = build_graph(
        8,
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (1, 7),
        ],
    );
    validate_blum(&g, 4);
}

#[test]
fn test_barbell() {
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]);
    validate_blum(&g, 3);
}

#[test]
fn test_cube_q3() {
    let g = build_graph(
        8,
        &[
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ],
    );
    validate_blum(&g, 4);
}

#[test]
fn test_friendship_graph() {
    let g =
        build_graph(7, &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)]);
    validate_blum(&g, 3);
}

#[test]
fn test_grid_2x3() {
    let g = build_graph(6, &[(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]);
    validate_blum(&g, 3);
}

// ============================================================================
// Regression cases
// ============================================================================

#[test]
fn test_regression_random_counterexample_invalid_matching() {
    let g = build_graph(
        8,
        &[
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 3),
            (2, 5),
            (2, 6),
            (3, 6),
            (4, 5),
            (6, 7),
        ],
    );
    validate_blum(&g, 4);
}

#[test]
fn test_regression_random_counterexample_repeated_vertex() {
    let g = build_graph(
        10,
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 3),
            (2, 6),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 8),
            (3, 9),
            (4, 6),
            (4, 7),
            (4, 8),
            (5, 6),
            (5, 7),
            (6, 7),
            (8, 9),
        ],
    );
    validate_blum(&g, 5);
}

#[test]
fn test_regression_random_counterexample_size_mismatch() {
    let g = build_graph(
        8,
        &[(0, 2), (0, 3), (0, 4), (0, 7), (1, 2), (1, 3), (3, 7), (4, 5), (4, 6), (4, 7), (5, 7)],
    );
    validate_blum(&g, 4);
}

#[test]
fn test_regression_phase_progression_stalls_before_maximum() {
    // Bug 2 + Bug 3 counterexample: triggers both bugs on the same graph.
    // Bug 3: MBFS leaves level[t] = INF (DOM exclusion gap).
    // Bug 2: the per-vertex MDFS fallback is also needed because the
    //   standard single-source fallback suffers from subtree poisoning.
    // Requires: per-vertex MDFS fallback for BOTH the level[t]=INF
    //   path AND the found==0 path.
    let g = build_graph(
        12,
        &[
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 6),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (4, 9),
            (4, 11),
            (5, 10),
            (6, 9),
            (7, 9),
            (7, 11),
            (8, 9),
            (8, 10),
        ],
    );
    validate_blum(&g, 6);
}

#[test]
fn test_regression_large_fixture_blum_size_mismatch() {
    // Bug 2 counterexample (only): DFS subtree poisoning.
    // The single-source MDFS explores from b(10) first, marking a(8)
    // as visited. When it later tries b(11)->a(8), the label mechanism
    // fails to bridge across subtrees.
    // Requires: per-vertex MDFS fallback when layered MDFS finds 0 paths.
    // Minimized from n=119 by renumbering the 13 active vertices.
    // Original mapping: 25→0, 49→1, 50→2, 52→3, 53→4, 54→5, 55→6,
    //                    56→7, 57→8, 58→9, 61→10, 73→11, 118→12.
    let g = build_graph(
        13,
        &[
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 8),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 8),
            (2, 4),
            (3, 7),
            (3, 10),
            (4, 6),
            (5, 6),
            (5, 9),
            (5, 10),
            (7, 10),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
        ],
    );

    let n = g.order();
    assert!(n <= 128, "fuzz regression should stay within the harness size cap");

    let blum_matching = g.blum();
    let blossom_matching = g.blossom();
    assert_eq!(
        blum_matching.len(),
        blossom_matching.len(),
        "Blum and Blossom disagree on matching size (n={n})",
    );

    let mut matched = vec![false; n];
    for &(u, v) in &blum_matching {
        let left = u;
        let right = v;
        assert!(u < v, "pair must have u < v, got ({u}, {v})");
        assert!(!matched[left], "vertex {u} matched twice");
        assert!(!matched[right], "vertex {v} matched twice");
        matched[left] = true;
        matched[right] = true;
        assert!(g.has_entry(u, v), "matched edge ({u}, {v}) not in graph");
    }

    for u in g.row_indices() {
        if matched[u] {
            continue;
        }
        for w in g.sparse_row(u) {
            assert!(w == u || matched[w], "edge ({u}, {w}) has both endpoints unmatched");
        }
    }
}

// ============================================================================
// Dandeh & Lukovszki (ICTCS 2025) counterexample graphs
//
// These graphs are transcribed from Figures 1 and 2 of Dandeh & Lukovszki,
// "Experimental Evaluation of Blum's Maximum Matching Algorithm in General
// Graphs," CEUR-WS Vol. 4039, 2025.  They demonstrate the two MDFS bugs
// (Cases 2.2.i and 2.3.i) that D&L identified and corrected.
// ============================================================================

/// Figure 1: Case 2.2.i (weak back edge).
///
/// 10-vertex graph.  Without the D&L Case 2.2.i R-set fix, the
/// backward search fails to set P[7_A], causing path reconstruction
/// to fail and the augmenting path 1-9-8-7-6-4-5-3-2-10 to be missed.
/// Maximum matching size: 5.
#[test]
fn test_dandeh_lukovszki_figure1_case_2_2_i() {
    let g = build_graph(
        11,
        &[
            (1, 2),
            (1, 9),
            (2, 3),
            (2, 10),
            (3, 5),
            (3, 9),
            (4, 5),
            (4, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (7, 8),
            (8, 9),
        ],
    );
    validate_blum(&g, 5);
}

/// Figure 2: Case 2.3.i (forward/cross edge with L = empty).
///
/// 13-vertex graph (vertices 1-12, 0-indexed as 0-12 in our representation
/// but the paper uses 1-indexed).  Without the D&L Case 2.3.i WC-set fix,
/// P[9_A] is undefined during reconstruction, causing the augmenting path
/// 1-11-10-9-8-6-7-5-4-3-2-12 to be missed.
/// Maximum matching size: 6.
#[test]
fn test_dandeh_lukovszki_figure2_case_2_3_i() {
    let g = build_graph(
        13,
        &[
            (1, 2),
            (1, 11),
            (2, 3),
            (2, 12),
            (3, 4),
            (3, 5),
            (3, 7),
            (4, 5),
            (4, 10),
            (5, 6),
            (5, 9),
            (6, 7),
            (8, 9),
            (9, 10),
            (10, 11),
        ],
    );
    validate_blum(&g, 6);
}

// ============================================================================
// Reference comparison (cross-validate with external blossom crate)
// ============================================================================

fn assert_matching_size_agrees(n: usize, edges: &[(usize, usize)]) {
    let matrix = build_graph(n, edges);
    let blum_matching = matrix.blum();
    let bl_matching = matrix.blossom();

    let adj: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|v| {
            let neighbors: Vec<usize> = edges
                .iter()
                .filter_map(|&(a, b)| {
                    if a == v {
                        Some(b)
                    } else if b == v {
                        Some(a)
                    } else {
                        None
                    }
                })
                .collect();
            (v, neighbors)
        })
        .collect();
    let ref_graph: blossom::Graph = adj.iter().collect();
    let ref_matching = ref_graph.maximum_matching();

    validate_matching(&matrix, &blum_matching, ref_matching.len());
    assert_eq!(
        blum_matching.len(),
        bl_matching.len(),
        "Blum and Blossom disagree on matching size"
    );
}

#[test]
fn test_reference_triangle() {
    assert_matching_size_agrees(3, &[(0, 1), (0, 2), (1, 2)]);
}

#[test]
fn test_reference_path_p5() {
    assert_matching_size_agrees(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
}

#[test]
fn test_reference_k4() {
    assert_matching_size_agrees(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
}

#[test]
fn test_reference_petersen() {
    assert_matching_size_agrees(
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
}

#[test]
fn test_reference_blossom_required() {
    assert_matching_size_agrees(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
}

#[test]
fn test_reference_two_blossoms() {
    assert_matching_size_agrees(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]);
}

#[test]
fn test_reference_hexagon() {
    assert_matching_size_agrees(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]);
}

#[test]
fn test_reference_complete_k5() {
    assert_matching_size_agrees(
        5,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    );
}

#[test]
fn test_reference_nested_blossom() {
    assert_matching_size_agrees(
        7,
        &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)],
    );
}

#[test]
fn test_reference_nested_blossom_deep() {
    assert_matching_size_agrees(
        10,
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (7, 8),
            (8, 9),
        ],
    );
}

#[test]
fn test_reference_complete_k7() {
    let mut edges = Vec::new();
    for i in 0..7 {
        for j in (i + 1)..7 {
            edges.push((i, j));
        }
    }
    assert_matching_size_agrees(7, &edges);
}

#[test]
fn test_reference_k33() {
    let mut edges = Vec::new();
    for i in 0..3 {
        for j in 3..6 {
            edges.push((i, j));
        }
    }
    assert_matching_size_agrees(6, &edges);
}

#[test]
fn test_reference_wheel_w5() {
    assert_matching_size_agrees(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
    );
}

#[test]
fn test_reference_barbell() {
    assert_matching_size_agrees(
        7,
        &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)],
    );
}

#[test]
fn test_reference_cube_q3() {
    assert_matching_size_agrees(
        8,
        &[
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ],
    );
}

#[test]
fn test_reference_friendship() {
    assert_matching_size_agrees(
        7,
        &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)],
    );
}
