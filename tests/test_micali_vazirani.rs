//! Tests for the Micali-Vazirani maximum matching algorithm.
//! Mirrors tests/test_blossom.rs with cross-validation against Blossom.
#![cfg(feature = "std")]

#[path = "support/max_matching_oracle.rs"]
mod max_matching_oracle;

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

/// Validates MV matching and cross-checks size against Blossom.
fn validate_mv(
    matrix: &(impl Blossom<Index = usize> + MicaliVazirani),
    _edges: &[(usize, usize)],
    expected_size: usize,
) {
    let mv = matrix.micali_vazirani();
    let bl = matrix.blossom();
    validate_matching(matrix, &mv, expected_size);
    assert_eq!(mv.len(), bl.len(), "MV and Blossom disagree on matching size");
}

// ============================================================================
// Basic tests
// ============================================================================

#[test]
fn test_empty_graph() {
    let matrix = SquareCSR2D::<CSR2D<usize, usize, usize>>::default();
    validate_mv(&matrix, &[], 0);
}

#[test]
fn test_single_edge() {
    let g = build_graph(2, &[(0, 1)]);
    validate_mv(&g, &[(0, 1)], 1);
}

#[test]
fn test_path_p3() {
    let g = build_graph(3, &[(0, 1), (1, 2)]);
    validate_mv(&g, &[(0, 1), (1, 2)], 1);
}

#[test]
fn test_triangle() {
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    validate_mv(&g, &[], 1);
}

#[test]
fn test_square_c4() {
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
    validate_mv(&g, &[], 2);
}

#[test]
fn test_pentagon_c5() {
    let g = build_graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
    validate_mv(&g, &[], 2);
}

#[test]
fn test_complete_k4() {
    let g = build_graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    validate_mv(&g, &[], 2);
}

#[test]
fn test_path_p4() {
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3)]);
    validate_mv(&g, &[], 2);
}

#[test]
fn test_star_graph() {
    let g = build_graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
    validate_mv(&g, &[], 1);
}

#[test]
fn test_isolated_nodes() {
    let g = build_graph(4, &[]);
    validate_mv(&g, &[], 0);
}

#[test]
fn test_disconnected() {
    let g = build_graph(5, &[(0, 1), (0, 2), (1, 2), (3, 4)]);
    validate_mv(&g, &[], 2);
}

// ============================================================================
// Blossom-contraction tests
// ============================================================================

#[test]
fn test_blossom_required() {
    let g = build_graph(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
    validate_mv(&g, &[], 3);
}

#[test]
fn test_nested_blossom() {
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]);
    validate_mv(&g, &[], 3);
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
    validate_mv(&g, &[], 5);
}

#[test]
fn test_single_vertex() {
    let g = build_graph(1, &[]);
    validate_mv(&g, &[], 0);
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
    validate_mv(&g, &[], 5);
}

#[test]
fn test_long_path() {
    let edges: Vec<(usize, usize)> = (0..19).map(|i| (i, i + 1)).collect();
    let g = build_graph(20, &edges);
    validate_mv(&g, &edges, 10);
}

#[test]
fn test_many_components() {
    let g = build_graph(12, &[(0, 1), (2, 3), (2, 4), (3, 4), (6, 7), (8, 9), (8, 10), (9, 10)]);
    validate_mv(&g, &[], 4);
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
    validate_mv(&g, &[], 5);
}

// ============================================================================
// Additional graph families
// ============================================================================

#[test]
fn test_self_loops_only() {
    let matrix = build_graph(3, &[]);
    validate_mv(&matrix, &[], 0);
}

#[test]
fn test_self_loops_with_edges() {
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    validate_mv(&g, &[], 1);
}

#[test]
fn test_complete_bipartite_k33() {
    let g =
        build_graph(6, &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]);
    validate_mv(&g, &[], 3);
}

#[test]
fn test_complete_bipartite_k27() {
    let mut edges = Vec::new();
    for j in 2..9 {
        edges.push((0, j));
        edges.push((1, j));
    }
    let g = build_graph(9, &edges);
    validate_mv(&g, &edges, 2);
}

#[test]
fn test_wheel_w5() {
    let g = build_graph(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
    );
    validate_mv(&g, &[], 3);
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
    validate_mv(&g, &[], 4);
}

#[test]
fn test_barbell() {
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]);
    validate_mv(&g, &[], 3);
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
    validate_mv(&g, &[], 4);
}

#[test]
fn test_friendship_graph() {
    let g =
        build_graph(7, &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)]);
    validate_mv(&g, &[], 3);
}

#[test]
fn test_grid_2x3() {
    let g = build_graph(6, &[(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]);
    validate_mv(&g, &[], 3);
}

// ============================================================================
// Reference comparison (cross-validate with exact oracle on small graphs)
// ============================================================================

fn assert_matching_size_agrees(n: usize, edges: &[(usize, usize)]) {
    let matrix = build_graph(n, edges);
    let mv_matching = matrix.micali_vazirani();
    let bl_matching = matrix.blossom();
    let oracle_size = max_matching_oracle::maximum_matching_size(n, edges);

    validate_matching(&matrix, &mv_matching, oracle_size);
    assert_eq!(mv_matching.len(), bl_matching.len(), "MV and Blossom disagree on matching size");
    assert_eq!(bl_matching.len(), oracle_size, "Blossom disagrees with the exact oracle");
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

// ============================================================================
// Minimal reproducer from fuzz crash
// ============================================================================

#[test]
fn test_augmenting_path_through_blossom() {
    // Relabeled version of crash 002. Vertices: 0..8
    // Edges: (0,1),(0,3),(0,5),(1,5),(2,5),(2,6),(4,6),(4,7),(5,6)
    // Optimal matching: (0,3),(1,5),(2,6),(4,7) = 4
    let g =
        build_graph(8, &[(0, 1), (0, 3), (0, 5), (1, 5), (2, 5), (2, 6), (4, 6), (4, 7), (5, 6)]);
    validate_mv(&g, &[], 4);
}

// ============================================================================
// Fuzz crash reproducers
// ============================================================================

#[cfg(feature = "arbitrary")]
fn run_fuzz_input(data: &[u8]) {
    let mut u = arbitrary::Unstructured::new(data);
    if let Ok(csr) = <SymmetricCSR2D<CSR2D<u16, u8, u8>> as arbitrary::Arbitrary>::arbitrary(&mut u)
    {
        let n = csr.order() as usize;
        if n > 128 {
            return;
        }
        let mv = csr.micali_vazirani();
        let bl = csr.blossom();
        assert_eq!(mv.len(), bl.len(), "MV({}) != Blossom({}) for n={n}", mv.len(), bl.len());
    }
}

#[cfg(feature = "arbitrary")]
#[test]
fn test_fuzz_crash_001() {
    run_fuzz_input(&[0x00, 0x32, 0x33, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00]);
}

#[cfg(feature = "arbitrary")]
#[test]
fn test_fuzz_crash_002() {
    run_fuzz_input(&[
        0x05, 0x55, 0x55, 0x54, 0x56, 0x55, 0x55, 0x55, 0x55, 0x54, 0x57, 0x55, 0x56, 0x55, 0x55,
        0x55, 0x55, 0x55, 0x55, 0x05, 0x55, 0x55, 0x55, 0x55, 0x56, 0x05, 0x55, 0x00, 0x0f, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x55, 0x00, 0xff, 0x55, 0x01, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x04, 0x01, 0x00, 0x03, 0x33, 0x33, 0xfe, 0xff, 0xff, 0xff, 0x10, 0x00,
        0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x33, 0x33, 0x35, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x21, 0x12, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf9, 0xff,
        0x15, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x00,
        0x00, 0xff, 0x00, 0x00, 0xff,
    ]);
}

#[cfg(feature = "arbitrary")]
#[test]
fn test_fuzz_crash_003() {
    run_fuzz_input(&[
        0x05, 0x55, 0x55, 0x54, 0x56, 0x55, 0x55, 0x55, 0x55, 0x54, 0x57, 0x55, 0x56, 0x55, 0x55,
        0x34, 0x33, 0x37, 0x55, 0x05, 0x55, 0x55, 0x55, 0x55, 0x56, 0x05, 0x55, 0x00, 0xff, 0x55,
        0x01, 0x35, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x03, 0x01, 0x00, 0x05, 0x55, 0x00, 0xff,
        0x55, 0x01, 0x35, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x03, 0x01, 0x00, 0x03, 0x33, 0x33,
        0xfe, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x33, 0x33,
        0x35, 0x00, 0xcd, 0xf9, 0xff, 0x15, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff,
    ]);
}
