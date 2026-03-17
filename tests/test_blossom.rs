//! Tests for the Edmonds blossom algorithm (maximum matching in general
//! graphs).
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

// ============================================================================
// Basic tests
// ============================================================================

#[test]
fn test_empty_graph() {
    let matrix = SquareCSR2D::<CSR2D<usize, usize, usize>>::default();
    let matching = matrix.blossom();
    validate_matching(&matrix, &matching, 0);
}

#[test]
fn test_single_edge() {
    let g = build_graph(2, &[(0, 1)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 1);
}

#[test]
fn test_path_p3() {
    // 0 - 1 - 2
    let g = build_graph(3, &[(0, 1), (1, 2)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 1);
}

#[test]
fn test_triangle() {
    // K3: odd cycle, max matching = 1
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 1);
}

#[test]
fn test_square_c4() {
    // C4: even cycle, max matching = 2
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

#[test]
fn test_pentagon_c5() {
    // C5: odd cycle, max matching = 2
    let g = build_graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

#[test]
fn test_complete_k4() {
    let g = build_graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

#[test]
fn test_path_p4() {
    // 0 - 1 - 2 - 3
    let g = build_graph(4, &[(0, 1), (1, 2), (2, 3)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

#[test]
fn test_star_graph() {
    // Center 0 connected to leaves 1, 2, 3, 4
    let g = build_graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 1);
}

#[test]
fn test_isolated_nodes() {
    let g = build_graph(4, &[]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 0);
}

#[test]
fn test_disconnected() {
    // Triangle (0,1,2) + separate edge (3,4)
    let g = build_graph(5, &[(0, 1), (0, 2), (1, 2), (3, 4)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

// ============================================================================
// Blossom-contraction tests
// ============================================================================

#[test]
fn test_blossom_required() {
    // Triangle (0,1,2) with path extending: 2-3, 3-4, 4-5
    // Max matching: {(0,1), (2,3), (4,5)} = 3
    // Requires blossom contraction of the triangle to augment through it.
    let g = build_graph(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_nested_blossom() {
    // Two overlapping odd cycles that force nested blossom contraction.
    // Triangle (0,1,2) shares vertex 2 with triangle (2,3,4), plus tail 4-5, 5-6.
    // First contraction: {0,1,2}. Second contraction encompasses {0,1,2,3,4}.
    // Max matching: {(0,1), (2,3), (4,5)} or similar, size 3.
    // Also validated against the reference crate below.
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_nested_blossom_deep() {
    // Three chained triangles with a tail, forcing deeply nested blossoms.
    // Tri 0-1-2, bridge 2-3, tri 3-4-5, bridge 5-6, tri 6-7-8, tail 8-9.
    let g = build_graph(
        10,
        &[
            (0, 1),
            (0, 2),
            (1, 2), // tri 1
            (2, 3), // bridge
            (3, 4),
            (3, 5),
            (4, 5), // tri 2
            (5, 6), // bridge
            (6, 7),
            (6, 8),
            (7, 8), // tri 3
            (8, 9), // tail
        ],
    );
    let matching = g.blossom();
    validate_matching(&g, &matching, 5);
}

#[test]
fn test_single_vertex() {
    let g = build_graph(1, &[]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 0);
}

#[test]
fn test_many_disjoint_triangles() {
    // 5 disjoint triangles = 15 vertices, max matching = 5
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
    let matching = g.blossom();
    validate_matching(&g, &matching, 5);
}

#[test]
fn test_long_path() {
    // Path P20: 0-1-2-...-19, max matching = 10
    let edges: Vec<(usize, usize)> = (0..19).map(|i| (i, i + 1)).collect();
    let g = build_graph(20, &edges);
    let matching = g.blossom();
    validate_matching(&g, &matching, 10);
}

#[test]
fn test_many_components() {
    // 6 components: K2, K3, isolated, K2, K3, isolated
    let g = build_graph(
        12,
        &[
            (0, 1), // K2
            (2, 3),
            (2, 4),
            (3, 4), // K3
            // 5 isolated
            (6, 7), // K2
            (8, 9),
            (8, 10),
            (9, 10), /* K3
                      * 11 isolated */
        ],
    );
    let matching = g.blossom();
    validate_matching(&g, &matching, 4);
}

#[test]
fn test_petersen_graph() {
    // Petersen graph: 10 vertices, 15 edges, perfect matching of size 5.
    // Outer cycle: 0-1-2-3-4-0
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    // Inner pentagram: 5-7, 7-9, 9-6, 6-8, 8-5
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
    let matching = g.blossom();
    validate_matching(&g, &matching, 5);
}

// ============================================================================
// Additional graph families
// ============================================================================

#[test]
fn test_self_loops_only() {
    // Graph with only self-loops (ignored by algorithm), max matching = 0
    let matrix = build_graph(3, &[]);
    let matching = matrix.blossom();
    validate_matching(&matrix, &matching, 0);
}

#[test]
fn test_self_loops_with_edges() {
    // Triangle with self-loops; self-loops should be ignored, max matching = 1
    let g = build_graph(3, &[(0, 1), (0, 2), (1, 2)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 1);
}

#[test]
fn test_complete_bipartite_k33() {
    // K_{3,3}: perfect matching of size 3
    let g =
        build_graph(6, &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_complete_bipartite_k27() {
    // K_{2,7}: max matching = 2
    let mut edges = Vec::new();
    for j in 2..9 {
        edges.push((0, j));
        edges.push((1, j));
    }
    let g = build_graph(9, &edges);
    let matching = g.blossom();
    validate_matching(&g, &matching, 2);
}

#[test]
fn test_wheel_w5() {
    // Wheel W5: hub 0 + cycle 1-2-3-4-5-1, max matching = 3
    let g = build_graph(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
    );
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_wheel_w7() {
    // Wheel W7: hub 0 + cycle 1-2-3-4-5-6-7-1, max matching = 4
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
    let matching = g.blossom();
    validate_matching(&g, &matching, 4);
}

#[test]
fn test_barbell() {
    // Two triangles connected by a path: 0-1-2-0, 2-3, 3-4, 4-5-6-4
    // Max matching = 3: {(0,1), (2,3), (4,5)} or similar
    let g = build_graph(7, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_cube_q3() {
    // Cube graph Q3: 8 vertices, 12 edges, perfect matching of size 4
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
    let matching = g.blossom();
    validate_matching(&g, &matching, 4);
}

#[test]
fn test_friendship_graph() {
    // Windmill / friendship graph: 3 triangles sharing hub vertex 0
    // Vertices: 0 (hub), 1-2, 3-4, 5-6
    // Max matching = 3: {(1,2), (3,4), (5,6)} — hub left exposed
    let g =
        build_graph(7, &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

#[test]
fn test_grid_2x3() {
    // 2x3 grid graph: 6 vertices, perfect matching of size 3
    // 0-1-2
    // | | |
    // 3-4-5
    let g = build_graph(6, &[(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]);
    let matching = g.blossom();
    validate_matching(&g, &matching, 3);
}

// ============================================================================
// Reference comparison
// ============================================================================

/// Builds a reference graph for the `blossom` crate from an edge list and
/// compares the matching size with our implementation.
fn assert_matching_size_agrees(n: usize, edges: &[(usize, usize)]) {
    let matrix = build_graph(n, edges);
    let our_matching = matrix.blossom();

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

    validate_matching(&matrix, &our_matching, ref_matching.len());
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
    // Two triangles connected by an edge: 0-1-2-0, 2-3, 3-4-5-3
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
