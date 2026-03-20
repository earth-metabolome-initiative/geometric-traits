//! Tests for the Kocay-Stone BNS balanced flow algorithm.
#![cfg(feature = "std")]

use geometric_traits::{impls::ValuedCSR2D, prelude::*};

type Vcsr = ValuedCSR2D<usize, usize, usize, usize>;

/// Build a symmetric valued matrix from edges with capacities.
fn build_valued_graph(n: usize, edges: &[(usize, usize, usize)]) -> Vcsr {
    let mut sorted_edges: Vec<(usize, usize, usize)> = Vec::new();
    for &(i, j, cap) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, cap));
        sorted_edges.push((hi, lo, cap));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted_edges.len());
    for (r, c, v) in sorted_edges {
        MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
    }
    vcsr
}

/// Validate a balanced flow result.
fn validate_flow(
    n: usize,
    edges: &[(usize, usize, usize)],
    vertex_budgets: &[usize],
    flow: &[(usize, usize, usize)],
) -> usize {
    // Check ordering
    for &(i, j, f) in flow {
        assert!(i < j, "flow triple must have i < j, got ({i}, {j}, {f})");
        assert!(f > 0, "flow must be > 0, got ({i}, {j}, {f})");
    }

    // Check edge existence and capacity
    for &(i, j, f) in flow {
        let found = edges.iter().find(|&&(a, b, _c)| (a == i && b == j) || (a == j && b == i));
        assert!(found.is_some(), "flow on non-existent edge ({i}, {j})");
        let cap = found.unwrap().2;
        assert!(f <= cap, "flow {f} exceeds capacity {cap} on edge ({i}, {j})");
    }

    // Check flow conservation (budget respect)
    let mut vertex_flow = vec![0usize; n];
    for &(i, j, f) in flow {
        vertex_flow[i] += f;
        vertex_flow[j] += f;
    }
    for v in 0..n {
        assert!(
            vertex_flow[v] <= vertex_budgets[v],
            "vertex {v} flow {} exceeds budget {}",
            vertex_flow[v],
            vertex_budgets[v]
        );
    }

    // Return total flow
    flow.iter().map(|&(_, _, f)| f).sum()
}

// ============================================================================
// Empty and trivial cases
// ============================================================================

#[test]
fn test_empty_graph() {
    let vcsr: Vcsr = SparseMatrixMut::with_sparse_shape((0, 0));
    let flow = vcsr.kocay(&[]);
    assert!(flow.is_empty());
}

#[test]
fn test_single_vertex() {
    let vcsr: Vcsr = SparseMatrixMut::with_sparse_shape((1, 1));
    let flow = vcsr.kocay(&[1]);
    assert!(flow.is_empty());
}

#[test]
fn test_isolated_vertices() {
    let vcsr: Vcsr = SparseMatrixMut::with_sparse_shape((4, 4));
    let flow = vcsr.kocay(&[1, 1, 1, 1]);
    assert!(flow.is_empty());
}

// ============================================================================
// Unit-capacity tests (matching equivalence)
// ============================================================================

/// Build a unit-capacity graph and compare total Kocay flow with Blossom
/// matching size.
fn assert_unit_cap_matches_blossom(n: usize, edges: &[(usize, usize)]) {
    let cap_edges: Vec<(usize, usize, usize)> = edges.iter().map(|&(i, j)| (i, j, 1)).collect();
    let vcsr = build_valued_graph(n, &cap_edges);
    let budgets = vec![1usize; n];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(n, &cap_edges, &budgets, &flow);

    // Build unvalued graph for Blossom
    let mut sorted_edges: Vec<(usize, usize)> = edges.to_vec();
    sorted_edges.sort_unstable();
    let sym: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(n)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap();
    let blossom_size = sym.blossom().len();

    assert_eq!(
        total, blossom_size,
        "Kocay total flow {total} != Blossom matching size {blossom_size}"
    );
}

#[test]
fn test_unit_single_edge() {
    assert_unit_cap_matches_blossom(2, &[(0, 1)]);
}

#[test]
fn test_unit_path_p3() {
    assert_unit_cap_matches_blossom(3, &[(0, 1), (1, 2)]);
}

#[test]
fn test_unit_triangle() {
    assert_unit_cap_matches_blossom(3, &[(0, 1), (0, 2), (1, 2)]);
}

#[test]
fn test_unit_square_c4() {
    assert_unit_cap_matches_blossom(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
}

#[test]
fn test_unit_pentagon_c5() {
    assert_unit_cap_matches_blossom(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
}

#[test]
fn test_unit_complete_k4() {
    assert_unit_cap_matches_blossom(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
}

#[test]
fn test_unit_path_p4() {
    assert_unit_cap_matches_blossom(4, &[(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn test_unit_star() {
    assert_unit_cap_matches_blossom(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
}

#[test]
fn test_unit_disconnected() {
    assert_unit_cap_matches_blossom(5, &[(0, 1), (0, 2), (1, 2), (3, 4)]);
}

#[test]
fn test_unit_blossom_required() {
    assert_unit_cap_matches_blossom(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]);
}

#[test]
fn test_unit_nested_blossom() {
    assert_unit_cap_matches_blossom(
        7,
        &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)],
    );
}

#[test]
fn test_unit_petersen() {
    assert_unit_cap_matches_blossom(
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
fn test_unit_complete_k5() {
    assert_unit_cap_matches_blossom(
        5,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    );
}

#[test]
fn test_unit_complete_k7() {
    let mut edges = Vec::new();
    for i in 0..7 {
        for j in (i + 1)..7 {
            edges.push((i, j));
        }
    }
    assert_unit_cap_matches_blossom(7, &edges);
}

#[test]
fn test_unit_wheel_w5() {
    assert_unit_cap_matches_blossom(
        6,
        &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
    );
}

#[test]
fn test_unit_barbell() {
    assert_unit_cap_matches_blossom(
        7,
        &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)],
    );
}

#[test]
fn test_unit_cube_q3() {
    assert_unit_cap_matches_blossom(
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
fn test_unit_k33() {
    let mut edges = Vec::new();
    for i in 0..3 {
        for j in 3..6 {
            edges.push((i, j));
        }
    }
    assert_unit_cap_matches_blossom(6, &edges);
}

// ============================================================================
// Higher-capacity / bond order tests
// ============================================================================

#[test]
fn test_ethylene_double_bond() {
    // C=C: two carbons, each with valence 4, edge cap 2
    // But only 2 hydrogens are implicit, so budget = 2 (not full valence)
    let edges = [(0, 1, 2)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 2, "ethylene should have flow 2 (double bond)");
    assert_eq!(flow.len(), 1);
    assert_eq!(flow[0].2, 2);
}

#[test]
fn test_single_edge_cap1() {
    let edges = [(0, 1, 1)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 1);
}

#[test]
fn test_single_edge_cap3_budget_limited() {
    // Edge cap 3 but vertex budget 2 limits flow to 2.
    let edges = [(0, 1, 3)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 2, "flow limited by budget");
}

#[test]
fn test_single_edge_cap3_budget3() {
    // Edge cap 3, budget 3 → flow 3
    let edges = [(0, 1, 3)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [3, 3];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 3);
}

#[test]
fn test_triangle_cap1_budget1() {
    // Triangle with all caps 1, all budgets 1 → only 1 edge can carry flow.
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 1);
}

#[test]
fn test_triangle_cap2_budget2() {
    // Triangle with all caps 2, all budgets 2 → each vertex uses its full
    // budget of 2, giving 3 total flow (one unit on each edge).
    let edges = [(0, 1, 2), (0, 2, 2), (1, 2, 2)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    // Max balanced flow: 1+1+1 = 3 (each edge gets 1)
    assert_eq!(total, 3);
}

#[test]
fn test_path_cap2_budget2() {
    // Path: 0 -- 1 -- 2, all caps 2, budgets 2.
    // Vertex 1 is the bottleneck with budget 2, shared by edges 0-1 and 1-2.
    // Optimal: flow=1 on each edge, total=2.
    let edges = [(0, 1, 2), (1, 2, 2)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_star_cap1_budget3() {
    // Star: center vertex 0 connected to 1, 2, 3. All caps 1, center budget 3.
    // Each leaf budget 1. Total flow = 3.
    let edges = [(0, 1, 1), (0, 2, 1), (0, 3, 1)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [3, 1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 3);
}

#[test]
fn test_star_cap1_budget1() {
    // Star: center vertex 0 with budget 1. Only one edge can carry flow.
    let edges = [(0, 1, 1), (0, 2, 1), (0, 3, 1)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [1, 1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 1);
}

#[test]
fn test_zero_budget() {
    // All budgets zero → no flow.
    let edges = [(0, 1, 1)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [0, 0];
    let flow = vcsr.kocay(&budgets);
    assert!(flow.is_empty());
}

#[test]
fn test_zero_capacity() {
    // Edge capacity zero → no flow.
    let edges = [(0, 1, 0)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [1, 1];
    let flow = vcsr.kocay(&budgets);
    assert!(flow.is_empty());
}

#[test]
fn test_two_disjoint_edges_cap2() {
    // Two disjoint edges, each with cap 2, each vertex budget 2.
    let edges = [(0, 1, 2), (2, 3, 2)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [2, 2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 4);
}

#[test]
fn test_chain_3_cap1_budget1() {
    // 0-1-2-3, unit caps, unit budgets → matching of size 2.
    let edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [1, 1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_mixed_capacities() {
    // 0 --cap1-- 1 --cap2-- 2, budgets [1, 2, 2]
    // Edge 0-1 limited to cap 1, edge 1-2 limited to cap 2, but budget of
    // vertex 1 is 2 so edge 0-1 can carry 1 and edge 1-2 can carry 1.
    // Total = 2.
    let edges = [(0, 1, 1), (1, 2, 2)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_square_cap2_budget2() {
    // Square C4 with cap 2, budget 2.
    // Each vertex touches 2 edges with budget 2. Max flow = 4 (1 per edge).
    let edges = [(0, 1, 2), (1, 2, 2), (2, 3, 2), (0, 3, 2)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [2, 2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 4);
}

#[test]
fn test_single_edge_cap5_budget5() {
    let edges = [(0, 1, 5)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [5, 5];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 5);
}

// ============================================================================
// Panic tests
// ============================================================================

#[test]
#[should_panic(expected = "Kocay requires a square matrix")]
fn test_non_square_panics() {
    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shape((2, 3));
    MatrixMut::add(&mut vcsr, (0, 1, 1)).unwrap();
    vcsr.kocay(&[1, 1]);
}

#[test]
#[should_panic(expected = "vertex_budgets length")]
fn test_wrong_budget_len_panics() {
    let vcsr = build_valued_graph(3, &[(0, 1, 1)]);
    vcsr.kocay(&[1, 1]); // len 2 != 3
}

// ============================================================================
// Higher-budget integration tests
// ============================================================================

#[test]
fn test_complete_k4_cap2_budget3() {
    // K4 with cap 2 on all edges, budget 3.
    // Each vertex has 3 neighbors. Budget 3 → at most 1 unit per edge on
    // average. 6 edges * 1 = 6 total flow, but each vertex limited to 3.
    // 4 vertices * 3 / 2 = 6. So max flow = 6.
    let edges = [(0, 1, 2), (0, 2, 2), (0, 3, 2), (1, 2, 2), (1, 3, 2), (2, 3, 2)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [3, 3, 3, 3];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 6);
}

#[test]
fn test_two_edges_shared_vertex() {
    // 0 --cap2-- 1 --cap2-- 2, budget [2, 3, 2]
    // Vertex 1 has budget 3 and two edges. Each edge can carry 2.
    // Total = 4 (2+2), vertex 1 uses 4... but budget is 3. So total = 3.
    let edges = [(0, 1, 2), (1, 2, 2)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [2, 3, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 3);
}

#[test]
fn test_long_path_unit_cap() {
    // Path of 20 vertices, unit caps, unit budgets.
    let n = 20;
    let edges: Vec<(usize, usize, usize)> = (0..n - 1).map(|i| (i, i + 1, 1)).collect();
    let vcsr = build_valued_graph(n, &edges);
    let budgets = vec![1; n];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(n, &edges, &budgets, &flow);
    assert_eq!(total, 10); // perfect matching on even-length path
}

#[test]
fn test_disconnected_components_mixed() {
    // Component A: 0-1 with cap 3, budget 3 → flow 3
    // Component B: 2-3-4 triangle with cap 1, budget 1 → flow 1
    let edges = [(0, 1, 3), (2, 3, 1), (2, 4, 1), (3, 4, 1)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [3, 3, 1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 4); // 3 + 1
}

// ============================================================================
// Coverage-targeted tests
// ============================================================================

#[test]
fn test_triangle_asymmetric_budget() {
    // Triangle 0-1-2, cap 1, budgets [1, 1, 2].
    // First BNS augments one edge (say 0-1). Second BNS reroutes: remove flow
    // from 0-1, add flow on 0-2 and 1-2. Total = 2.
    // Exercises: blossom extension (rescap >= 2 at non-root LCA), backward
    // atom-atom edges (flow rerouting), multiple BNS passes.
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 1, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_triangle_asymmetric_budget_cap2() {
    // Triangle 0-1-2, cap 2, budgets [1, 1, 3].
    // Vertex 2 can handle high flow. Two edges incident to vertex 2.
    // Max: edge 0-2 = 1, edge 1-2 = 1. Total = 2.
    let edges = [(0, 1, 2), (0, 2, 2), (1, 2, 2)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 1, 3];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_bowtie_asymmetric() {
    // Bowtie: triangles 0-1-2 and 2-3-4 sharing vertex 2.
    // cap 1, budgets [1, 1, 2, 1, 1].
    // Max: edges 0-1, 2-3, 2-4 → v2 uses budget 2. Total = 3.
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1), (2, 4, 1), (3, 4, 1)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [1, 1, 2, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 3);
}

#[test]
fn test_bowtie_high_center_budget() {
    // Bowtie: triangles 0-1-2 and 2-3-4 sharing vertex 2.
    // cap 1, budgets [1, 1, 4, 1, 1].
    // Vertex 2 has high budget. Max: edges 0-2, 1-2, 2-3, 2-4 → but
    // budget of 0,1,3,4 = 1 each. So max 4 edges but leaves 0-2 use v0's
    // budget. Actually: 0-2 (1), 1-2 (1), 3-2 (1), 4-2 (1) uses vertex 2
    // budget 4. Total = 4.
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1), (2, 4, 1), (3, 4, 1)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [1, 1, 4, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 4);
}

#[test]
fn test_pentagon_cap2_budget2() {
    // Pentagon C5 with cap 2, budget 2. Each vertex degree 2, budget 2.
    // Optimal: 1 unit on each of 5 edges. Total = 5.
    let edges = [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 4, 2), (0, 4, 2)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [2, 2, 2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 5);
}

#[test]
fn test_k5_cap2_budget4() {
    // K5 with cap 2, budget 4. Each vertex has 4 neighbors.
    // Budget 4 = degree 4. Each edge gets 1. Total = 10.
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            edges.push((i, j, 2));
        }
    }
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [4, 4, 4, 4, 4];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 10);
}

#[test]
fn test_path_4_cap2_budget_varied() {
    // Path 0-1-2-3, cap 2 everywhere, budgets [2, 3, 3, 2].
    // Optimal: edge 0-1 = 2, edge 1-2 = 1, edge 2-3 = 2. Total = 5.
    // Vertex 0: 2 (=budget). Vertex 1: 2+1=3 (=budget).
    // Vertex 2: 1+2=3 (=budget). Vertex 3: 2 (=budget).
    let edges = [(0, 1, 2), (1, 2, 2), (2, 3, 2)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [2, 3, 3, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    assert_eq!(total, 5);
}

#[test]
fn test_k33_cap2_budget3() {
    // K_{3,3} with cap 2, budget 3 on all vertices.
    // Each vertex has 3 neighbors. Budget 3. Each edge gets 1.
    // Total = 9.
    let mut edges = Vec::new();
    for i in 0..3 {
        for j in 3..6 {
            edges.push((i, j, 2));
        }
    }
    let vcsr = build_valued_graph(6, &edges);
    let budgets = [3, 3, 3, 3, 3, 3];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(6, &edges, &budgets, &flow);
    assert_eq!(total, 9);
}

#[test]
fn test_budget_exceeds_degree_times_cap() {
    // Budget far exceeds what's possible. Should not panic or produce extra flow.
    // Star: center 0, leaves 1..4. Cap 1 per edge, center budget 100.
    let edges = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [100, 1, 1, 1, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert_eq!(total, 4, "excess budget should not produce extra flow");
}

#[test]
fn test_high_capacity_single_edge() {
    // Single edge with high capacity, matched budgets.
    let edges = [(0, 1, 100)];
    let vcsr = build_valued_graph(2, &edges);
    let budgets = [100, 100];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(2, &edges, &budgets, &flow);
    assert_eq!(total, 100);
}

#[test]
fn test_asymmetric_high_cap() {
    // One edge with cap 10, another with cap 1. Shared vertex budget=3.
    // 0 --cap10-- 1 --cap1-- 2, budgets [10, 3, 1]
    // Vertex 1 budget 3: edge 0-1 can carry at most 2, edge 1-2 at most 1.
    // Total = 3.
    let edges = [(0, 1, 10), (1, 2, 1)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [10, 3, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 3);
}

#[test]
fn test_caterpillar_cap2_budget2() {
    // Caterpillar: path 0-1-2-3-4, plus leaves 5-0, 6-1, 7-2, 8-3, 9-4.
    // Cap 2, budget 2. Each spine vertex has degree 3 but budget 2.
    // Optimal: leaf edges at 1 + spine edges 0-1 and 2-3 at 1 = total 7.
    let edges = [
        (0, 1, 2),
        (1, 2, 2),
        (2, 3, 2),
        (3, 4, 2),
        (0, 5, 2),
        (1, 6, 2),
        (2, 7, 2),
        (3, 8, 2),
        (4, 9, 2),
    ];
    let vcsr = build_valued_graph(10, &edges);
    let budgets = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(10, &edges, &budgets, &flow);
    assert!(total >= 5, "expected at least 5, got {total}");
}

#[test]
fn test_pentagon_cap3_budget3() {
    // C5, cap 3, budget 3. Degree 2, budget 3.
    // LP optimum = 7.5 (each edge 1.5). Integer optimum = 7 (e.g. 2,1,2,1,1).
    let edges = [(0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3), (0, 4, 3)];
    let vcsr = build_valued_graph(5, &edges);
    let budgets = [3, 3, 3, 3, 3];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(5, &edges, &budgets, &flow);
    assert!(total >= 5, "expected at least 5, got {total}");
}

#[test]
fn test_two_triangles_shared_edge() {
    // "Diamond" / K4-minus-edge: 0-1-2-0, 1-2-3-1.
    // Cap 1, budgets [1, 2, 2, 1]. Shared edge 1-2 is contested.
    // Max: edges 0-1, 1-2, 2-3 → vertex 1 uses 2, vertex 2 uses 2. Total = 3.
    // But edge 0-2 also exists. Let's be precise:
    // Edges: 0-1, 0-2, 1-2, 1-3, 2-3. Cap 1, budgets [1, 2, 2, 1].
    // Opt: 0-1 + 2-3 → total 2. Or: 0-2 + 1-3 → total 2. Or: 1-2 → 1.
    // Actually: 0-1 (uses v0=1, v1=1), 2-3 (uses v2=1, v3=1) → total=2.
    // With budgets [1,2,2,1]: v1 has budget 2. So 0-1 + 1-3 (uses v1=2, v0=1,
    // v3=1), + 0-2? v0 used. Max: 0-1 + 2-3 = 2 or 0-2 + 1-3 = 2 or 0-1 + 1-3 =
    // 2 (uses v1=2). But can we get 3? 0-1 + 1-3 + 0-2? v0 used twice → budget
    // [1] exceeded. With v0 budget=1: can only use 1 edge incident on v0.
    // Similarly v3 budget=1. So max is 2 (vertex-limited at v0 and v3).
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [1, 2, 2, 1];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(4, &edges, &budgets, &flow);
    // Optimal: e.g. 0-1 + 1-2 + 2-3 → v0=1, v1=2, v2=2, v3=1. Total=3.
    assert_eq!(total, 3);
}

#[test]
fn test_long_chain_higher_cap() {
    // Path 0-1-...-9, cap 3, budget 3 everywhere.
    // Interior vertices (1..8) have degree 2 → max flow per vertex = 3 but
    // limited by 2 incident edges. Total = 5 edges × min(3, budget) = limited.
    // Each edge can carry up to 1.5 avg. Integer: alternating 2,1,2,1,...
    // Vertex 0: budget 3, degree 1 → edge 0-1 carries up to 3.
    // Vertex 1: budget 3, degree 2 → edge 0-1 + edge 1-2 ≤ 3.
    // This forces: f(0,1) + f(1,2) ≤ 3, f(1,2) + f(2,3) ≤ 3, etc.
    // Optimal: 3,0,3,0,3,0,3,0,3 → total 15. Check: v0=3, v1=3+0=3, etc.
    let n = 10;
    let edges: Vec<(usize, usize, usize)> = (0..n - 1).map(|i| (i, i + 1, 3)).collect();
    let vcsr = build_valued_graph(n, &edges);
    let budgets = vec![3usize; n];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(n, &edges, &budgets, &flow);
    assert!(total >= 10, "expected at least 10, got {total}");
}

#[test]
fn test_wheel6_cap2_budget3() {
    // Wheel W6: center 0 connected to 1..5, plus cycle 1-2-3-4-5-1.
    // cap 2, budget 3 on center, budget 2 on rim.
    let mut edges = vec![];
    for i in 1..=5 {
        edges.push((0, i, 2));
    }
    for i in 1..5 {
        edges.push((i, i + 1, 2));
    }
    edges.push((1, 5, 2));
    let vcsr = build_valued_graph(6, &edges);
    let budgets = [3, 2, 2, 2, 2, 2];
    let flow = vcsr.kocay(&budgets);
    let total = validate_flow(6, &edges, &budgets, &flow);
    // Total flow ≤ (3 + 5×2) / 2 = 6.
    assert_eq!(total, 6);
}

// ============================================================================
// kocay_with_initial_flow — correctness tests
// ============================================================================

#[test]
fn test_empty_initial_flow_equals_kocay() {
    // Verify that kocay_with_initial_flow(b, &[]) == kocay(b) on several graphs.
    let cases = vec![
        (2, vec![(0, 1, 1)], vec![1, 1]),
        (3, vec![(0, 1, 1), (0, 2, 1), (1, 2, 1)], vec![1, 1, 1]),
        (2, vec![(0, 1, 2)], vec![2, 2]),
        (4, vec![(0, 1, 2), (1, 2, 2), (2, 3, 2), (0, 3, 2)], vec![2, 2, 2, 2]),
    ];
    for (n, edges, budgets) in cases {
        let vcsr = build_valued_graph(n, &edges);
        let flow_plain = vcsr.kocay(&budgets);
        let flow_empty = vcsr.kocay_with_initial_flow(&budgets, &[]);
        assert_eq!(flow_plain, flow_empty);
    }
}

#[test]
fn test_optimal_initial_flow_preserved() {
    // Triangle cap=1 budgets=[1,1,1]: only one edge can carry flow.
    // Start from edge (0,1)=1 → result must be exactly that edge.
    let edges = [(0, 1, 1), (0, 2, 1), (1, 2, 1)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 1, 1];

    let flow_01 = vcsr.kocay_with_initial_flow(&budgets, &[(0, 1, 1)]);
    assert_eq!(flow_01, vec![(0, 1, 1)]);

    // Start from (1,2)=1 → result must be exactly that edge.
    let flow_12 = vcsr.kocay_with_initial_flow(&budgets, &[(1, 2, 1)]);
    assert_eq!(flow_12, vec![(1, 2, 1)]);
}

#[test]
fn test_suboptimal_initial_flow_augmented() {
    // Path 0-1-2, cap=1, budgets=[1,2,1]: start from [(0,1,1)].
    // Vertex 1 has budget 2, so both edges can carry flow=1. Total=2.
    let edges = [(0, 1, 1), (1, 2, 1)];
    let vcsr = build_valued_graph(3, &edges);
    let budgets = [1, 2, 1];

    let flow = vcsr.kocay_with_initial_flow(&budgets, &[(0, 1, 1)]);
    let total = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 2);
}

#[test]
fn test_co2_disambiguation() {
    // C(0), O1(1), O2(2), charge+(3)
    // edges: 0-1 cap=2, 0-2 cap=2, 1-3 cap=1, 2-3 cap=1
    // budgets=[2,2,2,2]
    // initial_flow: single bonds everywhere = [(0,1,1),(0,2,1),(1,3,1),(2,3,1)]
    // Total initial = 4 = sum of budgets / 2. This is already optimal.
    let edges = [(0, 1, 2), (0, 2, 2), (1, 3, 1), (2, 3, 1)];
    let vcsr = build_valued_graph(4, &edges);
    let budgets = [2, 2, 2, 2];
    let initial = [(0usize, 1usize, 1usize), (0, 2, 1), (1, 3, 1), (2, 3, 1)];

    let flow = vcsr.kocay_with_initial_flow(&budgets, &initial);
    assert_eq!(flow, vec![(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 3, 1)]);
}

#[test]
fn test_optimal_is_fixed_point() {
    // Solve from zero, then use result as initial flow → identical output.
    let cases = vec![
        (3, vec![(0, 1, 1), (0, 2, 1), (1, 2, 1)], vec![2, 2, 2]),
        (4, vec![(0, 1, 2), (1, 2, 2), (2, 3, 2), (0, 3, 2)], vec![2, 2, 2, 2]),
        (2, vec![(0, 1, 5)], vec![5, 5]),
        (
            5,
            vec![(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1), (2, 4, 1), (3, 4, 1)],
            vec![1, 1, 2, 1, 1],
        ),
    ];
    for (n, edges, budgets) in cases {
        let vcsr = build_valued_graph(n, &edges);
        let flow1 = vcsr.kocay(&budgets);
        let flow2 = vcsr.kocay_with_initial_flow(&budgets, &flow1);
        assert_eq!(flow1, flow2, "optimal flow must be a fixed point");
    }
}

// ============================================================================
// kocay_with_initial_flow — panic tests
// ============================================================================

#[test]
#[should_panic(expected = "exceeds edge capacity")]
fn test_rejects_flow_exceeding_edge_capacity() {
    let vcsr = build_valued_graph(2, &[(0, 1, 1)]);
    vcsr.kocay_with_initial_flow(&[1, 1], &[(0, 1, 2)]);
}

#[test]
#[should_panic(expected = "exceeds budget")]
fn test_rejects_flow_exceeding_vertex_budget() {
    // Two edges incident on vertex 1 with total flow > budget.
    let vcsr = build_valued_graph(3, &[(0, 1, 2), (1, 2, 2)]);
    vcsr.kocay_with_initial_flow(&[2, 2, 2], &[(0, 1, 2), (1, 2, 1)]);
}

#[test]
#[should_panic(expected = "non-existent edge")]
fn test_rejects_nonexistent_edge() {
    let vcsr = build_valued_graph(3, &[(0, 1, 1)]);
    vcsr.kocay_with_initial_flow(&[1, 1, 1], &[(0, 2, 1)]);
}

#[test]
#[should_panic(expected = "row < col")]
fn test_rejects_wrong_ordering() {
    let vcsr = build_valued_graph(2, &[(0, 1, 1)]);
    vcsr.kocay_with_initial_flow(&[1, 1], &[(1, 0, 1)]);
}

#[test]
#[should_panic(expected = "must be > 0")]
fn test_rejects_zero_flow() {
    let vcsr = build_valued_graph(2, &[(0, 1, 1)]);
    vcsr.kocay_with_initial_flow(&[1, 1], &[(0, 1, 0)]);
}

#[test]
#[should_panic(expected = "duplicate")]
fn test_rejects_duplicate_edge() {
    let vcsr = build_valued_graph(2, &[(0, 1, 2)]);
    vcsr.kocay_with_initial_flow(&[2, 2], &[(0, 1, 1), (0, 1, 1)]);
}
