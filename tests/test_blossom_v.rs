//! Tests for the Blossom V min-cost perfect matching algorithm.
#![cfg(feature = "std")]
#![allow(clippy::pedantic)]

#[path = "support/blossom_v_regression_cases.rs"]
mod blossom_v_regression_cases;

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
};

use self::blossom_v_regression_cases::{
    HONGGFUZZ_SIGABRT_CASE_1_EDGES, HONGGFUZZ_SIGABRT_CASE_1_ORDER, HONGGFUZZ_SIGABRT_CASE_2_EDGES,
    HONGGFUZZ_SIGABRT_CASE_2_ORDER, HONGGFUZZ_SIGABRT_CASE_3_EDGES, HONGGFUZZ_SIGABRT_CASE_3_ORDER,
    HONGGFUZZ_SIGABRT_CASE_4_EDGES, HONGGFUZZ_SIGABRT_CASE_4_ORDER,
};

type SupportGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type Vcsr = ValuedCSR2D<usize, usize, usize, i32>;
const HONGGFUZZ_SIGABRT_CASE_4_COST: i64 = -186717;

/// Build a symmetric valued matrix from weighted edges.
fn build_valued_graph(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
    let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, w));
        sorted_edges.push((hi, lo, w));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted_edges.len());
    for (r, c, v) in sorted_edges {
        MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
    }
    vcsr
}

/// Build a symmetric valued matrix while preserving the input edge order.
fn build_valued_graph_preserve_input_order(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), edges.len() * 2);
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        MatrixMut::add(&mut vcsr, (i, j, w)).unwrap();
        MatrixMut::add(&mut vcsr, (j, i, w)).unwrap();
    }
    vcsr
}

/// Build an unweighted symmetric graph from an edge list.
fn build_support_graph(n: usize, edges: &[(usize, usize, i32)]) -> SupportGraph {
    let mut sorted_edges: Vec<(usize, usize)> = Vec::new();
    for &(i, j, _) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup();
    UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(n)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap()
}

/// Validate a perfect matching result.
fn validate_matching(n: usize, edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) {
    assert_eq!(matching.len(), n / 2);
    let mut used = vec![false; n];
    for &(u, v) in matching {
        assert!(u < v, "Pairs must be ordered u < v, got ({u}, {v})");
        assert!(!used[u], "Vertex {u} used twice");
        assert!(!used[v], "Vertex {v} used twice");
        assert!(
            edges.iter().any(|&(a, b, _)| (a == u && b == v) || (a == v && b == u)),
            "Matched edge ({u}, {v}) does not exist in graph"
        );
        used[u] = true;
        used[v] = true;
    }
}

/// Compute the total cost of a matching.
fn matching_cost(edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) -> i32 {
    matching
        .iter()
        .map(|&(u, v)| {
            edges
                .iter()
                .find(|&&(a, b, _)| (a == u && b == v) || (a == v && b == u))
                .map(|&(_, _, w)| w)
                .expect("matched edge not found")
        })
        .sum()
}

fn brute_force_oracle_cost(order: usize, edges: &[(usize, usize, i32)]) -> Option<i64> {
    if order == 0 {
        return Some(0);
    }

    let mut weights = vec![vec![None; order]; order];
    for &(u, v, weight) in edges {
        weights[u][v] = Some(i64::from(weight));
        weights[v][u] = Some(i64::from(weight));
    }

    fn solve(
        mask: usize,
        weights: &[Vec<Option<i64>>],
        memo: &mut [Option<Option<i64>>],
    ) -> Option<i64> {
        if mask == 0 {
            return Some(0);
        }
        if let Some(cached) = memo[mask] {
            return cached;
        }

        let i = mask.trailing_zeros() as usize;
        let rest = mask & !(1usize << i);
        let mut best = None;
        let mut candidates = rest;

        while candidates != 0 {
            let j_bit = candidates & candidates.wrapping_neg();
            let j = j_bit.trailing_zeros() as usize;
            candidates &= candidates - 1;

            let Some(weight) = weights[i][j] else {
                continue;
            };
            let submask = rest & !(1usize << j);
            if let Some(subcost) = solve(submask, weights, memo) {
                let total = subcost + weight;
                best = Some(best.map_or(total, |current: i64| current.min(total)));
            }
        }

        memo[mask] = Some(best);
        best
    }

    let mut memo = vec![None; 1usize << order];
    solve((1usize << order) - 1, &weights, &mut memo)
}

fn assert_blossom_v_matches_oracle_cost(
    label: &str,
    order: usize,
    edges: &[(usize, usize, i32)],
    oracle: Option<i64>,
) {
    let g = build_valued_graph(order, edges);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Ok(matching)) => {
            let optimum =
                oracle.unwrap_or_else(|| panic!("{label}: oracle found no perfect matching"));
            validate_matching(order, edges, &matching);
            let actual = i64::from(matching_cost(edges, &matching));
            assert_eq!(actual, optimum, "{label}: Blossom V returned non-optimal matching cost");
        }
        Ok(Err(BlossomVError::NoPerfectMatching)) => {
            assert!(
                oracle.is_none(),
                "{label}: Blossom V reported no perfect matching incorrectly"
            );
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("{label}: Blossom V panicked: {msg}");
        }
    }
}

// ===== Hand-crafted tests =====

#[test]
fn test_blossom_v_single_edge() {
    let edges = [(0, 1, 5)];
    let g = build_valued_graph(2, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(2, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), 5);
}

#[test]
fn test_blossom_v_path_4_vertices() {
    let edges = [(0, 1, 1), (1, 2, 10), (2, 3, 1)];
    let g = build_valued_graph(4, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(4, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), 2);
}

#[test]
fn test_blossom_v_no_perfect_matching() {
    let edges = [(0, 1, 5)];
    let g = build_valued_graph(4, &edges);
    assert!(g.blossom_v().is_err());
}

#[test]
fn test_blossom_v_even_connected_graph_without_perfect_matching_returns_error() {
    let edges = [(0, 1, 1), (0, 2, 1), (0, 3, 1)];
    let g = build_valued_graph(4, &edges);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => {
            panic!("expected NoPerfectMatching for even connected infeasible graph, got {other:?}")
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on even connected infeasible graph: {msg}");
        }
    }
}

#[test]
fn test_blossom_v_two_isolated_vertices_returns_no_perfect_matching_without_panicking() {
    let g = build_valued_graph(2, &[]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => panic!("expected NoPerfectMatching for two isolated vertices, got {other:?}"),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on two isolated vertices: {msg}");
        }
    }
}

#[test]
fn test_blossom_v_two_isolated_vertices_returns_no_perfect_matching_via_solver_path() {
    let g = build_valued_graph(2, &[]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => panic!("expected NoPerfectMatching for two isolated vertices, got {other:?}"),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on two isolated vertices: {msg}");
        }
    }
}

#[test]
fn test_blossom_v_sparse_graph_with_isolates_returns_no_perfect_matching_via_solver_path() {
    let edges = [(0, 1, 5)];
    let g = build_valued_graph(4, &edges);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => {
            panic!("expected NoPerfectMatching for sparse graph with isolates, got {other:?}")
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on sparse graph with isolates: {msg}");
        }
    }
}

#[test]
fn test_blossom_v_disconnected_odd_components_returns_no_perfect_matching_via_solver_path() {
    let edges = [(0, 1, 1), (1, 4, 2), (0, 4, 3), (2, 3, 4), (3, 5, 5), (2, 5, 6)];
    let g = build_valued_graph(6, &edges);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => {
            panic!("expected NoPerfectMatching for disconnected odd components, got {other:?}")
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on disconnected odd components: {msg}");
        }
    }
}

#[test]
fn test_blossom_v_even_connected_graph_without_perfect_matching_returns_error_via_solver_path() {
    let edges = [(0, 1, 1), (0, 2, 1), (0, 3, 1)];
    let g = build_valued_graph(4, &edges);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.blossom_v()));
    match result {
        Ok(Err(BlossomVError::NoPerfectMatching)) => {}
        Ok(other) => {
            panic!("expected NoPerfectMatching for even connected infeasible graph, got {other:?}")
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("blossom_v() must not panic on even connected infeasible graph: {msg}");
        }
    }
}

#[test]
#[should_panic(expected = "even number of vertices")]
fn test_blossom_v_odd_vertices_panics() {
    let g = build_valued_graph(3, &[(0, 1, 1), (1, 2, 1)]);
    let _ = g.blossom_v();
}

#[test]
#[should_panic(expected = "square matrix")]
fn test_blossom_v_non_square_matrix_panics() {
    let mut g: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((2, 3), 2);
    MatrixMut::add(&mut g, (0, 1, 1)).unwrap();
    MatrixMut::add(&mut g, (1, 2, 1)).unwrap();
    let _ = g.blossom_v();
}

#[test]
fn test_blossom_v_empty_graph() {
    let g: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((0, 0), 0);
    assert!(g.blossom_v().expect("trivial").is_empty());
}

#[test]
fn test_blossom_v_negative_weights() {
    let edges = [(0, 1, -10), (2, 3, -20), (0, 2, 100), (1, 3, 100)];
    let g = build_valued_graph(4, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(4, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -30);
}

#[test]
fn test_blossom_v_k4_complete() {
    let edges = [(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)];
    let g = build_valued_graph(4, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(4, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), 7);
}

#[test]
fn test_blossom_v_graph1_from_paper() {
    let edges = [
        (0, 1, 3),
        (0, 3, 10),
        (0, 4, 7),
        (1, 2, -1),
        (1, 4, 5),
        (1, 5, 4),
        (2, 5, -7),
        (3, 4, 0),
        (4, 5, 4),
    ];
    let g = build_valued_graph(6, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(6, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -4);
}

#[test]
fn test_blossom_v_regression_case_909() {
    // First failing case from the small ground-truth corpus.
    let edges = [(0, 2, 16), (1, 3, -65), (0, 1, -64), (2, 3, 16)];
    let g = build_valued_graph(4, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(4, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -49);
    assert_eq!(matching, vec![(0, 2), (1, 3)]);
}

#[test]
fn test_blossom_v_generated_case_214() {
    let edges = [
        (0, 1, 90),
        (0, 2, -66),
        (0, 3, -13),
        (0, 4, -83),
        (0, 5, 73),
        (0, 7, -70),
        (0, 9, -67),
        (0, 11, 39),
        (0, 12, 40),
        (0, 13, -57),
        (0, 14, -32),
        (0, 18, -54),
        (0, 19, 73),
        (0, 20, -60),
        (1, 2, -96),
        (1, 3, -17),
        (1, 4, 12),
        (1, 5, -20),
        (1, 6, 42),
        (1, 8, -5),
        (1, 11, 31),
        (1, 14, 0),
        (1, 16, -90),
        (1, 21, -90),
        (2, 3, -88),
        (2, 4, -38),
        (2, 5, 80),
        (2, 7, 7),
        (2, 9, -27),
        (2, 10, -30),
        (2, 12, 66),
        (2, 17, -46),
        (2, 18, -66),
        (2, 19, -61),
        (3, 4, -11),
        (3, 5, 90),
        (3, 6, 77),
        (3, 8, 82),
        (3, 10, 96),
        (3, 11, -90),
        (3, 12, -91),
        (3, 18, 30),
        (4, 6, -93),
        (4, 7, -5),
        (4, 8, -38),
        (4, 10, 2),
        (4, 14, 57),
        (4, 18, -11),
        (5, 6, 98),
        (5, 7, 91),
        (5, 15, -63),
        (5, 21, -83),
        (6, 8, 89),
        (6, 12, 58),
        (6, 13, 98),
        (7, 9, -27),
        (7, 10, -64),
        (7, 15, 59),
        (7, 16, -66),
        (7, 19, -5),
        (8, 9, -28),
        (8, 11, -29),
        (8, 13, -70),
        (8, 15, 38),
        (9, 17, -91),
        (10, 17, -2),
        (10, 20, 28),
        (10, 21, -11),
        (12, 13, -76),
        (12, 16, -8),
        (12, 19, 68),
        (13, 14, -57),
        (13, 20, -42),
        (13, 21, -16),
        (14, 15, 78),
        (14, 16, -2),
        (16, 17, -47),
        (18, 20, -35),
    ];
    let g = build_valued_graph(22, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(22, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -697);
    assert_eq!(
        matching,
        vec![
            (0, 14),
            (1, 21),
            (2, 19),
            (3, 11),
            (4, 6),
            (5, 15),
            (7, 10),
            (8, 13),
            (9, 17),
            (12, 16),
            (18, 20),
        ]
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_1() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 1",
        HONGGFUZZ_SIGABRT_CASE_1_ORDER,
        HONGGFUZZ_SIGABRT_CASE_1_EDGES,
        brute_force_oracle_cost(HONGGFUZZ_SIGABRT_CASE_1_ORDER, HONGGFUZZ_SIGABRT_CASE_1_EDGES),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_2() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 2",
        HONGGFUZZ_SIGABRT_CASE_2_ORDER,
        HONGGFUZZ_SIGABRT_CASE_2_EDGES,
        brute_force_oracle_cost(HONGGFUZZ_SIGABRT_CASE_2_ORDER, HONGGFUZZ_SIGABRT_CASE_2_EDGES),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_3() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 3",
        HONGGFUZZ_SIGABRT_CASE_3_ORDER,
        HONGGFUZZ_SIGABRT_CASE_3_EDGES,
        brute_force_oracle_cost(HONGGFUZZ_SIGABRT_CASE_3_ORDER, HONGGFUZZ_SIGABRT_CASE_3_EDGES),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_4() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 4",
        HONGGFUZZ_SIGABRT_CASE_4_ORDER,
        HONGGFUZZ_SIGABRT_CASE_4_EDGES,
        Some(HONGGFUZZ_SIGABRT_CASE_4_COST),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_8() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 8",
        HONGGFUZZ_SIGABRT_CASE_2_ORDER,
        HONGGFUZZ_SIGABRT_CASE_2_EDGES,
        brute_force_oracle_cost(HONGGFUZZ_SIGABRT_CASE_2_ORDER, HONGGFUZZ_SIGABRT_CASE_2_EDGES),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_15() {
    assert_blossom_v_matches_oracle_cost(
        "honggfuzz sigabrt case 15",
        HONGGFUZZ_SIGABRT_CASE_4_ORDER,
        HONGGFUZZ_SIGABRT_CASE_4_EDGES,
        Some(HONGGFUZZ_SIGABRT_CASE_4_COST),
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_9_support_feasible() {
    let edges = [
        (0, 1, 0),
        (0, 3, -21251),
        (0, 6, -2023),
        (0, 9, 14768),
        (0, 12, 12819),
        (0, 14, 0),
        (0, 15, 0),
        (0, 16, -27420),
        (0, 17, -26215),
        (1, 3, -1),
        (1, 5, 32512),
        (1, 9, -30271),
        (1, 10, 5020),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 4, 100),
        (2, 14, 76),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 6, -1),
        (3, 12, 3072),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 5, 2752),
        (4, 8, 26125),
        (4, 17, -18671),
        (5, 8, 12331),
        (5, 14, -10251),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 9, 13364),
        (8, 9, -2846),
        (8, 10, -1387),
        (8, 12, -24415),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 14, -26215),
        (9, 16, -18577),
        (10, 11, -12279),
        (10, 13, -8642),
        (11, 13, -7374),
        (11, 14, 32018),
        (12, 14, 14393),
        (12, 15, -24),
        (12, 17, 50),
        (14, 17, 1128),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(
        support.blossom().len(),
        9,
        "support graph should admit a perfect matching for this fuzz regression"
    );

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect(
        "Blossom V should not report NoPerfectMatching on this support-feasible honggfuzz regression",
    );
    validate_matching(18, &edges, &matching);
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_10_support_feasible() {
    let edges = [
        (0, 1, 0),
        (0, 3, -21251),
        (0, 6, -2023),
        (0, 8, 21258),
        (0, 9, 14768),
        (0, 11, 0),
        (0, 12, 12819),
        (0, 14, -23283),
        (0, 15, 0),
        (0, 16, -27420),
        (0, 17, -26215),
        (1, 3, -1),
        (1, 5, 32512),
        (1, 8, 1),
        (1, 9, -30271),
        (1, 10, 5020),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 4, 100),
        (2, 9, -2846),
        (2, 14, 76),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 6, -1),
        (3, 12, 3072),
        (3, 13, 511),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 5, 2752),
        (4, 8, 26125),
        (4, 17, -18671),
        (5, 8, 12331),
        (5, 14, -10251),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, 0),
        (7, 9, 13364),
        (8, 10, -1387),
        (8, 12, -24415),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 14, -26215),
        (9, 16, -18577),
        (10, 11, -12279),
        (10, 13, -8642),
        (11, 13, -7374),
        (11, 14, 26851),
        (12, 14, 14393),
        (12, 15, -24),
        (12, 17, 50),
        (16, 17, 1128),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(
        support.blossom().len(),
        9,
        "support graph should admit a perfect matching for this fuzz regression"
    );

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect(
        "Blossom V should not report NoPerfectMatching on this support-feasible honggfuzz regression",
    );
    validate_matching(18, &edges, &matching);
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_11_support_feasible() {
    let edges = [
        (0, 1, 0),
        (0, 3, -21251),
        (0, 4, 0),
        (0, 5, -18577),
        (0, 6, 32018),
        (0, 7, 0),
        (0, 8, -31624),
        (0, 10, -1387),
        (0, 11, 5376),
        (0, 12, 12819),
        (0, 14, 14768),
        (0, 15, 0),
        (0, 16, -27420),
        (1, 2, -12194),
        (1, 3, -10864),
        (1, 4, 0),
        (1, 5, 32512),
        (1, 11, 0),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 15, 246),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 5, 21845),
        (3, 12, 3072),
        (3, 14, 2920),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 16, -320),
        (5, 8, 12331),
        (5, 14, 3053),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 10, 256),
        (7, 11, -26516),
        (8, 9, 32738),
        (8, 12, -24415),
        (8, 13, 17552),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 16, 0),
        (10, 11, -12279),
        (10, 13, -8642),
        (10, 14, 9837),
        (12, 15, 0),
        (12, 17, -16846),
        (13, 16, 12857),
        (14, 17, 1080),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(
        support.blossom().len(),
        9,
        "support graph should admit a perfect matching for this fuzz regression"
    );

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect(
        "Blossom V should not report NoPerfectMatching on this support-feasible honggfuzz regression",
    );
    validate_matching(18, &edges, &matching);
}

#[test]
fn test_blossom_v_honggfuzz_case_5() {
    let edges = [
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 11, 0),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 20, 21398),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 16, 13954),
        (2, 17, 3199),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (5, 7, 21845),
        (5, 11, -15360),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 14, 381),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (7, 17, 29485),
        (8, 11, 896),
        (8, 15, -318),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, -9253),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 24, 2782),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ];
    let g = build_valued_graph(26, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(26, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -116000);
    assert_eq!(
        matching,
        vec![
            (0, 19),
            (1, 6),
            (2, 7),
            (3, 22),
            (4, 5),
            (8, 15),
            (9, 12),
            (10, 11),
            (13, 20),
            (14, 21),
            (16, 25),
            (17, 24),
            (18, 23),
        ]
    );
}

#[test]
fn test_blossom_v_honggfuzz_case_6() {
    let edges = [
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 11, 0),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 10, 12302),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 17, 3199),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (4, 11, 31704),
        (5, 7, 21845),
        (5, 11, -15360),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 14, 381),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (8, 11, 896),
        (8, 15, -318),
        (8, 17, 19387),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, 23441),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (13, 24, -10229),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 24, 2782),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ];
    let g = build_valued_graph(26, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(26, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -114562);
    assert_eq!(
        matching,
        vec![
            (0, 11),
            (1, 18),
            (2, 7),
            (3, 22),
            (4, 5),
            (6, 14),
            (8, 15),
            (9, 19),
            (10, 17),
            (12, 23),
            (13, 20),
            (16, 25),
            (21, 24),
        ]
    );
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_16() {
    // The saved case-16 replay decodes to the same normalized graph as case 6.
    test_blossom_v_honggfuzz_case_6();
}

#[test]
fn test_blossom_v_honggfuzz_case_7() {
    let edges = [
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 15, 0),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 10, 12302),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 16, 14210),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (4, 11, 31704),
        (5, 7, 21845),
        (5, 11, 0),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (7, 17, 29485),
        (8, 11, 896),
        (8, 15, -318),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, 23441),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 20, 381),
        (17, 24, 2782),
        (18, 20, 21398),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ];
    let g = build_valued_graph(26, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(26, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -113140);
    assert_eq!(
        matching,
        vec![
            (0, 24),
            (1, 23),
            (2, 11),
            (3, 7),
            (4, 5),
            (6, 12),
            (8, 15),
            (9, 22),
            (10, 17),
            (13, 19),
            (14, 21),
            (16, 25),
            (18, 20),
        ]
    );
}

#[test]
fn test_blossom_v_generated_case_416() {
    let edges = [
        (0, 1, 5),
        (0, 3, 65),
        (0, 5, 96),
        (0, 6, 63),
        (0, 8, -85),
        (0, 9, -65),
        (0, 11, -12),
        (0, 13, 0),
        (0, 15, 34),
        (0, 18, -21),
        (0, 19, 64),
        (1, 2, -3),
        (1, 3, 20),
        (1, 4, -88),
        (1, 6, -5),
        (1, 9, -23),
        (1, 10, 86),
        (1, 12, 56),
        (1, 13, 53),
        (1, 15, -50),
        (1, 18, 54),
        (1, 19, 32),
        (2, 5, -30),
        (2, 8, -82),
        (2, 9, -3),
        (2, 10, 38),
        (2, 12, -1),
        (2, 14, -43),
        (2, 15, 21),
        (2, 16, -61),
        (2, 17, 74),
        (3, 4, 2),
        (3, 5, -51),
        (3, 7, 94),
        (3, 8, 12),
        (3, 9, -48),
        (3, 14, -39),
        (3, 16, -57),
        (4, 6, -59),
        (4, 7, 18),
        (4, 8, -70),
        (4, 9, -92),
        (4, 14, 75),
        (4, 17, -89),
        (4, 18, -81),
        (5, 6, 40),
        (5, 7, -48),
        (5, 8, 17),
        (5, 9, 33),
        (5, 10, -5),
        (5, 13, 25),
        (6, 7, 33),
        (6, 8, -94),
        (6, 9, 66),
        (6, 11, 71),
        (6, 12, 98),
        (6, 15, -47),
        (6, 17, 87),
        (6, 18, 75),
        (7, 8, -15),
        (7, 9, 82),
        (7, 10, 35),
        (7, 12, -46),
        (7, 13, -63),
        (7, 14, 89),
        (7, 15, -79),
        (7, 17, 6),
        (7, 18, 15),
        (8, 9, -50),
        (8, 10, -36),
        (8, 11, -20),
        (8, 12, 74),
        (8, 14, 46),
        (8, 16, 98),
        (8, 19, 33),
        (9, 11, -92),
        (9, 12, 92),
        (9, 13, 85),
        (9, 14, 92),
        (9, 15, 23),
        (9, 17, -5),
        (10, 11, 50),
        (10, 12, -32),
        (10, 13, -14),
        (10, 14, -48),
        (10, 15, -74),
        (10, 16, 2),
        (10, 18, -85),
        (10, 19, -36),
        (11, 12, 59),
        (11, 13, 73),
        (11, 14, -94),
        (11, 15, 70),
        (11, 16, 8),
        (11, 18, 35),
        (11, 19, 71),
        (12, 14, -59),
        (12, 15, -86),
        (12, 16, 59),
        (12, 17, 54),
        (12, 18, 11),
        (12, 19, 57),
        (13, 14, 71),
        (13, 15, 57),
        (13, 16, 17),
        (13, 17, 71),
        (13, 19, -52),
        (14, 16, 73),
        (14, 17, -94),
        (14, 18, -31),
        (14, 19, -90),
        (15, 16, 92),
        (15, 17, 87),
        (15, 19, 77),
        (16, 17, -47),
        (16, 18, 75),
        (16, 19, -23),
        (17, 18, -5),
        (17, 19, 89),
        (18, 19, -93),
    ];
    let g = build_valued_graph(20, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(20, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -707);
    assert_eq!(
        matching,
        vec![
            (0, 8),
            (1, 6),
            (2, 16),
            (3, 5),
            (4, 17),
            (7, 13),
            (9, 11),
            (10, 18),
            (12, 15),
            (14, 19),
        ]
    );
}

#[test]
fn test_blossom_v_generated_case_97() {
    let edges = [
        (0, 1, 94),
        (0, 2, 62),
        (0, 3, -67),
        (0, 4, -71),
        (0, 5, -32),
        (0, 6, 71),
        (0, 7, 47),
        (0, 8, -70),
        (0, 9, 32),
        (0, 10, 85),
        (0, 11, 71),
        (0, 12, -43),
        (1, 2, 99),
        (1, 3, 14),
        (1, 4, 82),
        (1, 5, 71),
        (1, 7, 65),
        (1, 8, 99),
        (1, 9, -85),
        (1, 17, 43),
        (2, 3, -82),
        (2, 4, 74),
        (2, 6, -8),
        (2, 10, 27),
        (2, 11, 40),
        (2, 16, 41),
        (2, 17, -40),
        (3, 4, -6),
        (3, 5, 56),
        (3, 6, -6),
        (3, 7, -12),
        (3, 8, 26),
        (3, 11, 94),
        (3, 12, 19),
        (3, 13, -95),
        (3, 14, -7),
        (3, 15, -77),
        (3, 17, -74),
        (4, 5, 65),
        (4, 6, 23),
        (4, 7, -21),
        (4, 11, 37),
        (4, 12, -83),
        (4, 14, -100),
        (5, 13, -19),
        (5, 15, 57),
        (6, 9, -91),
        (7, 8, -11),
        (7, 9, -16),
        (7, 14, -76),
        (7, 15, 95),
        (8, 10, -86),
        (8, 13, 3),
        (8, 14, -14),
        (8, 16, 11),
        (9, 10, -5),
        (9, 12, 41),
        (9, 15, 36),
        (10, 13, 73),
        (10, 16, 35),
        (11, 16, 74),
        (13, 17, 93),
    ];
    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(18, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -329);
    assert_eq!(
        matching,
        vec![(0, 12), (1, 9), (2, 17), (3, 15), (4, 6), (5, 13), (7, 14), (8, 10), (11, 16),]
    );
}

#[test]
fn test_blossom_v_generated_case_91838() {
    let edges = [
        (0, 1, 87),
        (0, 2, -14),
        (0, 15, -84),
        (0, 16, -84),
        (0, 17, 11),
        (1, 2, 32),
        (1, 17, 48),
        (2, 3, -65),
        (2, 4, -50),
        (3, 4, 99),
        (3, 5, 41),
        (4, 5, -84),
        (4, 6, 6),
        (5, 6, -53),
        (5, 7, 23),
        (6, 7, 26),
        (6, 8, 2),
        (7, 8, -19),
        (7, 9, -49),
        (8, 9, -65),
        (8, 10, 33),
        (9, 11, -86),
        (9, 12, -31),
        (10, 11, -44),
        (10, 12, 20),
        (11, 12, -48),
        (11, 13, 91),
        (12, 13, 67),
        (12, 14, 56),
        (13, 14, -91),
        (13, 15, 93),
        (14, 15, -11),
        (14, 16, -17),
        (15, 16, -94),
        (15, 17, -15),
        (16, 17, 0),
    ];
    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(18, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -226);
    assert_eq!(
        matching,
        vec![(0, 15), (1, 17), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 16)]
    );
}

#[test]
fn test_blossom_v_generated_case_87417() {
    let edges = [
        (0, 11, -88),
        (0, 17, -39),
        (0, 18, 19),
        (0, 21, -22),
        (0, 23, -54),
        (0, 24, -25),
        (0, 27, 12),
        (1, 12, -8),
        (1, 15, -97),
        (1, 16, -32),
        (1, 24, 55),
        (2, 10, -49),
        (2, 15, -74),
        (2, 16, -47),
        (3, 12, 100),
        (3, 21, -65),
        (3, 29, -8),
        (4, 9, -45),
        (4, 11, -73),
        (4, 13, -97),
        (4, 15, 81),
        (4, 16, 4),
        (4, 21, -100),
        (4, 22, -25),
        (4, 27, -27),
        (4, 29, -77),
        (5, 9, 16),
        (5, 13, 57),
        (5, 26, -96),
        (5, 27, 63),
        (6, 9, 67),
        (6, 12, 82),
        (6, 13, 44),
        (6, 15, -49),
        (6, 16, 1),
        (7, 8, -8),
        (7, 11, -77),
        (7, 12, -51),
        (7, 13, -50),
        (7, 15, -77),
        (7, 17, 20),
        (7, 19, -21),
        (7, 20, 53),
        (7, 21, -50),
        (7, 22, 64),
        (7, 23, 81),
        (8, 14, -83),
        (8, 15, -25),
        (8, 18, 99),
        (8, 22, -51),
        (8, 25, -32),
        (8, 27, 84),
        (8, 29, 78),
        (9, 11, 78),
        (9, 18, 32),
        (9, 23, -71),
        (9, 29, 2),
        (10, 14, -57),
        (10, 23, -89),
        (11, 14, -22),
        (11, 16, 10),
        (11, 17, 87),
        (11, 20, -91),
        (11, 23, 17),
        (11, 24, -39),
        (11, 26, 11),
        (11, 27, 22),
        (12, 22, 87),
        (12, 23, -83),
        (12, 24, 75),
        (12, 25, -36),
        (12, 27, 12),
        (12, 28, 51),
        (13, 20, 26),
        (13, 22, -58),
        (13, 27, 26),
        (13, 28, -57),
        (14, 15, 41),
        (14, 22, -73),
        (14, 25, -63),
        (14, 26, 73),
        (14, 29, -50),
        (15, 16, 11),
        (15, 18, 28),
        (15, 19, -19),
        (15, 26, -5),
        (15, 29, 73),
        (16, 19, -67),
        (16, 20, -85),
        (16, 21, -29),
        (16, 25, -99),
        (17, 19, -79),
        (17, 24, 85),
        (18, 23, 5),
        (18, 28, 79),
        (19, 20, -86),
        (19, 21, 77),
        (20, 23, -19),
        (20, 25, -38),
        (20, 26, -83),
        (20, 28, 13),
        (21, 22, 31),
        (21, 27, -12),
        (21, 28, 97),
        (21, 29, -97),
        (22, 23, 41),
        (23, 28, -22),
        (24, 26, -76),
        (25, 28, 57),
        (26, 28, 39),
        (27, 28, 27),
    ];
    let g = build_valued_graph(30, &edges);
    let matching = g.blossom_v().expect("should find perfect matching");
    validate_matching(30, &edges, &matching);
    assert_eq!(matching_cost(&edges, &matching), -771);
    assert_eq!(
        matching,
        vec![
            (0, 24),
            (1, 15),
            (2, 10),
            (3, 21),
            (4, 29),
            (5, 26),
            (6, 9),
            (7, 12),
            (8, 14),
            (11, 20),
            (13, 22),
            (16, 25),
            (17, 19),
            (18, 23),
            (27, 28),
        ]
    );
}

#[test]
#[ignore = "debug helper for case #214 order sensitivity"]
fn debug_blossom_v_generated_case_214_order_sensitivity() {
    let edges = [
        (0, 1, 90),
        (0, 2, -66),
        (0, 3, -13),
        (0, 4, -83),
        (0, 5, 73),
        (0, 7, -70),
        (0, 9, -67),
        (0, 11, 39),
        (0, 12, 40),
        (0, 13, -57),
        (0, 14, -32),
        (0, 18, -54),
        (0, 19, 73),
        (0, 20, -60),
        (1, 2, -96),
        (1, 3, -17),
        (1, 4, 12),
        (1, 5, -20),
        (1, 6, 42),
        (1, 8, -5),
        (1, 11, 31),
        (1, 14, 0),
        (1, 16, -90),
        (1, 21, -90),
        (2, 3, -88),
        (2, 4, -38),
        (2, 5, 80),
        (2, 7, 7),
        (2, 9, -27),
        (2, 10, -30),
        (2, 12, 66),
        (2, 17, -46),
        (2, 18, -66),
        (2, 19, -61),
        (3, 4, -11),
        (3, 5, 90),
        (3, 6, 77),
        (3, 8, 82),
        (3, 10, 96),
        (3, 11, -90),
        (3, 12, -91),
        (3, 18, 30),
        (4, 6, -93),
        (4, 7, -5),
        (4, 8, -38),
        (4, 10, 2),
        (4, 14, 57),
        (4, 18, -11),
        (5, 6, 98),
        (5, 7, 91),
        (5, 15, -63),
        (5, 21, -83),
        (6, 8, 89),
        (6, 12, 58),
        (6, 13, 98),
        (7, 9, -27),
        (7, 10, -64),
        (7, 15, 59),
        (7, 16, -66),
        (7, 19, -5),
        (8, 9, -28),
        (8, 11, -29),
        (8, 13, -70),
        (8, 15, 38),
        (9, 17, -91),
        (10, 17, -2),
        (10, 20, 28),
        (10, 21, -11),
        (12, 13, -76),
        (12, 16, -8),
        (12, 19, 68),
        (13, 14, -57),
        (13, 20, -42),
        (13, 21, -16),
        (14, 15, 78),
        (14, 16, -2),
        (16, 17, -47),
        (18, 20, -35),
    ];

    let sorted = build_valued_graph(22, &edges);
    let sorted_matching = sorted.blossom_v().expect("sorted graph should solve");
    eprintln!(
        "sorted matching={sorted_matching:?} cost={}",
        matching_cost(&edges, &sorted_matching)
    );

    let preserved = build_valued_graph_preserve_input_order(22, &edges);
    let preserved_matching = preserved.blossom_v().expect("preserved graph should solve");
    eprintln!(
        "preserved matching={preserved_matching:?} cost={}",
        matching_cost(&edges, &preserved_matching)
    );
}

#[test]
fn test_blossom_v_generated_case_24595() {
    let edges = vec![
        (0usize, 7usize, -6i32),
        (0, 9, 14),
        (0, 16, -17),
        (1, 2, 34),
        (1, 3, 98),
        (1, 6, 58),
        (1, 10, 24),
        (1, 11, -22),
        (1, 15, 49),
        (2, 3, 72),
        (2, 5, 55),
        (2, 6, 72),
        (2, 8, -82),
        (2, 10, 90),
        (2, 11, -13),
        (2, 15, 26),
        (2, 17, 93),
        (3, 5, -51),
        (3, 6, -66),
        (3, 8, 10),
        (3, 9, -84),
        (3, 11, -93),
        (3, 14, 43),
        (3, 15, -9),
        (3, 17, -50),
        (4, 12, 45),
        (4, 13, 43),
        (4, 14, 16),
        (5, 6, 84),
        (5, 7, 32),
        (5, 8, -92),
        (5, 9, -58),
        (5, 14, 66),
        (5, 15, 76),
        (5, 16, 47),
        (5, 17, -59),
        (6, 8, -9),
        (6, 9, -1),
        (6, 10, 70),
        (6, 15, -55),
        (6, 17, -23),
        (7, 9, 42),
        (7, 16, -98),
        (7, 17, 41),
        (8, 9, -71),
        (8, 14, -66),
        (8, 15, 10),
        (8, 17, 24),
        (9, 15, -75),
        (9, 16, -62),
        (9, 17, 35),
        (10, 11, -67),
        (10, 15, 53),
        (12, 13, 17),
        (14, 17, -75),
        (15, 17, -18),
        (16, 17, 20),
    ];
    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("generated case #24595 should solve");
    assert_eq!(matching_cost(&edges, &matching), -316);
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_12_support_feasible() {
    let edges = [
        (0usize, 1usize, 0i32),
        (0, 2, 0),
        (0, 3, -21251),
        (0, 5, -18577),
        (0, 6, 32018),
        (0, 8, -31624),
        (0, 10, -1387),
        (0, 11, -2023),
        (0, 12, 12819),
        (0, 14, 21845),
        (0, 15, 0),
        (0, 16, -26363),
        (0, 17, 0),
        (1, 2, -12194),
        (1, 3, -10864),
        (1, 4, 0),
        (1, 5, 32512),
        (1, 11, 0),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 15, 13302),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 10, -1),
        (3, 12, 3072),
        (3, 14, 2920),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 16, -320),
        (5, 8, 12331),
        (5, 13, 21356),
        (5, 14, 3053),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 10, -768),
        (7, 11, -26516),
        (8, 9, 32738),
        (8, 12, -24415),
        (8, 13, 17552),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 16, 0),
        (10, 11, -12279),
        (10, 13, -8642),
        (12, 15, 0),
        (12, 17, -16846),
        (13, 16, 12857),
        (14, 17, 7981),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(support.blossom().len(), 9, "support graph should admit a perfect matching");

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("honggfuzz case 12 should solve");
    validate_matching(18, &edges, &matching);
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_13_support_feasible() {
    let edges = [
        (0usize, 1usize, 0i32),
        (0, 3, -21502),
        (0, 4, 0),
        (0, 5, -18577),
        (0, 6, -2023),
        (0, 7, 0),
        (0, 8, -31624),
        (0, 10, -1387),
        (0, 11, 5376),
        (0, 12, 12819),
        (0, 14, 14768),
        (0, 15, 0),
        (0, 16, -27420),
        (1, 2, -12194),
        (1, 3, -10864),
        (1, 4, 0),
        (1, 5, 0),
        (1, 11, 0),
        (1, 12, -1),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 14, 21845),
        (2, 15, 29174),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 5, 32767),
        (3, 10, -1),
        (3, 12, 3072),
        (3, 14, 1128),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 16, -320),
        (5, 8, 12331),
        (5, 13, 21612),
        (5, 14, 3053),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (6, 13, 32018),
        (7, 10, 256),
        (7, 11, -26516),
        (8, 9, 32738),
        (8, 12, -24415),
        (8, 13, 27792),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 16, 0),
        (10, 11, -12279),
        (10, 13, -8642),
        (12, 15, -24),
        (12, 17, -16846),
        (13, 16, 12857),
        (14, 17, 1080),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(support.blossom().len(), 9, "support graph should admit a perfect matching");

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("honggfuzz case 13 should solve");
    validate_matching(18, &edges, &matching);
}

#[test]
fn test_blossom_v_regression_honggfuzz_sigabrt_case_14_support_feasible() {
    let edges = [
        (0usize, 1usize, 0i32),
        (0, 2, 0),
        (0, 3, -21251),
        (0, 5, -18577),
        (0, 6, 32018),
        (0, 8, -31624),
        (0, 10, -1387),
        (0, 11, -2023),
        (0, 12, 12819),
        (0, 14, 21845),
        (0, 15, 0),
        (0, 16, -26363),
        (0, 17, 0),
        (1, 2, -12194),
        (1, 3, -10864),
        (1, 4, 0),
        (1, 5, 32512),
        (1, 7, 0),
        (1, 11, 0),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 5, 0),
        (2, 15, 13302),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 10, -1),
        (3, 12, 3072),
        (3, 14, 2920),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 16, -320),
        (5, 8, 12331),
        (5, 13, 21356),
        (5, 14, 3053),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 10, -768),
        (7, 11, -26516),
        (8, 9, 32738),
        (8, 12, -24415),
        (8, 13, 17552),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 16, 0),
        (10, 11, -12279),
        (10, 13, -8642),
        (12, 15, 0),
        (12, 17, -16846),
        (13, 16, 12857),
        (14, 17, 7981),
    ];
    let support = build_support_graph(18, &edges);
    assert_eq!(support.blossom().len(), 9, "support graph should admit a perfect matching");

    let g = build_valued_graph(18, &edges);
    let matching = g.blossom_v().expect("honggfuzz case 14 should solve");
    validate_matching(18, &edges, &matching);
}
