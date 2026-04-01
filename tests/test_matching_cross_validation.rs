//! Cross-validation tests for matching algorithms (Blossom, Gabow 1976,
//! Micali-Vazirani, Kocay) on generated graph families.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::*,
};

// ============================================================================
// Helpers
// ============================================================================

/// Validate a matching: pairs are ordered, edges exist, no vertex reused.
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

fn windmill_matching_size(num_cliques: usize, clique_size: usize) -> usize {
    num_cliques * ((clique_size - 1) / 2) + usize::from(clique_size % 2 == 0)
}

/// Run all exact matchers on the same graph and assert they agree.
fn assert_all_agree(g: &SymmetricCSR2D<CSR2D<usize, usize, usize>>) {
    let bl = g.blossom();
    let gabow = g.gabow_1976();
    let mv = g.micali_vazirani();
    validate_matching(g, &bl);
    validate_matching(g, &gabow);
    validate_matching(g, &mv);
    assert_eq!(
        bl.len(),
        gabow.len(),
        "Blossom ({}) != Gabow1976 ({}) on graph with order {}",
        bl.len(),
        gabow.len(),
        g.order()
    );
    assert_eq!(
        bl.len(),
        mv.len(),
        "Blossom ({}) != MV ({}) on graph with order {}",
        bl.len(),
        mv.len(),
        g.order()
    );

    // Kocay with unit capacities should match.
    let n: usize = g.order();
    if n == 0 {
        return;
    }
    let mut vcsr: ValuedCSR2D<usize, usize, usize, usize> =
        SparseMatrixMut::with_sparse_shaped_capacity(
            (n, n),
            geometric_traits::traits::Edges::number_of_edges(g),
        );
    for row in g.row_indices() {
        for col in g.sparse_row(row) {
            MatrixMut::add(&mut vcsr, (row, col, 1)).unwrap();
        }
    }
    let budgets = vec![1usize; n];
    let flow = vcsr.kocay(&budgets);
    let kocay_total: usize = flow.iter().map(|&(_, _, f)| f).sum();
    assert_eq!(
        bl.len(),
        kocay_total,
        "Blossom ({}) != Kocay ({}) on graph with order {}",
        bl.len(),
        kocay_total,
        n
    );
}

// ============================================================================
// Deterministic graph generators
// ============================================================================

#[test]
fn test_regression_random_counterexample() {
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
    assert_all_agree(&g);
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
    assert_all_agree(&g);
}

#[test]
fn test_regression_random_counterexample_size_mismatch() {
    let g = build_graph(
        8,
        &[(0, 2), (0, 3), (0, 4), (0, 7), (1, 2), (1, 3), (3, 7), (4, 5), (4, 6), (4, 7), (5, 7)],
    );
    assert_all_agree(&g);
}

#[test]
fn test_complete_graphs() {
    for n in 0..=20 {
        let g = complete_graph(n);
        assert_all_agree(&g);
        // Known: matching size = floor(n/2)
        assert_eq!(g.blossom().len(), n / 2, "K{n} matching size");
    }
}

#[test]
fn test_cycle_graphs() {
    for n in 3..=30 {
        let g = cycle_graph(n);
        assert_all_agree(&g);
        assert_eq!(g.blossom().len(), n / 2, "C{n} matching size");
    }
}

#[test]
fn test_path_graphs() {
    for n in 1..=30 {
        let g = path_graph(n);
        assert_all_agree(&g);
        assert_eq!(g.blossom().len(), n / 2, "P{n} matching size");
    }
}

#[test]
fn test_star_graphs() {
    for n in 1..=20 {
        let g = star_graph(n);
        assert_all_agree(&g);
        let expected = usize::from(n > 1);
        assert_eq!(g.blossom().len(), expected, "star({n}) matching size");
    }
}

#[test]
fn test_grid_graphs() {
    for rows in 1..=8 {
        for cols in 1..=8 {
            let g = grid_graph(rows, cols);
            assert_all_agree(&g);
            // Grid r×c has perfect matching iff r*c is even
            let n = rows * cols;
            assert_eq!(g.blossom().len(), n / 2, "grid {rows}x{cols}");
        }
    }
}

#[test]
fn test_torus_graphs() {
    for rows in 3..=8 {
        for cols in 3..=8 {
            let g = torus_graph(rows, cols);
            assert_all_agree(&g);
            let n = rows * cols;
            assert_eq!(g.blossom().len(), n / 2, "torus {rows}x{cols}");
        }
    }
}

#[test]
fn test_hypercube_graphs() {
    for d in 0..=6 {
        let g = hypercube_graph(d);
        assert_all_agree(&g);
        let n = 1 << d;
        assert_eq!(g.blossom().len(), n / 2, "Q{d} matching size");
    }
}

#[test]
fn test_barbell_graphs() {
    for k in 3..=8 {
        for p in 0..=4 {
            let g = barbell_graph(k, p);
            assert_all_agree(&g);
        }
    }
}

#[test]
fn test_crown_graphs() {
    for n in 2..=12 {
        let g = crown_graph(n);
        assert_all_agree(&g);
        // Crown graph on 2n vertices: perfect matching of size n
        assert_eq!(g.blossom().len(), n, "crown({n}) matching size");
    }
}

#[test]
fn test_wheel_graphs() {
    for n in 3..=20 {
        let g = wheel_graph(n);
        assert_all_agree(&g);
        // wheel_graph(n) has n+1 vertices: hub + n-rim.
        // Odd rim: hub matches one rim vertex, rest pair → (n+1)/2.
        // Even rim: all rim vertices pair → n/2 (hub exposed).
        let expected = n.div_ceil(2);
        assert_eq!(g.blossom().len(), expected, "wheel({n}) matching size");
    }
}

#[test]
fn test_complete_bipartite_graphs() {
    for m in 1..=8 {
        for n in m..=8 {
            let g = complete_bipartite_graph(m, n);
            assert_all_agree(&g);
            assert_eq!(g.blossom().len(), m, "K_{m},{n} matching size");
        }
    }
}

#[test]
fn test_petersen_graph_gen() {
    let g = petersen_graph();
    assert_all_agree(&g);
    assert_eq!(g.blossom().len(), 5);
}

#[test]
fn test_turan_graphs() {
    for n in 2..=15 {
        for r in 2..=n.min(6) {
            let g = turan_graph(n, r);
            assert_all_agree(&g);
        }
    }
}

#[test]
fn test_friendship_graphs() {
    for n in 1..=8 {
        let g = friendship_graph(n);
        assert_all_agree(&g);
        // Friendship graph with n triangles: matching = n
        assert_eq!(g.blossom().len(), n, "friendship({n}) matching size");
    }
}

#[test]
fn test_windmill_graphs() {
    for num_cliques in 1..=6 {
        for clique_size in 2..=5 {
            let g = windmill_graph(num_cliques, clique_size);
            assert_all_agree(&g);
            assert_eq!(
                g.blossom().len(),
                windmill_matching_size(num_cliques, clique_size),
                "windmill({num_cliques},{clique_size}) matching size"
            );
        }
    }
}

// ============================================================================
// Random graph generators (seeded for determinism)
// ============================================================================

#[test]
fn test_erdos_renyi_gnp() {
    for n in 5..=40 {
        let g = erdos_renyi_gnp(42 + n as u64, n, 0.3);
        assert_all_agree(&g);
    }
}

#[test]
fn test_erdos_renyi_gnm() {
    for n in 5..=40 {
        let m = n * (n - 1) / 4; // ~50% density
        let g = erdos_renyi_gnm(123 + n as u64, n, m);
        assert_all_agree(&g);
    }
}

#[test]
fn test_barabasi_albert() {
    for n in 5..=40 {
        let g = barabasi_albert(77 + n as u64, n, 2);
        assert_all_agree(&g);
    }
}

#[test]
fn test_watts_strogatz() {
    for n in 10..=40 {
        let g = watts_strogatz(99 + n as u64, n, 4, 0.3);
        assert_all_agree(&g);
    }
}

#[test]
fn test_random_regular() {
    // k must divide n*k evenly (n*k even)
    for &(n, k) in &[(10, 3), (12, 3), (20, 4), (15, 4), (30, 3)] {
        let g = random_regular_graph(55, n, k)
            .expect("cross-validation inputs should admit a simple regular graph");
        assert_all_agree(&g);
    }
}
