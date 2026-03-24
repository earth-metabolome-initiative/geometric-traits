//! Cross-validation tests for exact Karp-Sipser wrappers.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::*,
};

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

fn assert_all_ks_exact(g: &SymmetricCSR2D<CSR2D<usize, usize, usize>>) {
    let blossom = g.blossom();
    let mv = g.micali_vazirani();
    let blum = g.blum();

    let blossom_ks1 = g.blossom_with_karp_sipser(KarpSipserRules::Degree1);
    let blossom_ks12 = g.blossom_with_karp_sipser(KarpSipserRules::Degree1And2);
    let mv_ks1 = g.micali_vazirani_with_karp_sipser(KarpSipserRules::Degree1);
    let mv_ks12 = g.micali_vazirani_with_karp_sipser(KarpSipserRules::Degree1And2);
    let blum_ks1 = g.blum_with_karp_sipser(KarpSipserRules::Degree1);
    let blum_ks12 = g.blum_with_karp_sipser(KarpSipserRules::Degree1And2);

    for matching in [
        &blossom,
        &mv,
        &blum,
        &blossom_ks1,
        &blossom_ks12,
        &mv_ks1,
        &mv_ks12,
        &blum_ks1,
        &blum_ks12,
    ] {
        validate_matching(g, matching);
    }

    let expected = blossom.len();
    assert_eq!(mv.len(), expected);
    assert_eq!(blum.len(), expected);
    assert_eq!(blossom_ks1.len(), expected);
    assert_eq!(blossom_ks12.len(), expected);
    assert_eq!(mv_ks1.len(), expected);
    assert_eq!(mv_ks12.len(), expected);
    assert_eq!(blum_ks1.len(), expected);
    assert_eq!(blum_ks12.len(), expected);
}

#[test]
fn test_structured_graph_families() {
    for n in 0..=12 {
        assert_all_ks_exact(&complete_graph(n));
    }
    for n in 1..=16 {
        assert_all_ks_exact(&path_graph(n));
    }
    for n in 3..=16 {
        assert_all_ks_exact(&cycle_graph(n));
    }
    for n in 1..=12 {
        assert_all_ks_exact(&star_graph(n));
    }
    for rows in 1..=4 {
        for cols in 1..=4 {
            assert_all_ks_exact(&grid_graph(rows, cols));
        }
    }
    for rows in 3..=5 {
        for cols in 3..=5 {
            assert_all_ks_exact(&torus_graph(rows, cols));
        }
    }
    for n in 2..=8 {
        assert_all_ks_exact(&crown_graph(n));
    }
    for k in 3..=5 {
        for p in 0..=2 {
            assert_all_ks_exact(&barbell_graph(k, p));
        }
    }
    for d in 0..=4 {
        assert_all_ks_exact(&hypercube_graph(d));
    }
    for n in 1..=6 {
        assert_all_ks_exact(&friendship_graph(n));
    }
    for m in 1..=5 {
        for n in m..=5 {
            assert_all_ks_exact(&complete_bipartite_graph(m, n));
        }
    }
}

#[test]
fn test_seeded_random_graph_families() {
    for n in 5..=18 {
        assert_all_ks_exact(&erdos_renyi_gnp(42 + n as u64, n, 0.3));
        let m = n * (n - 1) / 4;
        assert_all_ks_exact(&erdos_renyi_gnm(123 + n as u64, n, m));
        assert_all_ks_exact(&barabasi_albert(77 + n as u64, n, 2));
    }

    for n in 10..=18 {
        assert_all_ks_exact(&watts_strogatz(99 + n as u64, n, 4, 0.3));
    }

    for &(n, k) in &[(10, 3), (12, 3), (16, 4), (18, 3)] {
        assert_all_ks_exact(&random_regular_graph(55 + n as u64, n, k));
    }
}
