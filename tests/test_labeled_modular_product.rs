//! Tests for the labeled modular product algorithm.
#![cfg(feature = "alloc")]

use geometric_traits::{
    impls::{BitSquareMatrix, SquareCSR2D, ValuedCSR2D},
    prelude::*,
    traits::MatrixMut,
};

type TestValued = SquareCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;

/// Build a symmetric valued square matrix from upper-triangular edges.
fn build_valued(n: usize, edges: &[(usize, usize, u8)]) -> TestValued {
    let mut valued: ValuedCSR2D<usize, usize, usize, u8> =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), edges.len() * 2);
    // Expand to both directions and sort.
    let mut all: Vec<(usize, usize, u8)> = Vec::with_capacity(edges.len() * 2);
    for &(r, c, v) in edges {
        all.push((r, c, v));
        if r != c {
            all.push((c, r, v));
        }
    }
    all.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (r, c, v) in all {
        MatrixMut::add(&mut valued, (r, c, v)).unwrap();
    }
    SquareCSR2D::from_parts(valued, 0)
}

// ============================================================================
// Basic structural tests
// ============================================================================

#[test]
fn test_empty_pairs() {
    let g = build_valued(3, &[(0, 1, 1), (1, 2, 2)]);
    let mp = g.labeled_modular_product(&g, &[], |a, b| a == b);
    assert_eq!(mp.order(), 0);
}

#[test]
fn test_uniform_labels_matches_unlabeled() {
    // When all edges have the same label, labeled modular product should
    // match the unlabeled one exactly.
    let g1 = build_valued(3, &[(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
    let g2 = build_valued(3, &[(0, 1, 1), (1, 2, 1)]);

    let b1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let b2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);

    let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();

    let labeled = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);
    let unlabeled = b1.modular_product(&b2, &pairs);

    let n = pairs.len();
    for a in 0..n {
        for b in 0..n {
            assert_eq!(
                labeled.has_entry(a, b),
                unlabeled.has_entry(a, b),
                "mismatch at ({a}, {b})"
            );
        }
    }
}

#[test]
fn test_mixed_labels_filters() {
    // G1: edge (0,1) with label 1, edge (1,2) with label 2
    // G2: edge (0,1) with label 1, edge (1,2) with label 1
    //
    // Pair (0,0)-(1,1): G1 has edge (0,1)=Some(1), G2 has edge (0,1)=Some(1) ->
    // match Pair (1,1)-(2,2): G1 has edge (1,2)=Some(2), G2 has edge
    // (1,2)=Some(1) -> no match
    let g1 = build_valued(3, &[(0, 1, 1), (1, 2, 2)]);
    let g2 = build_valued(3, &[(0, 1, 1), (1, 2, 1)]);

    let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);

    // (0,0) is pair index 0, (1,1) is pair index 4, (2,2) is pair index 8
    // Pair (0,0)-(1,1): both have edge -> check label match
    assert!(mp.has_entry(0, 4), "same label edges should connect");

    // Pair (1,1)-(2,2): both have edge but labels differ
    assert!(!mp.has_entry(4, 8), "different label edges should NOT connect");
}

#[test]
fn test_none_none_compatibility() {
    // When neither graph has the edge, the labels are None == None -> compatible.
    // G1: empty graph on 3 nodes
    // G2: empty graph on 3 nodes
    let g1 = build_valued(3, &[]);
    let g2 = build_valued(3, &[]);

    let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);

    // All pairs with u1!=v1 and u2!=v2 should be connected (None == None).
    for (a, &(u1, u2)) in pairs.iter().enumerate() {
        for (b, &(v1, v2)) in pairs.iter().enumerate().skip(a + 1) {
            let expected = u1 != v1 && u2 != v2;
            assert_eq!(mp.has_entry(a, b), expected, "pair ({a},{b}) mismatch");
        }
    }
}

#[test]
fn test_some_none_incompatibility() {
    // G1 has an edge, G2 doesn't -> Some(x) != None -> no connection.
    let g1 = build_valued(2, &[(0, 1, 1)]);
    let g2 = build_valued(2, &[]);

    let pairs = vec![(0usize, 0usize), (1, 1)];
    let mp = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);

    // (0,0)-(1,1): G1 has edge (0,1)=Some(1), G2 has no edge (0,1)=None -> no match
    assert!(!mp.has_entry(0, 1));
}

#[test]
fn test_symmetry_and_no_self_loops() {
    let g1 = build_valued(4, &[(0, 1, 1), (1, 2, 2), (2, 3, 1)]);
    let g2 = build_valued(3, &[(0, 1, 1), (1, 2, 2)]);

    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let mp = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);

    let n = mp.order();
    for i in 0..n {
        assert!(!mp.has_entry(i, i), "self-loop at {i}");
        for j in 0..n {
            assert_eq!(mp.has_entry(i, j), mp.has_entry(j, i), "asymmetry at ({i}, {j})");
        }
    }
}

#[test]
fn test_labeled_subset_of_unlabeled() {
    // The labeled modular product is always a subset of the unlabeled one,
    // since label compatibility is a stricter condition than adjacency match.
    let g1 = build_valued(4, &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (0, 3, 3)]);
    let g2 = build_valued(3, &[(0, 1, 1), (0, 2, 2), (1, 2, 1)]);

    let b1 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3), (0, 3)]);
    let b2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);

    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    let labeled = g1.labeled_modular_product(&g2, &pairs, |a, b| a == b);
    let unlabeled = b1.modular_product(&b2, &pairs);

    let n = pairs.len();
    for a in 0..n {
        for b in (a + 1)..n {
            if labeled.has_entry(a, b) {
                assert!(
                    unlabeled.has_entry(a, b),
                    "labeled edge ({a},{b}) not in unlabeled product"
                );
            }
        }
    }
}

#[test]
fn test_labeled_modular_product_filtered_and_into_parts() {
    let g1 = build_valued(3, &[(0, 1, 1), (1, 2, 2)]);
    let g2 = build_valued(3, &[(0, 1, 3), (1, 2, 5)]);

    let result = g1.labeled_modular_product_filtered(
        &g2,
        |left, right| left == right,
        |a, b| {
            match (a, b) {
                (None, None) => true,
                (Some(left), Some(right)) => left % 2 == right % 2,
                _ => false,
            }
        },
    );

    assert_eq!(result.vertex_pairs(), &[(0, 0), (1, 1), (2, 2)]);
    assert_eq!(result.matrix().order(), 3);
    assert!(result.matrix().has_entry(0, 1), "odd labels should be compatible");
    assert!(!result.matrix().has_entry(1, 2), "even/odd labels should be incompatible");

    let expected_pairs = result.vertex_pairs().to_vec();
    let expected_matrix = result.matrix().clone();
    let (matrix, pairs) = result.into_parts();

    assert_eq!(pairs, expected_pairs);
    assert_eq!(matrix, expected_matrix);
}
