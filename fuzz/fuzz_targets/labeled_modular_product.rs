//! Fuzz harness for the labeled modular product algorithm.
//!
//! Verifies symmetry, no self-loops, label-based edge condition, and that the
//! labeled result is always a subset of the unlabeled modular product.

use geometric_traits::{
    impls::{BitSquareMatrix, SquareCSR2D, ValuedCSR2D},
    prelude::*,
    traits::MatrixMut,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            if data.len() < 3 {
                return;
            }
            let n1 = (data[0] % 5) as usize + 2; // 2..6
            let n2 = (data[1] % 5) as usize + 2;
            let max_e1 = n1 * (n1 - 1) / 2;
            let max_e2 = n2 * (n2 - 1) / 2;
            // Each edge needs 1 byte for presence + 1 byte for label
            let needed = 2 + max_e1 * 2 + max_e2 * 2;
            if data.len() < needed {
                return;
            }

            // Build G1 as a valued symmetric matrix.
            let mut edges1 = Vec::new();
            let mut idx = 2;
            for u in 0..n1 {
                for v in (u + 1)..n1 {
                    if data[idx] & 1 != 0 {
                        let label = data[idx + 1] % 3; // 3 possible labels
                        edges1.push((u, v, label));
                    }
                    idx += 2;
                }
            }

            let mut edges2 = Vec::new();
            for u in 0..n2 {
                for v in (u + 1)..n2 {
                    if data[idx] & 1 != 0 {
                        let label = data[idx + 1] % 3;
                        edges2.push((u, v, label));
                    }
                    idx += 2;
                }
            }

            // Build valued symmetric matrices.
            let g1 = build_valued_sym(n1, &edges1);
            let g2 = build_valued_sym(n2, &edges2);

            // Build unvalued BitSquareMatrix for comparison.
            let b1 = BitSquareMatrix::from_symmetric_edges(
                n1,
                edges1.iter().map(|&(u, v, _)| (u, v)),
            );
            let b2 = BitSquareMatrix::from_symmetric_edges(
                n2,
                edges2.iter().map(|&(u, v, _)| (u, v)),
            );

            let pairs: Vec<(usize, usize)> =
                (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect();

            let labeled = g1.labeled_modular_product(&g2, &pairs);
            let unlabeled = b1.modular_product(&b2, &pairs);

            let p = pairs.len();
            assert_eq!(labeled.order(), p);

            for a in 0..p {
                assert!(!labeled.has_entry(a, a), "self-loop");
                for b in (a + 1)..p {
                    // Symmetry.
                    assert_eq!(
                        labeled.has_entry(a, b),
                        labeled.has_entry(b, a),
                        "asymmetric"
                    );

                    // Labeled is subset of unlabeled.
                    if labeled.has_entry(a, b) {
                        assert!(
                            unlabeled.has_entry(a, b),
                            "labeled edge not in unlabeled product"
                        );
                    }

                    // Verify edge condition.
                    let (u1, u2) = pairs[a];
                    let (v1, v2) = pairs[b];
                    if u1 != v1 && u2 != v2 {
                        let l1 = g1.sparse_value_at(u1, v1);
                        let l2 = g2.sparse_value_at(u2, v2);
                        assert_eq!(
                            labeled.has_entry(a, b),
                            l1 == l2,
                            "edge condition violated"
                        );
                    } else {
                        assert!(!labeled.has_entry(a, b), "invalid edge");
                    }
                }
            }
        });
    }
}

fn build_valued_sym(
    n: usize,
    upper_edges: &[(usize, usize, u8)],
) -> SquareCSR2D<ValuedCSR2D<usize, usize, usize, u8>> {
    let mut all: Vec<(usize, usize, u8)> = Vec::new();
    for &(r, c, v) in upper_edges {
        all.push((r, c, v));
        all.push((c, r, v));
    }
    all.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut valued: ValuedCSR2D<usize, usize, usize, u8> =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), all.len());
    for (r, c, v) in all {
        MatrixMut::add(&mut valued, (r, c, v)).unwrap();
    }
    SquareCSR2D::from_parts(valued, 0)
}
