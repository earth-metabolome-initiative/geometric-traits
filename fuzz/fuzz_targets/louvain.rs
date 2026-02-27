//! Fuzz harness for the Louvain community detection algorithm.
//!
//! Invariants checked:
//! 1. Louvain never panics on arbitrary input.
//! 2. On valid symmetric graphs: partition length == node count,
//!    modularity in [-0.5, 1.0], and deterministic across two runs.

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        GenericEdgesBuilder, IntoUsize, Louvain, LouvainConfig, Matrix2D, SparseMatrix2D,
        SparseValuedMatrix2D,
    },
    traits::EdgesBuilder,
};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;
type SymCsr = ValuedCSR2D<u8, u8, u8, f64>;

fn check_arbitrary_input(csr: &Csr) {
    let _ = Louvain::<usize>::louvain(csr, &LouvainConfig::default());
}

/// Returns `true` when edge weights span a numerically stable range,
/// avoiding extreme floating-point regimes that make modularity
/// incomparable across runs even though the algorithm is correct.
fn weights_are_numerically_stable(csr: &Csr) -> bool {
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0_f64;

    for row in csr.row_indices() {
        for val in csr.sparse_row_values(row) {
            if val > 0.0 && val.is_finite() && val.is_normal() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    if min_val.is_infinite() {
        return true; // no valid weights — will be filtered by edge extraction
    }

    min_val >= f64::MIN_POSITIVE && max_val <= 1e150 && (max_val / min_val) <= 1e12
}

fn check_symmetric_invariants(csr: &Csr) {
    let rows = csr.number_of_rows().into_usize();
    let cols = csr.number_of_columns().into_usize();
    if rows != cols || rows == 0 || rows > u8::MAX as usize {
        return;
    }

    // Skip modularity-bound assertions for extreme weight ranges.
    if !weights_are_numerically_stable(csr) {
        return;
    }

    let n = rows as u8;

    // Extract upper-triangle edges with finite positive weights, then mirror.
    let mut edges: Vec<(u8, u8, f64)> = Vec::new();
    for row in csr.row_indices() {
        let r = row.into_usize();
        if r >= rows {
            continue;
        }
        for (col, val) in csr.sparse_row(row).zip(csr.sparse_row_values(row)) {
            let c = col.into_usize();
            if r <= c && val.is_finite() && val.is_normal() && val > 0.0 {
                let r8 = r as u8;
                let c8 = c as u8;
                edges.push((r8, c8, val));
                if r8 != c8 {
                    edges.push((c8, r8, val));
                }
            }
        }
    }

    if edges.is_empty() {
        return;
    }

    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    edges.dedup_by(|(r1, c1, _), (r2, c2, _)| (*r1, *c1) == (*r2, *c2));

    let Ok(edge_count) = u8::try_from(edges.len()) else {
        return;
    };

    let sym_csr = GenericEdgesBuilder::<_, SymCsr>::default()
        .expected_number_of_edges(edge_count)
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build();

    let Ok(sym_csr) = sym_csr else {
        return;
    };

    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&sym_csr, &config)
        .expect("Louvain must not fail on a valid symmetric graph");

    let n = n as usize;
    assert_eq!(
        result.final_partition().len(),
        n,
        "partition length must equal node count"
    );
    let modularity = result.final_modularity();
    assert!(
        modularity >= -0.5 - 1e-9 && modularity <= 1.0 + 1e-9,
        "modularity {modularity} out of [-0.5, 1.0] (with FP tolerance)"
    );

    // Determinism check.
    let result2 = Louvain::<usize>::louvain(&sym_csr, &config).unwrap();
    assert_eq!(
        result.final_partition(),
        result2.final_partition(),
        "Louvain must be deterministic for the same seed"
    );
    assert!(
        (result.final_modularity() - result2.final_modularity()).abs() <= 1.0e-12,
        "modularity must be deterministic"
    );
}

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_arbitrary_input(&csr);
            check_symmetric_invariants(&csr);
        });
    }
}
