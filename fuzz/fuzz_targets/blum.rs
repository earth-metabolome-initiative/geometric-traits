//! Submodule for fuzzing Blum's maximum matching algorithm.

use geometric_traits::prelude::*;
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SymmetricCSR2D<CSR2D<u16, u16, u16>>| {
            let n = csr.order() as usize;
            if n > 256 {
                return;
            }
            let blum_matching = csr.blum();
            let bl_matching = csr.blossom();

            assert_eq!(
                blum_matching.len(),
                bl_matching.len(),
                "Blum and Blossom disagree on matching size (n={n})"
            );

            assert!(blum_matching.len() <= n / 2);

            let mut matched = vec![false; n];
            for &(u, v) in &blum_matching {
                let ui = usize::from(u);
                let vi = usize::from(v);
                assert!(u < v);
                assert!(!matched[ui], "vertex {u} matched twice");
                assert!(!matched[vi], "vertex {v} matched twice");
                matched[ui] = true;
                matched[vi] = true;
                assert!(csr.has_entry(u, v));
            }

            // Maximality: no edge may connect two unmatched vertices.
            for u in csr.row_indices() {
                if matched[usize::from(u)] {
                    continue;
                }
                for w in csr.sparse_row(u) {
                    if w != u && !matched[usize::from(w)] {
                        panic!("edge ({u}, {w}) has both endpoints unmatched");
                    }
                }
            }
        });
    }
}
