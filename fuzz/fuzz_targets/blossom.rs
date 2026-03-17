//! Submodule for fuzzing the Edmonds blossom algorithm.

use geometric_traits::prelude::*;
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SquareCSR2D<CSR2D<u16, u8, u8>>| {
            let n = csr.order() as usize;
            if n > 128 {
                return;
            }
            let matching = csr.blossom();
            assert!(matching.len() <= n / 2);

            let mut matched = vec![false; n];
            for &(u, v) in &matching {
                assert!(u < v);
                assert!(!matched[u as usize]);
                assert!(!matched[v as usize]);
                matched[u as usize] = true;
                matched[v as usize] = true;
                assert!(csr.has_entry(u, v) || csr.has_entry(v, u));
            }

            // Maximality: no symmetric edge may connect two unmatched vertices.
            // A maximum matching is always maximal, so this must hold.
            for u in csr.row_indices() {
                if matched[u as usize] {
                    continue;
                }
                for w in csr.sparse_row(u) {
                    if w != u && !matched[w as usize] && csr.has_entry(w, u) {
                        panic!(
                            "symmetric edge ({u}, {w}) has both endpoints unmatched"
                        );
                    }
                }
            }
        });
    }
}
