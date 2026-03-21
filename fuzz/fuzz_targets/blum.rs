//! Submodule for fuzzing Blum's maximum matching algorithm.

use geometric_traits::prelude::*;
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SymmetricCSR2D<CSR2D<u16, u8, u8>>| {
            let n = csr.order() as usize;
            if n > 128 {
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
                assert!(u < v);
                assert!(!matched[u as usize]);
                assert!(!matched[v as usize]);
                matched[u as usize] = true;
                matched[v as usize] = true;
                assert!(csr.has_entry(u, v));
            }

            // Maximality: no edge may connect two unmatched vertices.
            for u in csr.row_indices() {
                if matched[u as usize] {
                    continue;
                }
                for w in csr.sparse_row(u) {
                    if w != u && !matched[w as usize] {
                        panic!("edge ({u}, {w}) has both endpoints unmatched");
                    }
                }
            }
        });
    }
}
