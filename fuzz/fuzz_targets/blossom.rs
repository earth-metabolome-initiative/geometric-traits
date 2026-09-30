//! Submodule for fuzzing the Edmonds blossom algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::prelude::*;
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<SquareCSR2D<CSR2D<u16, u8, u8>>>() else {
        return;
    };
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
                panic!("symmetric edge ({u}, {w}) has both endpoints unmatched");
            }
        }
    }
});
