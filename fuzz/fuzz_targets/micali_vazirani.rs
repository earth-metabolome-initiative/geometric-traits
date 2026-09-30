//! Submodule for fuzzing the Micali-Vazirani maximum matching algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::prelude::*;
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<SymmetricCSR2D<CSR2D<u16, u8, u8>>>() else {
        return;
    };
    let n = csr.order() as usize;
    if n > 128 {
        return;
    }
    let mv_matching = csr.micali_vazirani().unwrap();
    let bl_matching = csr.blossom();

    assert_eq!(
        mv_matching.len(),
        bl_matching.len(),
        "MV and Blossom disagree on matching size (n={n})"
    );

    assert!(mv_matching.len() <= n / 2);

    let mut matched = vec![false; n];
    for &(u, v) in &mv_matching {
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
