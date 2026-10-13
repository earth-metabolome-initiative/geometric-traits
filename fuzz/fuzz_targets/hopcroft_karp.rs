//! Submodule for fuzzing the execution of the Hopcroft-Karp algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::prelude::{HopcroftKarp, CSR2D};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<CSR2D<u16, u8, u8>>() else {
        return;
    };
    let _ = csr.hopcroft_karp();
});
