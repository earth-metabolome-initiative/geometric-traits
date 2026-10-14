//! Fuzzing submodule for the Floyd-Warshall algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{impls::ValuedCSR2D, test_utils::check_floyd_warshall_invariants};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<ValuedCSR2D<u16, u8, u8, f64>>() else {
        return;
    };
    check_floyd_warshall_invariants(&csr);
});
