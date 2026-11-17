//! Fuzz harness for directed Leiden restricted to acyclic contractions.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{impls::ValuedCSR2D, test_utils::check_acyclic_directed_leiden_invariants};
use libfuzzer_sys::fuzz_target;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<Csr>() else {
        return;
    };
    check_acyclic_directed_leiden_invariants(&csr);
});
