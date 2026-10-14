//! Fuzz harness for the ForceAtlas2 layout.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{impls::ValuedCSR2D, test_utils::check_forceatlas2_invariants};
use libfuzzer_sys::fuzz_target;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(input) = Unstructured::new(bytes).arbitrary::<(Csr, u8)>() else {
        return;
    };
    let (csr, mode_bits) = input;
    check_forceatlas2_invariants(&csr, mode_bits);
});
