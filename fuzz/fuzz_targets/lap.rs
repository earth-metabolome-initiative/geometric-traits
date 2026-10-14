//! Unified fuzz harness for LAP wrappers and LAPMOD core.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    impls::ValuedCSR2D,
    test_utils::{check_lap_sparse_wrapper_invariants, check_lap_square_invariants},
};
use libfuzzer_sys::fuzz_target;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<Csr>() else {
        return;
    };
    check_lap_sparse_wrapper_invariants(&csr);
    check_lap_square_invariants(&csr);
});
