//! Fuzz harness for the dense GTH stationary-distribution solver.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{impls::VecMatrix2D, test_utils::check_gth_invariants};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(matrix) = Unstructured::new(bytes).arbitrary::<VecMatrix2D<f64>>() else {
        return;
    };
    check_gth_invariants(&matrix);
});
