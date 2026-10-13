//! Submodule for fuzzing the execution of Kahn's algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{prelude::*, test_utils::check_kahn_ordering};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(matrix) = Unstructured::new(bytes).arbitrary::<SquareCSR2D<CSR2D<u16, u8, u8>>>() else {
        return;
    };
    check_kahn_ordering(&matrix, 5);
});
