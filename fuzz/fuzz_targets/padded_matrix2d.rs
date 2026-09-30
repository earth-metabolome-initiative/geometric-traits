//! Fuzzing submodule on the `PaddedMatrix2D` struct.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{prelude::*, test_utils::check_padded_matrix2d_invariants};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<ValuedCSR2D<u16, u8, u8, u8>>() else {
        return;
    };
    check_padded_matrix2d_invariants(&csr);
});
