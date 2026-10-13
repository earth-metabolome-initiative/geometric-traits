//! Fuzzing submodule on the `GenericMatrix2DWithPaddedDiagonal` struct.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::test_utils::{check_padded_diagonal_invariants, FuzzPaddedDiag};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(padded_csr) = Unstructured::new(bytes).arbitrary::<FuzzPaddedDiag>() else {
        return;
    };
    check_padded_diagonal_invariants(&padded_csr);
});
