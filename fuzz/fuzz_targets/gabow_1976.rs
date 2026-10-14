//! Submodule for fuzzing Gabow's 1976 maximum matching algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    impls::{SymmetricCSR2D, CSR2D},
    test_utils::check_gabow_1976_invariants,
    traits::SquareMatrix,
};
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
    check_gabow_1976_invariants(&csr);
});
