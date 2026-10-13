//! Fuzzing exact VF2 agreement against a small brute-force oracle.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::test_utils::{check_vf2_invariants_fuzz, FuzzVf2Case};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(case) = Unstructured::new(bytes).arbitrary::<FuzzVf2Case>() else {
        return;
    };
    check_vf2_invariants_fuzz(&case);
});
