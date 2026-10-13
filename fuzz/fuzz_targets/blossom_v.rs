//! Fuzzing min-cost perfect matching invariants for Blossom V.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::test_utils::{check_blossom_v_invariants_fuzz, FuzzBlossomVCase};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(case) = Unstructured::new(bytes).arbitrary::<FuzzBlossomVCase>() else {
        return;
    };
    check_blossom_v_invariants_fuzz(&case);
});
