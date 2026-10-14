//! Fuzzing min-cost perfect matching invariants for Blossom V on structured
//! graph families.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::test_utils::{
    check_structured_blossom_v_invariants, FuzzStructuredBlossomVCase,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(case) = Unstructured::new(bytes).arbitrary::<FuzzStructuredBlossomVCase>() else {
        return;
    };
    check_structured_blossom_v_invariants(&case);
});
