//! Fuzzing min-cost perfect matching invariants for Blossom V.

use geometric_traits::test_utils::{FuzzBlossomVCase, check_blossom_v_invariants_fuzz};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|case: FuzzBlossomVCase| {
            check_blossom_v_invariants_fuzz(&case);
        });
    }
}
