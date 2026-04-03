//! Fuzzing exact VF2 agreement against a small brute-force oracle.

use geometric_traits::test_utils::{FuzzVf2Case, check_vf2_invariants_fuzz};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|case: FuzzVf2Case| {
            check_vf2_invariants_fuzz(&case);
        });
    }
}
