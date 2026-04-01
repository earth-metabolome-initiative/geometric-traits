//! Fuzzing min-cost perfect matching invariants for Blossom V on structured graph families.

use geometric_traits::test_utils::{
    FuzzStructuredBlossomVCase, check_structured_blossom_v_invariants,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|case: FuzzStructuredBlossomVCase| {
            check_structured_blossom_v_invariants(&case);
        });
    }
}
