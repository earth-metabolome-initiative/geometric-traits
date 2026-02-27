//! Fuzzing submodule on the `ValuedCSR2D` struct.

use geometric_traits::{prelude::*, test_utils::check_valued_matrix_invariants};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: ValuedCSR2D<u16, u8, u8, f64>| {
            check_valued_matrix_invariants(&csr);
        });
    }
}
