//! Fuzzing submodule on the `PaddedMatrix2D` struct.

use geometric_traits::{prelude::*, test_utils::check_padded_matrix2d_invariants};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: ValuedCSR2D<u16, u8, u8, u8>| {
            check_padded_matrix2d_invariants(&csr);
        });
    }
}
