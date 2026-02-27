//! Fuzzing submodule on the `GenericMatrix2DWithPaddedDiagonal` struct.

use geometric_traits::test_utils::{FuzzPaddedDiag, check_padded_diagonal_invariants};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|padded_csr: FuzzPaddedDiag| {
            check_padded_diagonal_invariants(&padded_csr);
        });
    }
}
