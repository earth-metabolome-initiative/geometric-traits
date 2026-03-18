//! Fuzzing submodule on the `GenericMatrix2DWithPaddedDiagonal` struct.

use geometric_traits::test_utils::{check_padded_diagonal_invariants, FuzzPaddedDiag};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|padded_csr: FuzzPaddedDiag| {
            check_padded_diagonal_invariants(&padded_csr);
        });
    }
}
