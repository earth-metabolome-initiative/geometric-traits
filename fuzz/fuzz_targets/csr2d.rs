//! Fuzzing submodule on the `CSR2D` struct.

use geometric_traits::{prelude::*, test_utils::check_sparse_matrix_invariants};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: CSR2D<u16, u8, u8>| {
            check_sparse_matrix_invariants(&csr);
        });
    }
}
