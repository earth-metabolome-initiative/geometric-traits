//! Submodule for fuzzing the execution of Kahn's algorithm.

use geometric_traits::{prelude::*, test_utils::check_kahn_ordering};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|matrix: SquareCSR2D<CSR2D<u16, u8, u8>>| {
            check_kahn_ordering(&matrix, 5);
        });
    }
}
