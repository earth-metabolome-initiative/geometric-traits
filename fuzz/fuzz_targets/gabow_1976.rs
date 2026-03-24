//! Submodule for fuzzing Gabow's 1976 maximum matching algorithm.

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    test_utils::check_gabow_1976_invariants,
    traits::SquareMatrix,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SymmetricCSR2D<CSR2D<u16, u8, u8>>| {
            let n = csr.order() as usize;
            if n > 128 {
                return;
            }
            check_gabow_1976_invariants(&csr);
        });
    }
}
