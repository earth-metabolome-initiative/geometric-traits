//! Submodule for fuzzing exact Karp-Sipser preprocessing.

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    test_utils::check_karp_sipser_invariants,
    traits::SquareMatrix,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SymmetricCSR2D<CSR2D<u16, u8, u8>>| {
            let n = csr.order() as usize;
            if n > 64 {
                return;
            }
            check_karp_sipser_invariants(&csr);
        });
    }
}
