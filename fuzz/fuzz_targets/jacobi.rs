//! Fuzz harness for the Jacobi eigenvalue decomposition.

use geometric_traits::{impls::ValuedCSR2D, test_utils::check_jacobi_invariants};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_jacobi_invariants(&csr);
        });
    }
}
