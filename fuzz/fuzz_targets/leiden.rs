//! Fuzz harness for the Leiden community detection algorithm.

use geometric_traits::{impls::ValuedCSR2D, test_utils::check_leiden_invariants};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_leiden_invariants(&csr);
        });
    }
}
