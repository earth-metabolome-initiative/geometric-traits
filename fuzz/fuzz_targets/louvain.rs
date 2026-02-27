//! Fuzz harness for the Louvain community detection algorithm.

use geometric_traits::{impls::ValuedCSR2D, test_utils::check_louvain_invariants};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_louvain_invariants(&csr);
        });
    }
}
