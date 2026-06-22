//! Fuzz harness for the ForceAtlas2 layout.

use geometric_traits::{impls::ValuedCSR2D, test_utils::check_forceatlas2_invariants};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|input: (Csr, u8)| {
            let (csr, mode_bits) = input;
            check_forceatlas2_invariants(&csr, mode_bits);
        });
    }
}
