//! Fuzz harness for classical MDS (Torgerson) embedding.

use geometric_traits::{impls::ValuedCSR2D, test_utils::check_mds_invariants};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_mds_invariants(&csr);
        });
    }
}
