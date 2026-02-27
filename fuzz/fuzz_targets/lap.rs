//! Unified fuzz harness for LAP wrappers and LAPMOD core.

use geometric_traits::{
    impls::ValuedCSR2D,
    test_utils::{check_lap_sparse_wrapper_invariants, check_lap_square_invariants},
};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_lap_sparse_wrapper_invariants(&csr);
            check_lap_square_invariants(&csr);
        });
    }
}
