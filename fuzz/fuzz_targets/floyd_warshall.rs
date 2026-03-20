//! Fuzzing submodule for the Floyd-Warshall algorithm.

use geometric_traits::{
    impls::ValuedCSR2D,
    test_utils::check_floyd_warshall_invariants,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: ValuedCSR2D<u16, u8, u8, f64>| {
            check_floyd_warshall_invariants(&csr);
        });
    }
}
