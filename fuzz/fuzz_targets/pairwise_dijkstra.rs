//! Fuzzing submodule for PairwiseDijkstra against Floyd-Warshall.

use geometric_traits::{
    impls::ValuedCSR2D, test_utils::check_pairwise_dijkstra_matches_floyd_warshall,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: ValuedCSR2D<u16, u8, u8, f64>| {
            check_pairwise_dijkstra_matches_floyd_warshall(&csr);
        });
    }
}
