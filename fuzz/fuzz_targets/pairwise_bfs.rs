//! Fuzzing submodule for PairwiseBFS against unit-weight Floyd-Warshall.

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    test_utils::check_pairwise_bfs_matches_unit_floyd_warshall,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SquareCSR2D<CSR2D<u16, u8, u8>>| {
            check_pairwise_bfs_matches_unit_floyd_warshall(&csr);
        });
    }
}
