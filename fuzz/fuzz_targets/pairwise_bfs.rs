//! Fuzzing submodule for PairwiseBFS against unit-weight Floyd-Warshall.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    impls::{SquareCSR2D, CSR2D},
    test_utils::check_pairwise_bfs_matches_unit_floyd_warshall,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<SquareCSR2D<CSR2D<u16, u8, u8>>>() else {
        return;
    };
    check_pairwise_bfs_matches_unit_floyd_warshall(&csr);
});
