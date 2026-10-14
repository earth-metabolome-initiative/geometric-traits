//! Fuzzing submodule for PairwiseDijkstra against Floyd-Warshall.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    impls::ValuedCSR2D, test_utils::check_pairwise_dijkstra_matches_floyd_warshall,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<ValuedCSR2D<u16, u8, u8, f64>>() else {
        return;
    };
    check_pairwise_dijkstra_matches_floyd_warshall(&csr);
});
