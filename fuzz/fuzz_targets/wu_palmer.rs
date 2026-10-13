#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    prelude::{GenericGraph, SquareCSR2D, WuPalmer, CSR2D},
    test_utils::check_similarity_invariants,
    traits::MonopartiteGraph,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) =
        Unstructured::new(bytes).arbitrary::<GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>>()
    else {
        return;
    };
    let Ok(wu_palmer) = csr.wu_palmer() else {
        return;
    };
    let node_ids: Vec<u8> = csr.node_ids().collect();
    check_similarity_invariants(&wu_palmer, &node_ids, 10);
});
