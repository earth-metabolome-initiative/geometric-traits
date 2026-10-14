//! Fuzzing submodule for exact undirected diameter computation.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    impls::{SymmetricCSR2D, CSR2D},
    naive_structs::GenericGraph,
    test_utils::check_diameter_invariants,
    traits::SquareMatrix,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<SymmetricCSR2D<CSR2D<u16, u8, u8>>>() else {
        return;
    };
    let graph: GenericGraph<u8, _> = GenericGraph::from((csr.order(), csr));
    check_diameter_invariants(&graph);
});
