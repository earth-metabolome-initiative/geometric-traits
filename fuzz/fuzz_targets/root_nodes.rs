//! Submodule for fuzzing the execution of the RootNodes algorithm.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::prelude::{GenericGraph, RootNodes, SquareCSR2D, CSR2D};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) =
        Unstructured::new(bytes).arbitrary::<GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>>()
    else {
        return;
    };
    let _root_nodes = csr.root_nodes();
});
