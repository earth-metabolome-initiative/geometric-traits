//! Fuzz harness for the line graph algorithms (undirected and directed).

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::{
    prelude::{GenericGraph, SquareCSR2D, CSR2D},
    test_utils::check_line_graph_invariants,
};
use libfuzzer_sys::fuzz_target;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(graph) =
        Unstructured::new(bytes).arbitrary::<GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>>()
    else {
        return;
    };
    check_line_graph_invariants(&graph, 32);
});
