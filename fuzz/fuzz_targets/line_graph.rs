//! Fuzz harness for the line graph algorithms (undirected and directed).

use geometric_traits::{
    prelude::{GenericGraph, SquareCSR2D, CSR2D},
    test_utils::check_line_graph_invariants,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|graph: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            check_line_graph_invariants(&graph, 32);
        });
    }
}
