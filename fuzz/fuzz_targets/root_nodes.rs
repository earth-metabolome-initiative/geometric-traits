//! Submodule for fuzzing the execution of the RootNodes algorithm.

use geometric_traits::prelude::{GenericGraph, RootNodes, SquareCSR2D, CSR2D};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let _root_nodes = csr.root_nodes();
        });
    }
}
