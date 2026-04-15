//! Fuzzing submodule for exact undirected diameter computation.

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    naive_structs::GenericGraph,
    test_utils::check_diameter_invariants,
    traits::SquareMatrix,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: SymmetricCSR2D<CSR2D<u16, u8, u8>>| {
            let graph: GenericGraph<u8, _> = GenericGraph::from((csr.order(), csr));
            check_diameter_invariants(&graph);
        });
    }
}
