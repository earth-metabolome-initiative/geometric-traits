use geometric_traits::{
    prelude::{GenericGraph, Lin, SquareCSR2D, CSR2D},
    test_utils::check_similarity_invariants,
    traits::MonopartiteGraph,
};
use honggfuzz::fuzz;

type LinInput = (Vec<usize>, GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>);

fn main() {
    loop {
        fuzz!(|occurrences_csr: LinInput| {
            let (occurrences, csr) = occurrences_csr;
            let Ok(lin) = csr.lin(occurrences.as_ref()) else {
                return;
            };
            let node_ids: Vec<u8> = csr.node_ids().collect();
            check_similarity_invariants(&lin, &node_ids, 10);
        });
    }
}
