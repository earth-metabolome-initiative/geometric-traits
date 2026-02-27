use geometric_traits::{
    prelude::{GenericGraph, SquareCSR2D, WuPalmer, CSR2D},
    test_utils::check_similarity_invariants,
    traits::MonopartiteGraph,
};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let Ok(wu_palmer) = csr.wu_palmer() else {
                return;
            };
            let node_ids: Vec<u8> = csr.node_ids().collect();
            check_similarity_invariants(&wu_palmer, &node_ids, 10);
        });
    }
}
