use geometric_traits::prelude::{SquareCSR2D, CSR2D};
use geometric_traits::prelude::{GenericGraph, WuPalmer};
use honggfuzz::fuzz;
use geometric_traits::traits::MonopartiteGraph;
use geometric_traits::traits::ScalarSimilarity;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let Ok(wu_palmer) = csr.wu_palmer() else {
                return ;
            };
            for src in csr.node_ids().take(10) {
                for dst in csr.node_ids(){
                    let similarity = wu_palmer.similarity(&src, &dst);
                    if src == dst {
                        assert!(similarity > 0.99);
                    } else {
                        let symmetric_similarity = wu_palmer.similarity(&dst, &src);
                        assert!({symmetric_similarity - similarity}.abs() < f64::EPSILON, "Expected sim({src}, {dst}) == sim({dst},{src}) got {similarity}!={symmetric_similarity}");
                    }
                    assert!(similarity <= 1.0, "Expected sim({src},{dst}) = {similarity} <= 1");
                    assert!(similarity >= 0.0, "Expected sim({src},{dst}) = {similarity} >= 0");
                }
            }
        });
    }
}
