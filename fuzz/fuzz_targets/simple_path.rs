use geometric_traits::prelude::{SquareCSR2D, CSR2D};
use geometric_traits::prelude::{GenericGraph, SimplePath};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let _simple_path = csr.is_simple_path();
        });
    }
}
