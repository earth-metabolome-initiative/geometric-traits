use geometric_traits::prelude::{GenericGraph, SimplePath, SquareCSR2D, CSR2D};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let _simple_path = csr.is_simple_path();
        });
    }
}
