use geometric_traits::prelude::{SquareCSR2D, CSR2D};
use geometric_traits::prelude::{GenericGraph, SinkNodes};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|csr: GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>| {
            let _sink_nodes = csr.sink_nodes();
        });
    }
}
