//! Fuzz harness for the dense GTH stationary-distribution solver.

use geometric_traits::{impls::VecMatrix2D, test_utils::check_gth_invariants};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|matrix: VecMatrix2D<f64>| {
            check_gth_invariants(&matrix);
        });
    }
}
