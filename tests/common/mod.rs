#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::*,
    traits::EdgesBuilder,
};

/// Build a square CSR matrix from directed edges for integration tests.
pub fn build_square_csr(
    node_count: usize,
    mut edges: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}
