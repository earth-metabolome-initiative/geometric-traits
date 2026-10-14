//! Internal helper for building undirected graphs from edge lists.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use crate::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D},
    naive_structs::GenericUndirectedMonopartiteEdgesBuilder,
    traits::EdgesBuilder,
};

/// Type alias for the undirected edges builder used internally.
type InternalUndiBuilder<I> = GenericUndirectedMonopartiteEdgesBuilder<
    I,
    UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    SymmetricCSR2D<CSR2D<usize, usize, usize>>,
>;

/// Builds a `SymmetricCSR2D` from a sorted, deduplicated, upper-triangular edge
/// list.
///
/// # Arguments
/// * `n` — number of vertices
/// * `edges` — sorted edges with `u < v` for each `(u, v)`, no duplicates
///
/// # Panics
/// Panics if the edge list violates ordering or contains out-of-range vertices.
#[inline]
pub(crate) fn build_symmetric(
    n: usize,
    edges: Vec<(usize, usize)>,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let num_edges = edges.len();
    InternalUndiBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

/// Type alias for the directed edges builder used internally.
type InternalDiBuilder<I> =
    crate::naive_structs::GenericEdgesBuilder<I, SquareCSR2D<CSR2D<usize, usize, usize>>>;

/// Builds a `SquareCSR2D` from a sorted, deduplicated edge list.
///
/// # Arguments
/// * `n` — number of vertices
/// * `edges` — sorted edges, no duplicates
///
/// # Panics
/// Panics if the edge list violates ordering or contains out-of-range vertices.
#[inline]
pub(crate) fn build_directed(
    n: usize,
    edges: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    let num_edges = edges.len();
    InternalDiBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}
