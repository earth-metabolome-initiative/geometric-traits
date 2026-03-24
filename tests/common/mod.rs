#![cfg(feature = "std")]
#![allow(dead_code)]

use std::path::{Path, PathBuf};

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

/// Return an absolute path under `tests/fixtures`.
pub fn fixture_path(relative_path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative_path)
}

/// Read a binary fixture under `tests/fixtures`.
pub fn read_fixture(relative_path: &str) -> Vec<u8> {
    let path = fixture_path(relative_path);
    std::fs::read(&path).unwrap_or_else(|_| panic!("failed to read fixture {}", path.display()))
}

/// Read a UTF-8 text fixture under `tests/fixtures`.
pub fn read_fixture_string(relative_path: &str) -> String {
    let path = fixture_path(relative_path);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("failed to read fixture {}", path.display()))
}
