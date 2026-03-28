#![cfg(feature = "std")]
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, VecMatrix2D},
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

/// Flatten a row-major nested dense matrix fixture into a single buffer.
pub fn flatten_dense_rows(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Return the `L1` stationary residual `||πP - π||₁` for a dense matrix.
pub fn dense_gth_residual_l1(matrix: &VecMatrix2D<f64>, stationary: &[f64]) -> f64 {
    let n = matrix.number_of_rows();
    let mut total = 0.0;
    for column in 0..n {
        let mut projected = 0.0;
        for (row, value) in stationary.iter().enumerate().take(n) {
            projected += *value * matrix.value((row, column));
        }
        total += (projected - stationary[column]).abs();
    }
    total
}
