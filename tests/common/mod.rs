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

#[allow(clippy::too_many_arguments)]
pub fn assert_reference_corpus_contract<S, C>(
    relative_path: &str,
    expected_case_count: usize,
    load_fixture_suite: impl FnOnce(&str) -> S,
    schema_version_of: impl Fn(&S) -> u32,
    graph_kind_of: impl Fn(&S) -> &str,
    primary_oracle_of: impl Fn(&S) -> &str,
    cases_of: impl Fn(&S) -> &[C],
    family_of: impl Fn(&C) -> &str,
    expected_graph_kind: &str,
    expected_primary_oracle: &str,
    expected_families: &[&str],
) -> S {
    let path = fixture_path(relative_path);
    assert!(path.exists(), "reference corpus fixture missing at {}", path.display());

    let suite = load_fixture_suite(relative_path);
    assert_eq!(schema_version_of(&suite), 1);
    assert_eq!(graph_kind_of(&suite), expected_graph_kind);
    assert_eq!(primary_oracle_of(&suite), expected_primary_oracle);

    let cases = cases_of(&suite);
    assert_eq!(cases.len(), expected_case_count);

    let observed_families: std::collections::BTreeSet<&str> = cases.iter().map(family_of).collect();
    for expected_family in expected_families {
        assert!(
            observed_families.contains(expected_family),
            "reference corpus must contain at least one {expected_family} case"
        );
    }

    suite
}

pub fn assert_reference_corpus_family_sequence(
    observed_sequence: &[String],
    expected_sequence: &[&str],
) {
    let observed_sequence: Vec<&str> = observed_sequence.iter().map(String::as_str).collect();
    assert_eq!(observed_sequence, expected_sequence);
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
