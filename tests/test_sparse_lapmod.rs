//! Unit tests for the SparseLAPMOD wrapper.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        LAPError, MatrixMut, SparseLAPJV, SparseLAPMOD, SparseMatrix2D, SparseMatrixMut,
        SparseValuedMatrix2D,
    },
};

fn sorted(mut assignment: Vec<(u8, u8)>) -> Vec<(u8, u8)> {
    assignment.sort_unstable_by_key(|&(row, column)| (row, column));
    assignment
}

fn edge_cost(csr: &ValuedCSR2D<u8, u8, u8, f64>, row: u8, column: u8) -> Option<f64> {
    csr.sparse_row(row)
        .zip(csr.sparse_row_values(row))
        .find_map(|(candidate_column, value)| (candidate_column == column).then_some(value))
}

fn assignment_cost(csr: &ValuedCSR2D<u8, u8, u8, f64>, assignment: &[(u8, u8)]) -> f64 {
    assignment
        .iter()
        .map(|&(row, column)| {
            edge_cost(csr, row, column).unwrap_or_else(|| {
                panic!("Assignment contains an imputed/non-existing edge: ({row}, {column})")
            })
        })
        .sum()
}

#[test]
fn test_sparse_lapmod_zero_columns() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((10, 0), 0);
    let assignment = csr.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_sparse_lapmod_zero_rows() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 10), 0);
    let assignment = csr.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_sparse_lapmod_wide_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0, 20.0],
        [0.5, 10.0, 20.0, 20.0],
        [10.0, 20.0, 0.5, 10.0],
    ])
    .expect("Failed to create CSR matrix");

    let sparse_lapmod = sorted(csr.sparse_lapmod(900.0, 1000.0).expect("SparseLAPMOD failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(sparse_lapmod, sparse_lapjv);
    assert_eq!(sparse_lapmod, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_sparse_lapmod_tall_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0],
        [0.5, 10.0, 20.0],
        [10.0, 20.0, 0.5],
        [10.0, 20.0, 0.1],
    ])
    .expect("Failed to create CSR matrix");

    let sparse_lapmod = sorted(csr.sparse_lapmod(900.0, 1000.0).expect("SparseLAPMOD failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(sparse_lapmod, sparse_lapjv);
    assert_eq!(sparse_lapmod, vec![(0, 1), (1, 0), (3, 2)]);
}

#[test]
fn test_sparse_lapmod_partial_assignment_rectangular() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 2), 3);
    csr.add((0, 0, 1.0)).expect("Failed to add value");
    csr.add((1, 0, 2.0)).expect("Failed to add value");
    csr.add((2, 1, 1.0)).expect("Failed to add value");

    let sparse_lapmod = sorted(csr.sparse_lapmod(900.0, 1000.0).expect("SparseLAPMOD failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(sparse_lapmod, sparse_lapjv);
    assert_eq!(sparse_lapmod, vec![(0, 0), (2, 1)]);
}

#[test]
fn test_sparse_lapmod_parameter_validation() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 0);

    assert_eq!(csr.sparse_lapmod(f64::INFINITY, 1000.0), Err(LAPError::PaddingValueNotFinite));
    assert_eq!(csr.sparse_lapmod(-1.0, 1000.0), Err(LAPError::PaddingValueNotPositive));
    assert_eq!(csr.sparse_lapmod(1000.0, 1000.0), Err(LAPError::ValueTooLarge));

    let mut non_empty: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 1);
    non_empty.add((0, 0, 1.0)).expect("Failed to add value");
    assert_eq!(non_empty.sparse_lapmod(900.0, f64::NAN), Err(LAPError::MaximalCostNotFinite));
}

#[test]
fn test_sparse_lapmod_cost_matches_sparse_lapjv_with_ties() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 1.0, 5.0], [1.0, 1.0, 5.0], [5.0, 5.0, 1.0]])
            .expect("Failed to create CSR matrix");

    let sparse_lapmod = csr.sparse_lapmod(900.0, 1000.0).expect("SparseLAPMOD failed");
    let sparse_lapjv = csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed");

    assert_eq!(sparse_lapmod.len(), sparse_lapjv.len());

    let sparse_lapmod_cost = assignment_cost(&csr, &sparse_lapmod);
    let sparse_lapjv_cost = assignment_cost(&csr, &sparse_lapjv);
    assert!(
        (sparse_lapmod_cost - sparse_lapjv_cost).abs() < 1e-9,
        "SparseLAPMOD and SparseLAPJV costs differ: {sparse_lapmod_cost} vs {sparse_lapjv_cost}"
    );
}
