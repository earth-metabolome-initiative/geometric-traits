//! Unit tests for the Jaqaman wrapper.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        Jaqaman, LAPError, MatrixMut, SparseLAPJV, SparseMatrix2D, SparseMatrixMut,
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

fn assert_costs_close(lhs: f64, rhs: f64) {
    let denom = lhs.abs().max(rhs.abs()).max(1e-12);
    assert!(
        (lhs - rhs).abs() / denom < 1e-12,
        "costs differ: {lhs} vs {rhs}"
    );
}

#[test]
fn test_jaqaman_zero_columns() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((10, 0), 0);
    let assignment = csr.jaqaman(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_jaqaman_zero_rows() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 10), 0);
    let assignment = csr.jaqaman(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_jaqaman_wide_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0, 20.0],
        [0.5, 10.0, 20.0, 20.0],
        [10.0, 20.0, 0.5, 10.0],
    ])
    .expect("Failed to create CSR matrix");

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(jaqaman, sparse_lapjv);
    assert_eq!(jaqaman, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_jaqaman_tall_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0],
        [0.5, 10.0, 20.0],
        [10.0, 20.0, 0.5],
        [10.0, 20.0, 0.1],
    ])
    .expect("Failed to create CSR matrix");

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(jaqaman, sparse_lapjv);
    assert_eq!(jaqaman, vec![(0, 1), (1, 0), (3, 2)]);
}

#[test]
fn test_jaqaman_partial_assignment_rectangular() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 2), 3);
    csr.add((0, 0, 1.0)).expect("Failed to add value");
    csr.add((1, 0, 2.0)).expect("Failed to add value");
    csr.add((2, 1, 1.0)).expect("Failed to add value");

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(jaqaman, sparse_lapjv);
    assert_eq!(jaqaman, vec![(0, 0), (2, 1)]);
}

#[test]
fn test_jaqaman_parameter_validation() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 0);

    assert_eq!(csr.jaqaman(f64::INFINITY, 1000.0), Err(LAPError::PaddingValueNotFinite));
    assert_eq!(csr.jaqaman(-1.0, 1000.0), Err(LAPError::PaddingValueNotPositive));
    assert_eq!(csr.jaqaman(1000.0, 1000.0), Err(LAPError::ValueTooLarge));

    let mut non_empty: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 1);
    non_empty.add((0, 0, 1.0)).expect("Failed to add value");
    assert_eq!(non_empty.jaqaman(900.0, f64::NAN), Err(LAPError::MaximalCostNotFinite));
}

#[test]
fn test_jaqaman_cost_matches_sparse_lapjv_with_ties() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 1.0, 5.0], [1.0, 1.0, 5.0], [5.0, 5.0, 1.0]])
            .expect("Failed to create CSR matrix");

    let jaqaman = csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed");
    let sparse_lapjv = csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed");

    assert_eq!(jaqaman.len(), sparse_lapjv.len());

    let jaqaman_cost = assignment_cost(&csr, &jaqaman);
    let sparse_lapjv_cost = assignment_cost(&csr, &sparse_lapjv);
    assert!(
        (jaqaman_cost - sparse_lapjv_cost).abs() < 1e-9,
        "Jaqaman and SparseLAPJV costs differ: {jaqaman_cost} vs {sparse_lapjv_cost}"
    );
}

#[test]
fn test_jaqaman_truly_sparse_rectangular() {
    // Truly sparse matrix (not all entries present).
    // 4 rows × 3 columns, only 6 of 12 possible entries.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((4, 3), 6);
    csr.add((0, 0, 0.1)).unwrap();
    csr.add((0, 1, 0.9)).unwrap();
    csr.add((1, 0, 0.8)).unwrap();
    csr.add((1, 2, 0.3)).unwrap();
    csr.add((2, 1, 0.2)).unwrap();
    csr.add((3, 2, 0.4)).unwrap();

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_costs_close(assignment_cost(&csr, &jaqaman), assignment_cost(&csr, &sparse_lapjv));
    // Optimal: row 0→col 0 (0.1), row 1→col 2 (0.3), row 2→col 1 (0.2).
    assert_eq!(jaqaman, vec![(0, 0), (1, 2), (2, 1)]);
}

#[test]
fn test_jaqaman_rows_with_no_sparse_entries() {
    // Matrix with rows that have no edges (rows 1 and 3 are empty).
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((4, 3), 3);
    csr.add((0, 0, 0.5)).unwrap();
    csr.add((0, 1, 0.3)).unwrap();
    csr.add((2, 2, 0.1)).unwrap();

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_costs_close(assignment_cost(&csr, &jaqaman), assignment_cost(&csr, &sparse_lapjv));
    // Only rows 0 and 2 can match. Row 0→col 1 (0.3), row 2→col 2 (0.1).
    assert_eq!(jaqaman, vec![(0, 1), (2, 2)]);
}

#[test]
fn test_jaqaman_square_sparse() {
    // Square sparse matrix (L = R = 3).
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 5);
    csr.add((0, 0, 0.1)).unwrap();
    csr.add((0, 2, 0.7)).unwrap();
    csr.add((1, 1, 0.2)).unwrap();
    csr.add((2, 0, 0.5)).unwrap();
    csr.add((2, 2, 0.3)).unwrap();

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_costs_close(assignment_cost(&csr, &jaqaman), assignment_cost(&csr, &sparse_lapjv));
    // Optimal: row 0→col 0 (0.1), row 1→col 1 (0.2), row 2→col 2 (0.3).
    assert_eq!(jaqaman, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn test_jaqaman_single_entry() {
    // Minimal case: 1×1 matrix with one entry.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 1);
    csr.add((0, 0, 0.5)).unwrap();

    let jaqaman = csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed");
    assert_eq!(jaqaman, vec![(0, 0)]);
}

#[test]
fn test_jaqaman_no_matching_preferred() {
    // Matrix where unmatching is cheaper than matching for some rows.
    // Row 1 → col 0 has high cost (close to padding), so it should remain
    // unmatched and row 0 takes col 0 instead.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 1), 2);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((1, 0, 100.0)).unwrap();

    let jaqaman = sorted(csr.jaqaman(900.0, 1000.0).expect("Jaqaman failed"));
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_costs_close(assignment_cost(&csr, &jaqaman), assignment_cost(&csr, &sparse_lapjv));
    // Only row 0 should match col 0 (cost 1.0 vs 100.0).
    assert_eq!(jaqaman, vec![(0, 0)]);
}
