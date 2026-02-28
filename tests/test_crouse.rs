//! Integration tests for the Crouse trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{MatrixMut, SparseLAPJV, SparseMatrixMut, Crouse},
};

fn sorted(mut assignment: Vec<(u8, u8)>) -> Vec<(u8, u8)> {
    assignment.sort_unstable_by_key(|&(r, c)| (r, c));
    assignment
}

fn edge_cost(csr: &ValuedCSR2D<u8, u8, u8, f64>, row: u8, column: u8) -> Option<f64> {
    use geometric_traits::prelude::{SparseMatrix2D, SparseValuedMatrix2D};
    csr.sparse_row(row)
        .zip(csr.sparse_row_values(row))
        .find_map(|(c, v)| (c == column).then_some(v))
}

fn assignment_cost(csr: &ValuedCSR2D<u8, u8, u8, f64>, assignment: &[(u8, u8)]) -> f64 {
    assignment
        .iter()
        .map(|&(r, c)| edge_cost(csr, r, c).unwrap())
        .sum()
}

#[test]
fn test_wide_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0, 20.0],
        [0.5, 10.0, 20.0, 20.0],
        [10.0, 20.0, 0.5, 10.0],
    ])
    .expect("Failed to create CSR matrix");

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(result, sparse_lapjv);
    assert_eq!(result, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_tall_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0],
        [0.5, 10.0, 20.0],
        [10.0, 20.0, 0.5],
        [10.0, 20.0, 0.1],
    ])
    .expect("Failed to create CSR matrix");

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(result, sparse_lapjv);
    assert_eq!(result, vec![(0, 1), (1, 0), (3, 2)]);
}

#[test]
fn test_truly_sparse_rectangular() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((4, 3), 6);
    csr.add((0, 0, 0.1)).unwrap();
    csr.add((0, 1, 0.9)).unwrap();
    csr.add((1, 0, 0.8)).unwrap();
    csr.add((1, 2, 0.3)).unwrap();
    csr.add((2, 1, 0.2)).unwrap();
    csr.add((3, 2, 0.4)).unwrap();

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(
        assignment_cost(&csr, &result),
        assignment_cost(&csr, &sparse_lapjv),
    );
    assert_eq!(result, vec![(0, 0), (1, 2), (2, 1)]);
}

#[test]
fn test_rows_with_no_entries() {
    // Rows 1 and 3 have no edges.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((4, 3), 3);
    csr.add((0, 0, 0.5)).unwrap();
    csr.add((0, 1, 0.3)).unwrap();
    csr.add((2, 2, 0.1)).unwrap();

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(
        assignment_cost(&csr, &result),
        assignment_cost(&csr, &sparse_lapjv),
    );
    assert_eq!(result, vec![(0, 1), (2, 2)]);
}

#[test]
fn test_single_entry() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 1);
    csr.add((0, 0, 0.5)).unwrap();

    let result = csr
        .crouse(900.0, 1000.0)
        .expect("Crouse failed");
    assert_eq!(result, vec![(0, 0)]);
}

#[test]
fn test_no_matching_preferred() {
    // Row 1's only edge is very expensive (100.0) — it should remain unmatched,
    // and row 0 takes col 0 instead.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((2, 1), 2);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((1, 0, 100.0)).unwrap();

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    // Only row 0 should be matched.
    assert_eq!(result, vec![(0, 0)]);
}

#[test]
fn test_empty() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10, 0), 0);
    let result = csr
        .crouse(900.0, 1000.0)
        .unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_square_sparse() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 5);
    csr.add((0, 0, 0.1)).unwrap();
    csr.add((0, 2, 0.7)).unwrap();
    csr.add((1, 1, 0.2)).unwrap();
    csr.add((2, 0, 0.5)).unwrap();
    csr.add((2, 2, 0.3)).unwrap();

    let result = sorted(
        csr.crouse(900.0, 1000.0)
            .expect("Crouse failed"),
    );
    let sparse_lapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(
        assignment_cost(&csr, &result),
        assignment_cost(&csr, &sparse_lapjv),
    );
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 2)]);
}
