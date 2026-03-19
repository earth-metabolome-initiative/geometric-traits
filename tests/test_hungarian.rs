//! Unit tests to verify the correctness of the Hungarian algorithm
//! implementation.
#![cfg(feature = "std")]

use std::{iter::Copied, slice::Iter, vec::Vec};

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        DenseMatrix, DenseMatrix2D, DenseValuedMatrix, DenseValuedMatrix2D, Hungarian, LAPError,
        Matrix, Matrix2D, MatrixMut, SparseHungarian, SparseLAPJV, SparseMatrix2D, SparseMatrixMut,
        SparseValuedMatrix2D, ValuedMatrix, ValuedMatrix2D,
    },
};

#[derive(Clone)]
struct DenseTestMatrix {
    rows: u8,
    columns: u8,
    values: Vec<f64>,
}

impl DenseTestMatrix {
    fn new(rows: u8, columns: u8, values: Vec<f64>) -> Self {
        assert_eq!(usize::from(rows) * usize::from(columns), values.len());
        Self { rows, columns, values }
    }

    fn row_bounds(&self, row: u8) -> (usize, usize) {
        let start = usize::from(row) * usize::from(self.columns);
        let end = start + usize::from(self.columns);
        (start, end)
    }
}

impl Matrix for DenseTestMatrix {
    type Coordinates = (u8, u8);

    fn shape(&self) -> Vec<usize> {
        vec![usize::from(self.rows), usize::from(self.columns)]
    }
}

impl ValuedMatrix for DenseTestMatrix {
    type Value = f64;
}

impl Matrix2D for DenseTestMatrix {
    type RowIndex = u8;
    type ColumnIndex = u8;

    fn number_of_rows(&self) -> Self::RowIndex {
        self.rows
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.columns
    }
}

impl ValuedMatrix2D for DenseTestMatrix {}
impl DenseMatrix for DenseTestMatrix {}
impl DenseMatrix2D for DenseTestMatrix {}

impl DenseValuedMatrix for DenseTestMatrix {
    type Values<'a>
        = Copied<Iter<'a, f64>>
    where
        Self: 'a;

    fn value(&self, coordinates: Self::Coordinates) -> Self::Value {
        let (row, column) = coordinates;
        self.values[usize::from(row) * usize::from(self.columns) + usize::from(column)]
    }

    fn values(&self) -> Self::Values<'_> {
        self.values.iter().copied()
    }
}

impl DenseValuedMatrix2D for DenseTestMatrix {
    type RowValues<'a>
        = Copied<Iter<'a, f64>>
    where
        Self: 'a;

    fn row_values(&self, row: Self::RowIndex) -> Self::RowValues<'_> {
        let (start, end) = self.row_bounds(row);
        self.values[start..end].iter().copied()
    }
}

#[test]
fn test_hungarian_zero_columns() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((10, 0), 0);
    let assignment = csr.sparse_hungarian(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_hungarian_zero_rows() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 10), 0);
    let assignment = csr.sparse_hungarian(900.0, 1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
fn test_dense_hungarian_empty_matrix() {
    let matrix = DenseTestMatrix::new(0, 0, vec![]);
    let assignment = matrix.hungarian(1000.0).expect("Dense Hungarian failed");
    assert!(assignment.is_empty());
}

#[test]
fn test_dense_hungarian_rejects_non_square_matrix() {
    let matrix = DenseTestMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(matrix.hungarian(1000.0), Err(LAPError::NonSquareMatrix));
}

#[test]
fn test_hungarian() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 0.5, 10.0], [0.5, 10.0, 20.0], [10.0, 20.0, 0.5]])
            .expect("Failed to create CSR matrix");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_hungarian_crossing_assignment() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [1.0, 3.0]]).expect("Failed to create CSR matrix");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 1), (1, 0)]);
}

#[test]
fn test_hungarian_wide_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0, 20.0],
        [0.5, 10.0, 20.0, 20.0],
        [10.0, 20.0, 0.5, 10.0],
    ])
    .expect("Failed to create CSR matrix");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_hungarian_tall_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 0.5, 10.0],
        [0.5, 10.0, 20.0],
        [10.0, 20.0, 0.5],
        [10.0, 20.0, 0.1],
    ])
    .expect("Failed to create CSR matrix");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 1), (1, 0), (3, 2)]);
}

#[test]
fn test_hungarian_infinite_loop1() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0, 1.0)).expect("Failed to add value");
    csr.add((2, 2, 800.0)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 0), (2, 2)]);
}

#[test]
fn test_hungarian_infinite_loop2() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((1, 0, 1.0)).expect("Failed to add value");
    csr.add((1, 1, 2.0)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(1, 0)]);
}

#[test]
fn test_hungarian_infinite_loop3() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0, 1.0)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 0)]);
}

#[test]
fn test_hungarian_infinite_loop4() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0, 2e-5)).expect("Failed to add value");
    csr.add((0, 2, 3e-5)).expect("Failed to add value");
    csr.add((2, 0, 4.778_309_726_7e-5)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 2), (2, 0)]);
}

#[test]
fn test_hungarian_inconsistent_unassigned_rows() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0, 2.0)).expect("Failed to add value");
    csr.add((0, 1, 1e-3)).expect("Failed to add value");
    csr.add((1, 1, 1e-2)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(0, 0), (1, 1)]);
}

#[test]
fn test_hungarian_inconsistent_with_hopcroft_karp2() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((5, 5), 1);
    csr.add((3, 3, 0.1)).expect("Failed to add value");
    csr.add((3, 4, 2.0)).expect("Failed to add value");
    csr.add((4, 3, 2.0)).expect("Failed to add value");

    let mut assignment = csr.sparse_hungarian(900.0, 1_000_000.0).expect("Hungarian failed");
    assignment.sort_unstable_by_key(|a| (a.0, a.1));
    assert_eq!(assignment, vec![(3, 4), (4, 3)]);
}

/// Cross-validate Hungarian against LAPJV on several matrices.
#[test]
fn test_hungarian_cross_validation_with_lapjv() {
    let matrices: Vec<ValuedCSR2D<u8, u8, u8, f64>> = vec![
        ValuedCSR2D::try_from([[1.0, 0.5, 10.0], [0.5, 10.0, 20.0], [10.0, 20.0, 0.5]]).unwrap(),
        ValuedCSR2D::try_from([[1.0, 2.0], [1.0, 3.0]]).unwrap(),
        ValuedCSR2D::try_from([[5.0, 9.0, 1.0], [10.0, 3.0, 2.0], [8.0, 7.0, 4.0]]).unwrap(),
        ValuedCSR2D::try_from([[1.0]]).unwrap(),
    ];

    for csr in &matrices {
        let padding = 900.0;
        let max_cost = 1000.0;

        let mut hungarian = csr.sparse_hungarian(padding, max_cost).expect("Hungarian failed");
        let mut lapjv = csr.sparse_lapjv(padding, max_cost).expect("LAPJV failed");

        hungarian.sort_unstable_by_key(|a| (a.0, a.1));
        lapjv.sort_unstable_by_key(|a| (a.0, a.1));

        assert_eq!(
            hungarian.len(),
            lapjv.len(),
            "Cardinality mismatch: hungarian={hungarian:?} lapjv={lapjv:?}"
        );

        let cost_of = |assignment: &[(u8, u8)]| -> f64 {
            assignment
                .iter()
                .map(|&(r, c)| {
                    csr.sparse_row(r)
                        .zip(csr.sparse_row_values(r))
                        .find_map(|(col, v)| (col == c).then_some(v))
                        .unwrap_or(padding)
                })
                .sum()
        };
        let hungarian_cost = cost_of(&hungarian);
        let lapjv_cost = cost_of(&lapjv);

        let denom = hungarian_cost.abs().max(lapjv_cost.abs()).max(1e-30);
        assert!(
            (hungarian_cost - lapjv_cost).abs() / denom < 1e-9,
            "Cost mismatch: hungarian={hungarian_cost} lapjv={lapjv_cost}"
        );
    }
}
