//! Regression tests for the valued CSR storage helper APIs.
#![cfg(feature = "std")]

use std::{string::String, vec::Vec};

use geometric_traits::{
    impls::{
        CSR2D, MutabilityError, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D, ValuedCsrPartsError,
    },
    prelude::*,
    traits::{MatrixMut, SparseMatrix},
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestValuedCSR = ValuedCSR2D<usize, usize, usize, i32>;
type TestSquare = SquareCSR2D<TestValuedCSR>;
type TestSymmetric = SymmetricCSR2D<TestValuedCSR>;

fn build_csr(entries: &[(usize, usize)], shape: (usize, usize)) -> TestCSR {
    let mut csr = TestCSR::with_sparse_shape(shape);
    for &(row, column) in entries {
        MatrixMut::add(&mut csr, (row, column)).unwrap();
    }
    csr
}

fn build_valued_square(order: usize, edges: &[(usize, usize, i32)]) -> TestSquare {
    let mut sorted = edges.to_vec();
    sorted.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut valued = TestValuedCSR::with_sparse_shaped_capacity((order, order), sorted.len());
    let mut diagonal_values = 0;

    for (row, column, value) in sorted {
        MatrixMut::add(&mut valued, (row, column, value)).unwrap();
        if row == column {
            diagonal_values += 1;
        }
    }

    SquareCSR2D::from_parts(valued, diagonal_values)
}

fn build_valued_symmetric(order: usize, upper_edges: &[(usize, usize, i32)]) -> TestSymmetric {
    let mut mirrored = Vec::new();

    for &(row, column, value) in upper_edges {
        mirrored.push((row, column, value));
        if row != column {
            mirrored.push((column, row, value));
        }
    }

    mirrored.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    let mut valued = TestValuedCSR::with_sparse_shaped_capacity((order, order), mirrored.len());
    let mut diagonal_values = 0;

    for (row, column, value) in mirrored {
        MatrixMut::add(&mut valued, (row, column, value)).unwrap();
        if row == column {
            diagonal_values += 1;
        }
    }

    SymmetricCSR2D::from_parts(SquareCSR2D::from_parts(valued, diagonal_values))
}

#[test]
fn test_from_parts_and_into_parts_round_trip() {
    let csr = build_csr(&[(0, 1), (1, 2)], (2, 3));
    let matrix = ValuedCSR2D::<usize, usize, usize, String>::from_parts(
        csr,
        vec![String::from("left"), String::from("right")],
    )
    .unwrap();

    assert_eq!(
        matrix.values_ref().iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["left", "right"]
    );

    let (csr, values) = matrix.into_parts();
    assert_eq!(SparseMatrix::sparse_coordinates(&csr).collect::<Vec<_>>(), vec![(0, 1), (1, 2)]);
    assert_eq!(values, vec![String::from("left"), String::from("right")]);
}

#[test]
fn test_from_parts_rejects_mismatched_values() {
    let csr = build_csr(&[(0, 0), (1, 1)], (2, 2));

    assert_eq!(
        TestValuedCSR::from_parts(csr, vec![10]).unwrap_err(),
        ValuedCsrPartsError::ValuesLengthMismatch { expected: 2, actual: 1 }
    );
}

#[test]
fn test_values_ref_and_values_mut_follow_csr_storage_order() {
    let mut matrix =
        TestValuedCSR::from_parts(build_csr(&[(0, 1), (0, 2), (1, 0)], (2, 3)), vec![10, 20, 30])
            .unwrap();

    for (offset, value) in matrix.values_mut().iter_mut().enumerate() {
        *value += i32::try_from(offset).unwrap();
    }

    let entries: Vec<_> = SparseMatrix::sparse_coordinates(&matrix)
        .zip(matrix.values_ref().iter())
        .map(|(coordinates, value)| (coordinates, *value))
        .collect();

    assert_eq!(entries, vec![((0, 1), 10), ((0, 2), 21), ((1, 0), 32)]);
}

#[test]
fn test_square_wrapper_preserves_inner_raw_value_order() {
    let mut square = build_valued_square(3, &[(0, 0, 10), (0, 2, 20), (1, 1, 30)]);

    *square.sparse_value_at_mut(0, 2).unwrap() = 99;

    assert_eq!(square.as_ref().values_ref(), &[10, 99, 30]);
}

#[test]
fn test_symmetric_wrapper_preserves_inner_raw_value_order() {
    let mut symmetric = build_valued_symmetric(3, &[(0, 1, 10), (1, 2, 20)]);

    *symmetric.sparse_value_at_mut(2, 1).unwrap() = 77;

    assert_eq!(symmetric.as_ref().as_ref().values_ref(), &[10, 10, 20, 77]);
}

#[test]
fn test_csr_with_values_attaches_values_in_storage_order() {
    let valued = build_csr(&[(0, 1), (1, 0)], (2, 2)).with_values(vec![10, 20]).unwrap();

    let entries: Vec<_> = SparseMatrix::sparse_coordinates(&valued)
        .zip(valued.values_ref().iter().copied())
        .collect();

    assert_eq!(entries, vec![((0, 1), 10), ((1, 0), 20)]);
}

#[test]
fn test_from_sorted_upper_triangular_entries_builds_expected_storage() {
    let symmetric = TestSymmetric::from_sorted_upper_triangular_entries(
        4,
        vec![(0, 1, 10), (1, 1, 20), (1, 3, 30)],
    )
    .unwrap();

    assert_eq!(
        SparseMatrix::sparse_coordinates(&symmetric).collect::<Vec<_>>(),
        vec![(0, 1), (1, 0), (1, 1), (1, 3), (3, 1)]
    );
    assert_eq!(symmetric.sparse_values().collect::<Vec<_>>(), vec![10, 10, 20, 30, 30]);
    assert_eq!(symmetric.number_of_defined_diagonal_values(), 1);
}

#[test]
fn test_from_sorted_upper_triangular_entries_rejects_unsorted_input() {
    let error =
        TestSymmetric::from_sorted_upper_triangular_entries(3, vec![(0, 2, 10), (0, 1, 20)])
            .unwrap_err();

    assert!(matches!(error, MutabilityError::UnorderedCoordinate((0, 1))));
}

#[test]
fn test_from_sorted_upper_triangular_entries_rejects_duplicates() {
    let error =
        TestSymmetric::from_sorted_upper_triangular_entries(3, vec![(0, 1, 10), (0, 1, 20)])
            .unwrap_err();

    assert!(matches!(error, MutabilityError::DuplicatedEntry((0, 1))));
}

#[test]
fn test_from_sorted_upper_triangular_entries_rejects_lower_triangular_input() {
    let error =
        TestSymmetric::from_sorted_upper_triangular_entries(3, vec![(2, 1, 10)]).unwrap_err();

    assert!(matches!(error, MutabilityError::OutOfBounds((2, 1), (3, 3), _)));
}

#[test]
fn test_from_sorted_upper_triangular_entries_rejects_out_of_bounds_input() {
    let error =
        TestSymmetric::from_sorted_upper_triangular_entries(3, vec![(0, 3, 10)]).unwrap_err();

    assert!(matches!(error, MutabilityError::OutOfBounds((0, 3), (3, 3), _)));
}
