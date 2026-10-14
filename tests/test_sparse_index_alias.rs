//! Regression tests for the `sparse_index` naming alias.
#![cfg(feature = "std")]

use std::vec::Vec;

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::MatrixMut,
};

type TestValuedCSR = ValuedCSR2D<usize, usize, usize, i32>;
type TestSquare = SquareCSR2D<TestValuedCSR>;
type TestSymmetric = SymmetricCSR2D<TestValuedCSR>;

fn build_csr(entries: &[(usize, usize)], shape: (usize, usize)) -> CSR2D<usize, usize, usize> {
    let mut csr = CSR2D::with_sparse_shape(shape);
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
fn test_sparse_index_alias_matches_try_rank_on_valued_csr() {
    let matrix =
        TestValuedCSR::from_parts(build_csr(&[(0, 1), (1, 0), (1, 2)], (2, 3)), vec![10, 20, 30])
            .unwrap();

    for (row, column) in [(0, 1), (1, 0), (1, 2), (0, 0), (0, 2)] {
        assert_eq!(matrix.sparse_index(row, column), matrix.try_rank(row, column));
    }
}

#[test]
fn test_sparse_index_alias_matches_try_rank_on_wrappers() {
    let square = build_valued_square(3, &[(0, 0, 10), (0, 2, 20), (1, 1, 30)]);
    let symmetric = build_valued_symmetric(3, &[(0, 1, 5), (1, 2, 7)]);

    for (row, column) in [(0, 0), (0, 2), (1, 1), (2, 2)] {
        assert_eq!(square.sparse_index(row, column), square.try_rank(row, column));
    }

    for (row, column) in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 2)] {
        assert_eq!(symmetric.sparse_index(row, column), symmetric.try_rank(row, column));
    }
}
