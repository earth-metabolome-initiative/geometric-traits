//! Boundary and panic-path tests for CSR2D rank/select and mutability guards.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, MutabilityError},
    traits::{MatrixMut, SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrixMut},
};

type TestCSR = CSR2D<usize, usize, usize>;
type TinyCSR = CSR2D<u8, u8, u8>;

#[test]
fn test_rank_row_beyond_offsets_but_within_shape_returns_defined_values() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((5, 5));
    MatrixMut::add(&mut csr, (0, 0)).expect("insert (0,0)");

    // Rows beyond the stored offsets but within declared shape map to the
    // total number of defined values.
    assert_eq!(csr.rank_row(4), 1);
    assert_eq!(csr.rank_row(5), 1);
}

#[test]
#[should_panic(expected = "row index")]
fn test_rank_row_panics_when_row_is_out_of_bounds() {
    let csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    let _ = csr.rank_row(3);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_select_row_panics_when_sparse_index_is_out_of_bounds() {
    let mut csr: TestCSR = SparseMatrixMut::with_sparse_shape((2, 2));
    MatrixMut::add(&mut csr, (0, 1)).expect("insert (0,1)");
    let _ = csr.select_row(csr.number_of_defined_values());
}

#[test]
fn test_add_reports_maxed_out_row_and_column_indices() {
    let mut csr: TinyCSR = SparseMatrixMut::with_sparse_shape((1, 1));

    assert_eq!(
        MatrixMut::add(&mut csr, (0, u8::MAX)),
        Err(MutabilityError::MaxedOutColumnIndex)
    );
    assert_eq!(
        MatrixMut::add(&mut csr, (u8::MAX, 0)),
        Err(MutabilityError::MaxedOutRowIndex)
    );
}

#[test]
fn test_add_reports_maxed_out_sparse_index_on_last_row_growth() {
    let mut csr: TinyCSR = SparseMatrixMut::with_sparse_shape((u8::MAX, 2));

    for row in 0u8..=254 {
        MatrixMut::add(&mut csr, (row, 0)).expect("fill one entry per row");
    }

    assert_eq!(
        MatrixMut::add(&mut csr, (254, 1)),
        Err(MutabilityError::MaxedOutSparseIndex)
    );
}
