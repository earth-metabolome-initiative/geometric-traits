//! Regression tests for RaggedVector sparse-count bookkeeping semantics.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::RaggedVector,
    traits::{MatrixMut, SparseMatrixMut, SizedSparseMatrix, TransposableMatrix2D},
};

#[test]
fn test_with_sparse_capacity_does_not_set_defined_values() {
    let rv: RaggedVector<u16, u8, u8> = SparseMatrixMut::with_sparse_capacity(10_u16);
    assert_eq!(rv.number_of_defined_values(), 0);
}

#[test]
fn test_with_sparse_shaped_capacity_does_not_set_defined_values() {
    let rv: RaggedVector<u16, u8, u8> = SparseMatrixMut::with_sparse_shaped_capacity((4, 5), 7_u16);
    assert_eq!(rv.number_of_defined_values(), 0);
}

#[test]
fn test_transpose_preserves_defined_values_count() {
    let mut rv: RaggedVector<u16, u8, u8> = RaggedVector::default();
    rv.add((0, 1)).unwrap();
    rv.add((0, 3)).unwrap();
    rv.add((1, 2)).unwrap();

    let transposed = rv.transpose();
    assert_eq!(transposed.number_of_defined_values(), rv.number_of_defined_values());
}
