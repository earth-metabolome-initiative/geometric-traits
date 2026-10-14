//! Test submodule to test the `SquareCSR2D` struct.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
};

#[test]
/// Test case identified by fuzzing.
fn test_square_csr2d_fuzz_case2() {
    let matrix: SquareCSR2D<CSR2D<usize, usize, usize>> =
        SquareCSR2D::from_entries(vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]).unwrap();
    let first_row = matrix.sparse_row(0).collect::<Vec<usize>>();
    let second_row = matrix.sparse_row(1).collect::<Vec<usize>>();
    let third_row = matrix.sparse_row(2).collect::<Vec<usize>>();
    assert_eq!(first_row, vec![0, 1]);
    assert_eq!(second_row, vec![0, 1, 2]);
    assert_eq!(third_row, Vec::new());
}

#[test]
fn test_square_csr2d_is_symmetric_detects_asymmetry() {
    let matrix: SquareCSR2D<CSR2D<usize, usize, usize>> =
        SquareCSR2D::from_entries(vec![(0, 1), (1, 2)]).unwrap();
    assert!(!matrix.is_symmetric());
}

#[test]
fn test_square_csr2d_is_symmetric_accepts_symmetric_input() {
    let matrix: SquareCSR2D<CSR2D<usize, usize, usize>> =
        SquareCSR2D::from_entries(vec![(0, 1), (1, 0), (1, 2), (2, 1)]).unwrap();
    assert!(matrix.is_symmetric());
}

#[test]
fn test_symmetric_csr2d_is_symmetric_fast_path() {
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_shape(3)
        .edges([(0, 1), (1, 2)].into_iter())
        .build()
        .unwrap();
    assert!(matrix.is_symmetric());
}
