//! Integration tests for `VecMatrix2D` array conversion.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::VecMatrix2D,
    traits::{DenseValuedMatrix, DenseValuedMatrix2D, Matrix2D},
};

#[test]
fn test_vec_matrix2d_from_array() {
    let matrix: VecMatrix2D<i32> = VecMatrix2D::from([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(matrix.number_of_rows(), 2);
    assert_eq!(matrix.number_of_columns(), 3);
    assert_eq!(matrix.value((0, 0)), 1);
    assert_eq!(matrix.value((0, 2)), 3);
    assert_eq!(matrix.value((1, 0)), 4);
    assert_eq!(matrix.value((1, 2)), 6);
}

#[test]
fn test_vec_matrix2d_from_array_row_values() {
    let matrix: VecMatrix2D<i32> = VecMatrix2D::from([[10, 20], [30, 40], [50, 60]]);

    assert_eq!(matrix.number_of_rows(), 3);
    assert_eq!(matrix.number_of_columns(), 2);

    let row0: Vec<i32> = matrix.row_values(0).collect();
    assert_eq!(row0, vec![10, 20]);
    let row1: Vec<i32> = matrix.row_values(1).collect();
    assert_eq!(row1, vec![30, 40]);
    let row2: Vec<i32> = matrix.row_values(2).collect();
    assert_eq!(row2, vec![50, 60]);
}

#[test]
fn test_vec_matrix2d_from_single_element() {
    let matrix: VecMatrix2D<u8> = VecMatrix2D::from([[42]]);

    assert_eq!(matrix.number_of_rows(), 1);
    assert_eq!(matrix.number_of_columns(), 1);
    assert_eq!(matrix.value((0, 0)), 42);
}
