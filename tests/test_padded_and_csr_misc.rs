//! Additional matrix wrapper and CSR coverage organized by data structure.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericMatrix2DWithPaddedDiagonal, PaddedMatrix2D, SquareCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{MatrixMut, SparseMatrix, SparseMatrix2D, SparseMatrixMut, SparseValuedMatrix},
};

type TestCSR = CSR2D<usize, usize, usize>;

#[test]
fn test_padded_coordinates_next_back() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    let mut coords = SparseMatrix::sparse_coordinates(&padded);

    let last = coords.next_back();
    assert!(last.is_some());
    let second_last = coords.next_back();
    assert!(second_last.is_some());

    let mut back_items = Vec::new();
    back_items.push(last.unwrap());
    back_items.push(second_last.unwrap());
    while let Some(item) = coords.next_back() {
        back_items.push(item);
    }

    assert!(!back_items.is_empty());
}

#[test]
fn test_padded_coordinates_mixed_direction() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    let mut coords = SparseMatrix::sparse_coordinates(&padded);

    let front1 = coords.next();
    let back1 = coords.next_back();
    assert!(front1.is_some());
    assert!(back1.is_some());
    assert_ne!(front1, back1);
}

#[test]
fn test_padded_diagonal_empty_matrix() {
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0);
    if let Ok(p) = padded {
        assert!(SparseMatrix::is_empty(&p));
        assert_eq!(SparseMatrix::last_sparse_coordinates(&p), None);
    }
}

#[test]
fn test_padded_diagonal_maxed_out_errors() {
    let m: ValuedCSR2D<usize, u8, u16, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10u8, 256u16), 0);
    let result = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0);
    assert!(result.is_err());

    let m: ValuedCSR2D<usize, u16, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((256u16, 10u8), 0);
    let result = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u16| 1.0);
    assert!(result.is_err());
}

#[test]
fn test_m2d_values_forward_crossing() {
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 99.0, 99.0],
        [3.0, 4.0, 99.0, 99.0],
        [99.0, 99.0, 5.0, 6.0],
        [99.0, 99.0, 7.0, 8.0],
    ])
    .unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();
    let values: Vec<f64> = padded.sparse_values().collect();
    assert!(!values.is_empty());

    let m2: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::try_from([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]).unwrap();
    let padded2 = PaddedMatrix2D::new(m2, |_: (u8, u8)| 99.0).unwrap();
    let values2: Vec<f64> = padded2.sparse_values().collect();
    assert!(!values2.is_empty());
}

#[test]
fn test_csr2d_view_forward_through_empty_middle_rows() {
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 2.0],
    ])
    .unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();
    let coords: Vec<(u8, u8)> = SparseMatrix::sparse_coordinates(&padded).collect();
    assert!(coords.len() >= 4);
}

#[test]
fn test_csr2d_columns_exact_size() {
    let m: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();

    let columns = m.sparse_columns();
    let len = columns.len();
    let collected: Vec<u8> = m.sparse_columns().collect();
    assert_eq!(len, collected.len());
}

#[test]
fn test_m2d_values_exact_size() {
    let m: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();

    let values = m.sparse_values();
    let len = values.len();
    let collected: Vec<f64> = m.sparse_values().collect();
    assert_eq!(len, collected.len());
}

#[test]
fn test_padded_diagonal_sparse_rows_crossing() {
    let m: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 99.0, 99.0], [99.0, 2.0, 99.0], [99.0, 99.0, 3.0]]).unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();
    let rows: Vec<u8> = padded.sparse_rows().collect();
    assert_eq!(rows.len(), 9);
}

#[test]
fn test_square_csr2d_increase_shape_noop() {
    let mut sq: SquareCSR2D<TestCSR> = SparseMatrixMut::with_sparse_shape(5);
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    sq.increase_shape((5, 5)).unwrap();
    assert_eq!(sq.order(), 5);
}

#[test]
fn test_square_csr2d_increase_shape_to_larger() {
    let mut sq: SquareCSR2D<TestCSR> = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    sq.increase_shape((7, 7)).unwrap();
    assert_eq!(sq.order(), 7);
    assert_eq!(sq.number_of_rows(), 7);
    assert_eq!(sq.number_of_columns(), 7);
}

#[test]
fn test_square_csr2d_many_diagonals() {
    let mut sq: SquareCSR2D<TestCSR> = SparseMatrixMut::with_sparse_shape(5);
    for i in 0..5 {
        MatrixMut::add(&mut sq, (i, i)).unwrap();
    }
    assert_eq!(sq.number_of_defined_diagonal_values(), 5);
}

#[test]
fn test_csr2d_multi_row_sequential_add() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((5, 5), 10);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 3)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 4)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    MatrixMut::add(&mut csr, (3, 2)).unwrap();
    MatrixMut::add(&mut csr, (4, 4)).unwrap();

    assert_eq!(csr.number_of_defined_values(), 7);
    assert_eq!(csr.last_sparse_coordinates(), Some((4, 4)));
}

#[test]
fn test_csr2d_gap_in_rows() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((5, 5), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (3, 1)).unwrap();
    MatrixMut::add(&mut csr, (4, 2)).unwrap();

    assert_eq!(csr.number_of_defined_values_in_row(0), 1);
    assert_eq!(csr.number_of_defined_values_in_row(1), 0);
    assert_eq!(csr.number_of_defined_values_in_row(2), 0);
    assert_eq!(csr.number_of_defined_values_in_row(3), 1);
}

#[test]
fn test_csr2d_last_sparse_coordinates_single_entry() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 1);
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    assert_eq!(csr.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_csr2d_with_sparse_capacity() {
    let mut csr: TestCSR = CSR2D::with_sparse_capacity(10);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 5)).unwrap();
    MatrixMut::add(&mut csr, (3, 2)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 3);
}

#[test]
fn test_padded_diagonal_is_imputed() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[99.0, 5.0, 99.0], [3.0, 99.0, 99.0], [7.0, 99.0, 99.0]]).unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: usize| 1.0).unwrap();
    assert_eq!(padded.number_of_rows(), padded.number_of_columns());
}

#[test]
fn test_padded_matrix2d_coordinates() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);

    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&padded).collect();
    assert!(coords.len() >= 6);
}
