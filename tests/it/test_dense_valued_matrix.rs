//! Tests for DenseValuedMatrix2D trait methods: row_values, row_max_value,
//! row_min_value, row_max_value_and_column, row_min_value_and_column.
//! Also tests DenseValuedMatrix trait: values, max_value, min_value, value.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{DenseValuedMatrix, DenseValuedMatrix2D, EdgesBuilder},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;
type TestPadded = PaddedMatrix2D<TestValCSR, Box<dyn Fn((usize, usize)) -> f64>>;

/// Helper to create a PaddedMatrix2D from sparse valued entries.
fn create_padded(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
    default_val: f64,
) -> TestPadded {
    let inner = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    PaddedMatrix2D::new(
        inner,
        Box::new(move |_: (usize, usize)| default_val) as Box<dyn Fn((usize, usize)) -> f64>,
    )
    .unwrap()
}

// ============================================================================
// DenseValuedMatrix2D::row_values tests
// ============================================================================

#[test]
fn test_row_values_all_imputed() {
    let padded = create_padded(vec![], 2, 3, 0.0);

    let row0: Vec<f64> = padded.row_values(0).collect();
    assert_eq!(row0.len(), 3);
    assert!(row0.iter().all(|v| (*v - 0.0).abs() < f64::EPSILON));
}

#[test]
fn test_row_values_with_sparse_entries() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 2, 3.0)], 2, 3, 0.0);

    let row0: Vec<f64> = padded.row_values(0).collect();
    assert_eq!(row0.len(), 3);
    assert!((row0[0] - 1.0).abs() < f64::EPSILON);
    assert!((row0[1] - 0.0).abs() < f64::EPSILON); // imputed
    assert!((row0[2] - 3.0).abs() < f64::EPSILON);
}

#[test]
fn test_row_values_fully_sparse() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)], 1, 3, 99.0);

    let row0: Vec<f64> = padded.row_values(0).collect();
    assert_eq!(row0, vec![1.0, 2.0, 3.0]);
}

// ============================================================================
// DenseValuedMatrix2D::row_max_value tests
// ============================================================================

#[test]
fn test_row_max_value() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3, 0.0);

    assert_eq!(padded.row_max_value(0), Some(5.0));
}

#[test]
fn test_row_max_value_all_imputed() {
    let padded = create_padded(vec![], 2, 3, 7.5);

    assert_eq!(padded.row_max_value(0), Some(7.5));
}

#[test]
fn test_row_max_value_mixed() {
    let padded = create_padded(vec![(0, 1, 10.0)], 1, 3, 0.0);

    assert_eq!(padded.row_max_value(0), Some(10.0));
}

// ============================================================================
// DenseValuedMatrix2D::row_min_value tests
// ============================================================================

#[test]
fn test_row_min_value() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3, 0.0);

    assert_eq!(padded.row_min_value(0), Some(1.0));
}

#[test]
fn test_row_min_value_with_imputed_smaller() {
    let padded = create_padded(vec![(0, 0, 10.0)], 1, 3, -1.0);

    // The imputed value -1.0 is smaller than the sparse value 10.0
    assert_eq!(padded.row_min_value(0), Some(-1.0));
}

// ============================================================================
// DenseValuedMatrix2D::row_max_value_and_column tests
// ============================================================================

#[test]
fn test_row_max_value_and_column() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3, 0.0);

    let result = padded.row_max_value_and_column(0);
    assert_eq!(result, Some((1, 5.0)));
}

#[test]
fn test_row_max_value_and_column_imputed_wins() {
    let padded = create_padded(vec![(0, 0, 1.0)], 1, 3, 100.0);

    let result = padded.row_max_value_and_column(0);
    assert!(result.is_some());
    let (col, val) = result.unwrap();
    assert!((val - 100.0).abs() < f64::EPSILON);
    // The imputed column should be either 1 or 2
    assert!(col == 1 || col == 2);
}

// ============================================================================
// DenseValuedMatrix2D::row_min_value_and_column tests
// ============================================================================

#[test]
fn test_row_min_value_and_column() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3, 0.0);

    let result = padded.row_min_value_and_column(0);
    assert_eq!(result, Some((0, 1.0)));
}

#[test]
fn test_row_min_value_and_column_imputed_wins() {
    let padded = create_padded(vec![(0, 0, 10.0)], 1, 3, -5.0);

    let result = padded.row_min_value_and_column(0);
    assert!(result.is_some());
    let (col, val) = result.unwrap();
    assert!((val - (-5.0)).abs() < f64::EPSILON);
    assert!(col == 1 || col == 2);
}

// ============================================================================
// DenseValuedMatrix::values tests
// ============================================================================

#[test]
fn test_dense_values_all_imputed() {
    let padded = create_padded(vec![], 2, 2, 3.0);

    let values: Vec<f64> = padded.values().collect();
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|v| (*v - 3.0).abs() < f64::EPSILON));
}

#[test]
fn test_dense_values_mixed() {
    let padded = create_padded(vec![(0, 0, 1.0), (1, 1, 2.0)], 2, 2, 0.0);

    let values: Vec<f64> = padded.values().collect();
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < f64::EPSILON); // (0,0) = 1.0
    assert!((values[1] - 0.0).abs() < f64::EPSILON); // (0,1) = imputed 0.0
    assert!((values[2] - 0.0).abs() < f64::EPSILON); // (1,0) = imputed 0.0
    assert!((values[3] - 2.0).abs() < f64::EPSILON); // (1,1) = 2.0
}

// ============================================================================
// DenseValuedMatrix::value tests
// ============================================================================

#[test]
fn test_dense_value_sparse_entry() {
    let padded = create_padded(vec![(0, 0, 42.0)], 2, 2, 0.0);

    assert!((padded.value((0, 0)) - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_dense_value_imputed_entry() {
    let padded = create_padded(vec![(0, 0, 42.0)], 2, 2, -1.0);

    assert!((padded.value((0, 1)) - (-1.0)).abs() < f64::EPSILON);
    assert!((padded.value((1, 0)) - (-1.0)).abs() < f64::EPSILON);
}

// ============================================================================
// DenseValuedMatrix::max_value and min_value tests
// ============================================================================

#[test]
fn test_dense_max_value() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0)], 2, 2, 0.0);

    assert_eq!(padded.max_value(), Some(5.0));
}

#[test]
fn test_dense_min_value() {
    let padded = create_padded(vec![(0, 0, 1.0), (0, 1, 5.0), (1, 0, 3.0)], 2, 2, 0.0);

    assert_eq!(padded.min_value(), Some(0.0));
}

#[test]
fn test_dense_max_min_all_imputed() {
    let padded = create_padded(vec![], 2, 2, 7.0);

    assert_eq!(padded.max_value(), Some(7.0));
    assert_eq!(padded.min_value(), Some(7.0));
}

// ============================================================================
// Row values on second row
// ============================================================================

#[test]
fn test_row_values_second_row() {
    let padded = create_padded(vec![(0, 0, 1.0), (1, 0, 10.0), (1, 2, 30.0)], 2, 3, 0.0);

    let row1: Vec<f64> = padded.row_values(1).collect();
    assert_eq!(row1.len(), 3);
    assert!((row1[0] - 10.0).abs() < f64::EPSILON);
    assert!((row1[1] - 0.0).abs() < f64::EPSILON); // imputed
    assert!((row1[2] - 30.0).abs() < f64::EPSILON);
}
