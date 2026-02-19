//! Extended tests for ValuedCSR2D sparse valued matrix operations,
//! including sparse_values, max/min sparse values, and TryFrom conversions.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{EdgesBuilder, SizedSparseValuedMatrix, SparseValuedMatrix, WeightedEdges},
};

/// Helper to create a ValuedCSR2D from edge triples.
fn create_valued_csr(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> ValuedCSR2D<usize, usize, usize, f64> {
    GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// SparseValuedMatrix::sparse_values tests
// ============================================================================

#[test]
fn test_sparse_values_all() {
    let m = create_valued_csr(vec![(0, 1, 1.5), (0, 2, 2.5), (1, 2, 3.5)], 3, 3);

    let values: Vec<f64> = m.sparse_values().collect();
    assert_eq!(values.len(), 3);
    assert!((values[0] - 1.5).abs() < f64::EPSILON);
    assert!((values[1] - 2.5).abs() < f64::EPSILON);
    assert!((values[2] - 3.5).abs() < f64::EPSILON);
}

#[test]
fn test_sparse_values_empty_matrix() {
    let m = create_valued_csr(vec![], 3, 3);
    let values: Vec<f64> = m.sparse_values().collect();
    assert!(values.is_empty());
}

#[test]
fn test_sparse_values_single_entry() {
    let m = create_valued_csr(vec![(0, 0, 42.0)], 1, 1);
    let values: Vec<f64> = m.sparse_values().collect();
    assert_eq!(values.len(), 1);
    assert!((values[0] - 42.0).abs() < f64::EPSILON);
}

// ============================================================================
// SparseValuedMatrix max/min sparse value tests
// ============================================================================

#[test]
fn test_max_sparse_value() {
    let m = create_valued_csr(vec![(0, 1, 1.0), (0, 2, 5.0), (1, 0, 3.0)], 2, 3);

    let max = m.max_sparse_value();
    assert!(max.is_some());
    assert!((max.unwrap() - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_min_sparse_value() {
    let m = create_valued_csr(vec![(0, 1, 1.0), (0, 2, 5.0), (1, 0, 3.0)], 2, 3);

    let min = m.min_sparse_value();
    assert!(min.is_some());
    assert!((min.unwrap() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_max_min_sparse_value_empty() {
    let m = create_valued_csr(vec![], 3, 3);
    assert!(m.max_sparse_value().is_none());
    assert!(m.min_sparse_value().is_none());
}

#[test]
fn test_max_min_sparse_value_single() {
    let m = create_valued_csr(vec![(0, 0, 7.0)], 1, 1);
    assert_eq!(m.max_sparse_value(), Some(7.0));
    assert_eq!(m.min_sparse_value(), Some(7.0));
}

// ============================================================================
// SparseValuedMatrix2D::sparse_row_values tests
// ============================================================================

#[test]
fn test_sparse_row_values_per_row() {
    let m = create_valued_csr(
        vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 2, 5.0)],
        3,
        3,
    );

    let row0: Vec<f64> = m.sparse_row_values(0).collect();
    assert_eq!(row0.len(), 2);
    assert!((row0[0] - 1.0).abs() < f64::EPSILON);
    assert!((row0[1] - 2.0).abs() < f64::EPSILON);

    let row1: Vec<f64> = m.sparse_row_values(1).collect();
    assert_eq!(row1.len(), 1);
    assert!((row1[0] - 3.0).abs() < f64::EPSILON);

    let row2: Vec<f64> = m.sparse_row_values(2).collect();
    assert_eq!(row2.len(), 2);
    assert!((row2[0] - 4.0).abs() < f64::EPSILON);
    assert!((row2[1] - 5.0).abs() < f64::EPSILON);
}

#[test]
fn test_sparse_row_max_and_min_value() {
    let m = create_valued_csr(vec![(0, 0, 1.0), (0, 1, 5.0), (0, 2, 3.0)], 1, 3);

    let max = WeightedEdges::max_successor_weight(&m, 0);
    assert_eq!(max, Some(5.0));

    let min = WeightedEdges::min_successor_weight(&m, 0);
    assert_eq!(min, Some(1.0));

    let max_id = WeightedEdges::max_successor_weight_and_id(&m, 0);
    assert_eq!(max_id, Some((5.0, 1)));

    let min_id = WeightedEdges::min_successor_weight_and_id(&m, 0);
    assert_eq!(min_id, Some((1.0, 0)));
}

// ============================================================================
// SizedSparseValuedMatrix::select_value tests
// ============================================================================

#[test]
fn test_select_value() {
    let m = create_valued_csr(vec![(0, 1, 10.0), (1, 0, 20.0), (1, 2, 30.0)], 2, 3);

    assert!((m.select_value(0) - 10.0).abs() < f64::EPSILON);
    assert!((m.select_value(1) - 20.0).abs() < f64::EPSILON);
    assert!((m.select_value(2) - 30.0).abs() < f64::EPSILON);
}

// ============================================================================
// WeightedEdges::sparse_weights on edges directly
// ============================================================================

#[test]
fn test_sparse_weights() {
    let m = create_valued_csr(vec![(0, 1, 2.0), (0, 2, 4.0), (1, 2, 6.0)], 3, 3);

    let weights: Vec<f64> = WeightedEdges::sparse_weights(&m).collect();
    assert_eq!(weights.len(), 3);
    assert!((weights[0] - 2.0).abs() < f64::EPSILON);
    assert!((weights[1] - 4.0).abs() < f64::EPSILON);
    assert!((weights[2] - 6.0).abs() < f64::EPSILON);
}

// ============================================================================
// TryFrom dense array to sparse ValuedCSR2D
// ============================================================================

#[test]
fn test_try_from_dense_array() {
    // TryFrom stores ALL values (including zeros) as sparse entries
    let dense = [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]];

    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from(dense).unwrap();

    assert_eq!(m.number_of_rows(), 3);
    assert_eq!(m.number_of_columns(), 3);
    // All 9 entries are stored (dense conversion)
    assert_eq!(m.number_of_defined_values(), 9);

    // Check the sparse row has all columns
    let row0_cols: Vec<usize> = m.sparse_row(0).collect();
    assert_eq!(row0_cols, vec![0, 1, 2]);

    // Check values are correct
    let row0_values: Vec<f64> = m.sparse_row_values(0).collect();
    assert!((row0_values[0] - 0.0).abs() < f64::EPSILON);
    assert!((row0_values[1] - 1.0).abs() < f64::EPSILON);
    assert!((row0_values[2] - 0.0).abs() < f64::EPSILON);

    let row1_values: Vec<f64> = m.sparse_row_values(1).collect();
    assert!((row1_values[0] - 2.0).abs() < f64::EPSILON);
    assert!((row1_values[1] - 0.0).abs() < f64::EPSILON);
    assert!((row1_values[2] - 3.0).abs() < f64::EPSILON);
}

#[test]
fn test_try_from_dense_array_all_zeros() {
    let dense = [[0.0_f64; 2]; 2];

    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from(dense).unwrap();

    assert_eq!(m.number_of_rows(), 2);
    assert_eq!(m.number_of_columns(), 2);
    // TryFrom stores all entries including zeros
    assert_eq!(m.number_of_defined_values(), 4);
}

#[test]
fn test_try_from_dense_array_all_nonzero() {
    let dense = [[1.0, 2.0], [3.0, 4.0]];

    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from(dense).unwrap();

    assert_eq!(m.number_of_defined_values(), 4);

    let values: Vec<f64> = m.sparse_values().collect();
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < f64::EPSILON);
    assert!((values[1] - 2.0).abs() < f64::EPSILON);
    assert!((values[2] - 3.0).abs() < f64::EPSILON);
    assert!((values[3] - 4.0).abs() < f64::EPSILON);
}
