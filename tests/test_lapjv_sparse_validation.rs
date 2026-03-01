//! Tests for LAPJV/SparseLAPJV/LAPMOD/Jaqaman input validation paths.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
};

type TestCsr = ValuedCSR2D<u8, u8, u8, f64>;
type TestPadded = PaddedMatrix2D<TestCsr, fn((u8, u8)) -> f64>;

fn padded_value(_: (u8, u8)) -> f64 {
    900.0
}

// ============================================================================
// LAPJV max_cost validation via PaddedMatrix2D (DenseValuedMatrix2D)
// ============================================================================

fn make_padded_2x2() -> TestPadded {
    let csr: TestCsr = ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    PaddedMatrix2D::new(csr, padded_value as fn((u8, u8)) -> f64).unwrap()
}

#[test]
fn test_lapjv_max_cost_not_finite() {
    let padded = make_padded_2x2();
    let result = padded.lapjv(f64::INFINITY);
    assert_eq!(result, Err(LAPJVError::MaximalCostNotFinite));
}

#[test]
fn test_lapjv_max_cost_nan() {
    let padded = make_padded_2x2();
    let result = padded.lapjv(f64::NAN);
    assert_eq!(result, Err(LAPJVError::MaximalCostNotFinite));
}

#[test]
fn test_lapjv_max_cost_not_positive() {
    let padded = make_padded_2x2();
    let result = padded.lapjv(0.0);
    assert_eq!(result, Err(LAPJVError::MaximalCostNotPositive));
}

#[test]
fn test_lapjv_max_cost_negative() {
    let padded = make_padded_2x2();
    let result = padded.lapjv(-10.0);
    assert_eq!(result, Err(LAPJVError::MaximalCostNotPositive));
}

// ============================================================================
// LAPJV value validation (zero, negative, non-finite, too-large)
// ============================================================================

#[test]
fn test_lapjv_zero_values_error() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[0.0, 2.0], [3.0, 4.0]]).unwrap();
    let padded = PaddedMatrix2D::new(csr, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0);
    assert_eq!(result, Err(LAPJVError::ZeroValues));
}

#[test]
fn test_lapjv_negative_values_error() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[-1.0, 2.0], [3.0, 4.0]]).unwrap();
    let padded = PaddedMatrix2D::new(csr, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0);
    assert_eq!(result, Err(LAPJVError::NegativeValues));
}

#[test]
fn test_lapjv_non_finite_values_error() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[f64::NAN, 2.0], [3.0, 4.0]]).unwrap();
    let padded = PaddedMatrix2D::new(csr, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0);
    assert_eq!(result, Err(LAPJVError::NonFiniteValues));
}

#[test]
fn test_lapjv_value_too_large_error() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1000.0, 2.0], [3.0, 4.0]]).unwrap();
    let padded = PaddedMatrix2D::new(csr, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0);
    assert_eq!(result, Err(LAPJVError::ValueTooLarge));
}

// ============================================================================
// SparseLAPJV input validation
// ============================================================================

#[test]
fn test_sparse_lapjv_padding_not_finite() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(f64::INFINITY, 1000.0);
    assert_eq!(result, Err(LAPError::PaddingValueNotFinite));
}

#[test]
fn test_sparse_lapjv_padding_nan() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(f64::NAN, 1000.0);
    assert_eq!(result, Err(LAPError::PaddingValueNotFinite));
}

#[test]
fn test_sparse_lapjv_padding_not_positive() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(0.0, 1000.0);
    assert_eq!(result, Err(LAPError::PaddingValueNotPositive));
}

#[test]
fn test_sparse_lapjv_padding_negative() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(-5.0, 1000.0);
    assert_eq!(result, Err(LAPError::PaddingValueNotPositive));
}

#[test]
fn test_sparse_lapjv_padding_ge_max_cost() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    // padding_cost == max_cost
    let result = csr.sparse_lapjv(1000.0, 1000.0);
    assert_eq!(result, Err(LAPError::ValueTooLarge));
    // padding_cost > max_cost
    let result = csr.sparse_lapjv(2000.0, 1000.0);
    assert_eq!(result, Err(LAPError::ValueTooLarge));
}

#[test]
fn test_sparse_lapjv_max_cost_not_finite() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(900.0, f64::INFINITY);
    assert_eq!(result, Err(LAPError::MaximalCostNotFinite));
    let result = csr.sparse_lapjv(900.0, f64::NAN);
    assert_eq!(result, Err(LAPError::MaximalCostNotFinite));
}

#[test]
fn test_sparse_lapjv_max_cost_not_positive() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(900.0, 0.0);
    assert_eq!(result, Err(LAPError::MaximalCostNotPositive));
    let result = csr.sparse_lapjv(900.0, -1.0);
    assert_eq!(result, Err(LAPError::MaximalCostNotPositive));
}

#[test]
fn test_sparse_lapjv_empty_matrix() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);
    let result = csr.sparse_lapjv(900.0, 1000.0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_sparse_lapjv_empty_matrix_invalid_max_cost_still_errors() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);

    let result = csr.sparse_lapjv(900.0, f64::NAN);
    assert_eq!(result, Err(LAPError::MaximalCostNotFinite));

    let result = csr.sparse_lapjv(900.0, 0.0);
    assert_eq!(result, Err(LAPError::MaximalCostNotPositive));
}

#[test]
fn test_sparse_lapjv_padding_too_small() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[5.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.sparse_lapjv(1.0, 1000.0);
    assert_eq!(result, Err(LAPError::PaddingCostTooSmall));
}

// ============================================================================
// LAPMOD max_cost validation
// ============================================================================

#[test]
fn test_lapmod_max_cost_not_finite() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.lapmod(f64::INFINITY);
    assert_eq!(result, Err(LAPMODError::MaximalCostNotFinite));
}

#[test]
fn test_lapmod_max_cost_nan() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.lapmod(f64::NAN);
    assert_eq!(result, Err(LAPMODError::MaximalCostNotFinite));
}

#[test]
fn test_lapmod_max_cost_not_positive() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.lapmod(0.0);
    assert_eq!(result, Err(LAPMODError::MaximalCostNotPositive));
}

#[test]
fn test_lapmod_max_cost_negative() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let result = csr.lapmod(-1.0);
    assert_eq!(result, Err(LAPMODError::MaximalCostNotPositive));
}

#[test]
fn test_lapmod_non_square() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let result = csr.lapmod(1000.0);
    assert_eq!(result, Err(LAPMODError::NonSquareMatrix));
}

#[test]
fn test_lapmod_empty_matrix() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);
    let result = csr.lapmod(1000.0).unwrap();
    assert!(result.is_empty());
}

// ============================================================================
// Jaqaman error via From conversion (triggers LAPError::from(LAPMODError))
// ============================================================================

#[test]
fn test_jaqaman_nan_max_cost_validation() {
    // Build a non-empty sparse matrix for standard validation behavior.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
    MatrixMut::add(&mut csr, (0, 0, 1.0)).unwrap();
    MatrixMut::add(&mut csr, (1, 1, 1.0)).unwrap();

    let result = csr.jaqaman(900.0, f64::NAN);
    assert_eq!(result, Err(LAPError::MaximalCostNotFinite));

    let result = csr.jaqaman(900.0, -1.0);
    assert_eq!(result, Err(LAPError::MaximalCostNotPositive));
}

#[test]
fn test_jaqaman_empty_matrix_invalid_max_cost_still_errors() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);

    let result = csr.jaqaman(900.0, f64::NAN);
    assert_eq!(result, Err(LAPError::MaximalCostNotFinite));

    let result = csr.jaqaman(900.0, 0.0);
    assert_eq!(result, Err(LAPError::MaximalCostNotPositive));
}

// ============================================================================
// Complex LAPJV/LAPMOD matrices that exercise conflict resolution and
// augmenting paths in inner.rs
// ============================================================================

#[test]
fn test_lapjv_4x4_conflict_resolution() {
    // Matrix where rows 0 and 1 both prefer column 0 — column reduction conflict
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 1.0, 50.0],
        [50.0, 50.0, 50.0, 1.0],
    ])
    .unwrap();
    let mut result = csr.sparse_lapjv(900.0, 1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_lapjv_5x5_augmenting_paths() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 90.0, 90.0, 90.0],
        [2.0, 1.0, 90.0, 90.0, 90.0],
        [90.0, 90.0, 1.0, 2.0, 90.0],
        [90.0, 90.0, 2.0, 1.0, 90.0],
        [90.0, 90.0, 90.0, 90.0, 1.0],
    ])
    .unwrap();
    let mut result = csr.sparse_lapjv(900.0, 1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);
}

#[test]
fn test_lapmod_4x4_conflict_resolution() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 1.0, 50.0],
        [50.0, 50.0, 50.0, 1.0],
    ])
    .unwrap();
    let mut result = csr.lapmod(1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_lapmod_5x5_augmenting_paths() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 90.0, 90.0, 90.0],
        [2.0, 1.0, 90.0, 90.0, 90.0],
        [90.0, 90.0, 1.0, 2.0, 90.0],
        [90.0, 90.0, 2.0, 1.0, 90.0],
        [90.0, 90.0, 90.0, 90.0, 1.0],
    ])
    .unwrap();
    let mut result = csr.lapmod(1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);
}

#[test]
fn test_lapmod_6x6_complex_augmentation() {
    // Matrix forcing complex augmenting path searches and scan_sparse expansion
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        [3.0, 1.0, 5.0, 7.0, 9.0, 11.0],
        [5.0, 5.0, 1.0, 3.0, 9.0, 11.0],
        [7.0, 7.0, 3.0, 1.0, 9.0, 11.0],
        [9.0, 9.0, 9.0, 9.0, 1.0, 3.0],
        [11.0, 11.0, 9.0, 9.0, 3.0, 1.0],
    ])
    .unwrap();
    let mut result = csr.lapmod(1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result.len(), 6);
    // Verify valid assignment (all columns distinct)
    let mut cols: Vec<u8> = result.iter().map(|&(_, c)| c).collect();
    cols.sort_unstable();
    cols.dedup();
    assert_eq!(cols.len(), 6);
}

#[test]
fn test_lapmod_sparse_needs_augmentation() {
    // Sparse matrix where column reduction leaves unassigned rows, requiring
    // find_path_sparse and augmentation.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((4, 4), 8);
    MatrixMut::add(&mut csr, (0, 0, 1.0)).unwrap();
    MatrixMut::add(&mut csr, (0, 1, 5.0)).unwrap();
    MatrixMut::add(&mut csr, (1, 0, 2.0)).unwrap();
    MatrixMut::add(&mut csr, (1, 2, 3.0)).unwrap();
    MatrixMut::add(&mut csr, (2, 1, 2.0)).unwrap();
    MatrixMut::add(&mut csr, (2, 3, 4.0)).unwrap();
    MatrixMut::add(&mut csr, (3, 2, 1.0)).unwrap();
    MatrixMut::add(&mut csr, (3, 3, 5.0)).unwrap();

    let mut result = csr.lapmod(1000.0).unwrap();
    result.sort_unstable_by_key(|&(r, _)| r);
    assert_eq!(result.len(), 4);
    let mut cols: Vec<u8> = result.iter().map(|&(_, c)| c).collect();
    cols.sort_unstable();
    cols.dedup();
    assert_eq!(cols.len(), 4);
}
