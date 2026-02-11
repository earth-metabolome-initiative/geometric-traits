//! Tests for LAPJVError enum.

use std::error::Error;

use geometric_traits::prelude::LAPJVError;

#[test]
fn test_lapjv_error_non_square_matrix() {
    let err = LAPJVError::NonSquareMatrix;
    assert_eq!(format!("{err}"), "The matrix is not square.");
}

#[test]
fn test_lapjv_error_empty_matrix() {
    let err = LAPJVError::EmptyMatrix;
    assert_eq!(format!("{err}"), "The matrix is empty.");
}

#[test]
fn test_lapjv_error_zero_values() {
    let err = LAPJVError::ZeroValues;
    assert_eq!(format!("{err}"), "The matrix contains zero values.");
}

#[test]
fn test_lapjv_error_negative_values() {
    let err = LAPJVError::NegativeValues;
    assert_eq!(format!("{err}"), "The matrix contains negative values.");
}

#[test]
fn test_lapjv_error_non_finite_values() {
    let err = LAPJVError::NonFiniteValues;
    assert_eq!(format!("{err}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lapjv_error_value_too_large() {
    let err = LAPJVError::ValueTooLarge;
    assert_eq!(format!("{err}"), "The matrix contains a value larger than the maximum cost.");
}

#[test]
fn test_lapjv_error_maximal_cost_not_finite() {
    let err = LAPJVError::MaximalCostNotFinite;
    assert_eq!(format!("{err}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lapjv_error_maximal_cost_not_positive() {
    let err = LAPJVError::MaximalCostNotPositive;
    assert_eq!(format!("{err}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lapjv_error_padding_value_not_finite() {
    let err = LAPJVError::PaddingValueNotFinite;
    assert_eq!(format!("{err}"), "The provided padding value is not a finite number.");
}

#[test]
fn test_lapjv_error_padding_value_not_positive() {
    let err = LAPJVError::PaddingValueNotPositive;
    assert_eq!(format!("{err}"), "The provided padding value is not a positive number.");
}

#[test]
fn test_lapjv_error_debug() {
    let err = LAPJVError::EmptyMatrix;
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("EmptyMatrix"));
}

#[test]
fn test_lapjv_error_clone() {
    let err = LAPJVError::NonSquareMatrix;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_lapjv_error_eq() {
    assert_eq!(LAPJVError::EmptyMatrix, LAPJVError::EmptyMatrix);
    assert_ne!(LAPJVError::EmptyMatrix, LAPJVError::NonSquareMatrix);
}

#[test]
fn test_lapjv_error_is_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPJVError::NonSquareMatrix);
}
