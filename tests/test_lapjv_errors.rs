//! Tests for LAPError variants used by LAPJV/LAPMOD.
#![cfg(feature = "std")]

use std::error::Error;

use geometric_traits::prelude::LAPError;

#[test]
fn test_lapjv_error_non_fractional_value_type_unsupported() {
    let err = LAPError::NonFractionalValueTypeUnsupported;
    assert_eq!(
        format!("{err}"),
        "The matrix value type is non-fractional and is not supported by LAP algorithms."
    );
}

#[test]
fn test_lapjv_error_non_square_matrix() {
    let err = LAPError::NonSquareMatrix;
    assert_eq!(format!("{err}"), "The matrix is not square.");
}

#[test]
fn test_lapjv_error_empty_matrix() {
    let err = LAPError::EmptyMatrix;
    assert_eq!(format!("{err}"), "The matrix is empty.");
}

#[test]
fn test_lapjv_error_zero_values() {
    let err = LAPError::ZeroValues;
    assert_eq!(format!("{err}"), "The matrix contains zero values.");
}

#[test]
fn test_lapjv_error_negative_values() {
    let err = LAPError::NegativeValues;
    assert_eq!(format!("{err}"), "The matrix contains negative values.");
}

#[test]
fn test_lapjv_error_non_finite_values() {
    let err = LAPError::NonFiniteValues;
    assert_eq!(format!("{err}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lapjv_error_value_too_large() {
    let err = LAPError::ValueTooLarge;
    assert_eq!(format!("{err}"), "The matrix contains a value larger than the maximum cost.");
}

#[test]
fn test_lapjv_error_maximal_cost_not_finite() {
    let err = LAPError::MaximalCostNotFinite;
    assert_eq!(format!("{err}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lapjv_error_maximal_cost_not_positive() {
    let err = LAPError::MaximalCostNotPositive;
    assert_eq!(format!("{err}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lapjv_error_padding_value_not_finite() {
    let err = LAPError::PaddingValueNotFinite;
    assert_eq!(format!("{err}"), "The provided padding value is not a finite number.");
}

#[test]
fn test_lapjv_error_padding_value_not_positive() {
    let err = LAPError::PaddingValueNotPositive;
    assert_eq!(format!("{err}"), "The provided padding value is not a positive number.");
}

#[test]
fn test_lapjv_error_debug() {
    let err = LAPError::EmptyMatrix;
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("EmptyMatrix"));
}

#[test]
fn test_lapjv_error_clone() {
    let err = LAPError::NonSquareMatrix;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_lapjv_error_eq() {
    assert_eq!(LAPError::EmptyMatrix, LAPError::EmptyMatrix);
    assert_ne!(LAPError::EmptyMatrix, LAPError::NonSquareMatrix);
}

#[test]
fn test_lapjv_error_is_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPError::NonSquareMatrix);
}
