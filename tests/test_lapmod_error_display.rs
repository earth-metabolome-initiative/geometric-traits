//! Tests for LAPMODError and LAPError Display impls and From conversions.
#![cfg(feature = "std")]

use std::error::Error;

use geometric_traits::traits::{LAPError, LAPJVError, LAPMODError};

// ============================================================================
// LAPMODError Display
// ============================================================================

#[test]
fn test_lapmod_error_non_square_matrix() {
    let e = LAPMODError::NonSquareMatrix;
    assert_eq!(format!("{e}"), "The matrix is not square.");
}

#[test]
fn test_lapmod_error_empty_matrix() {
    let e = LAPMODError::EmptyMatrix;
    assert_eq!(format!("{e}"), "The matrix is empty.");
}

#[test]
fn test_lapmod_error_zero_values() {
    let e = LAPMODError::ZeroValues;
    assert_eq!(format!("{e}"), "The matrix contains zero values.");
}

#[test]
fn test_lapmod_error_negative_values() {
    let e = LAPMODError::NegativeValues;
    assert_eq!(format!("{e}"), "The matrix contains negative values.");
}

#[test]
fn test_lapmod_error_non_finite_values() {
    let e = LAPMODError::NonFiniteValues;
    assert_eq!(format!("{e}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lapmod_error_value_too_large() {
    let e = LAPMODError::ValueTooLarge;
    assert_eq!(
        format!("{e}"),
        "The matrix contains a value larger than or equal to the maximum cost."
    );
}

#[test]
fn test_lapmod_error_maximal_cost_not_finite() {
    let e = LAPMODError::MaximalCostNotFinite;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lapmod_error_maximal_cost_not_positive() {
    let e = LAPMODError::MaximalCostNotPositive;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lapmod_error_infeasible_assignment() {
    let e = LAPMODError::InfeasibleAssignment;
    assert_eq!(
        format!("{e}"),
        "The sparse structure has no perfect matching (infeasible assignment)."
    );
}

#[test]
fn test_lapmod_error_is_std_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPMODError::NonSquareMatrix);
}

#[test]
fn test_lapmod_error_debug_clone_eq() {
    let e = LAPMODError::EmptyMatrix;
    let e2 = e.clone();
    assert_eq!(e, e2);
    assert!(format!("{e:?}").contains("EmptyMatrix"));
}

// ============================================================================
// LAPError Display
// ============================================================================

#[test]
fn test_lap_error_non_square_matrix() {
    let e = LAPError::NonSquareMatrix;
    assert_eq!(format!("{e}"), "The matrix is not square.");
}

#[test]
fn test_lap_error_empty_matrix() {
    let e = LAPError::EmptyMatrix;
    assert_eq!(format!("{e}"), "The matrix is empty.");
}

#[test]
fn test_lap_error_zero_values() {
    let e = LAPError::ZeroValues;
    assert_eq!(format!("{e}"), "The matrix contains zero values.");
}

#[test]
fn test_lap_error_negative_values() {
    let e = LAPError::NegativeValues;
    assert_eq!(format!("{e}"), "The matrix contains negative values.");
}

#[test]
fn test_lap_error_non_finite_values() {
    let e = LAPError::NonFiniteValues;
    assert_eq!(format!("{e}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lap_error_value_too_large() {
    let e = LAPError::ValueTooLarge;
    assert_eq!(format!("{e}"), "The matrix contains a value larger than the maximum cost.");
}

#[test]
fn test_lap_error_maximal_cost_not_finite() {
    let e = LAPError::MaximalCostNotFinite;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lap_error_maximal_cost_not_positive() {
    let e = LAPError::MaximalCostNotPositive;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lap_error_padding_value_not_finite() {
    let e = LAPError::PaddingValueNotFinite;
    assert_eq!(format!("{e}"), "The provided padding value is not a finite number.");
}

#[test]
fn test_lap_error_padding_value_not_positive() {
    let e = LAPError::PaddingValueNotPositive;
    assert_eq!(format!("{e}"), "The provided padding value is not a positive number.");
}

#[test]
fn test_lap_error_infeasible_assignment() {
    let e = LAPError::InfeasibleAssignment;
    assert_eq!(
        format!("{e}"),
        "The sparse structure has no perfect matching (infeasible assignment)."
    );
}

#[test]
fn test_lap_error_is_std_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPError::NonSquareMatrix);
}

#[test]
fn test_lap_error_debug_clone_eq() {
    let e = LAPError::EmptyMatrix;
    let e2 = e.clone();
    assert_eq!(e, e2);
    assert!(format!("{e:?}").contains("EmptyMatrix"));
}

// ============================================================================
// From<LAPMODError> for LAPError
// ============================================================================

#[test]
fn test_from_lapmod_to_lap_error_all_variants() {
    assert_eq!(LAPError::from(LAPMODError::NonSquareMatrix), LAPError::NonSquareMatrix);
    assert_eq!(LAPError::from(LAPMODError::EmptyMatrix), LAPError::EmptyMatrix);
    assert_eq!(LAPError::from(LAPMODError::ZeroValues), LAPError::ZeroValues);
    assert_eq!(LAPError::from(LAPMODError::NegativeValues), LAPError::NegativeValues);
    assert_eq!(LAPError::from(LAPMODError::NonFiniteValues), LAPError::NonFiniteValues);
    assert_eq!(LAPError::from(LAPMODError::ValueTooLarge), LAPError::ValueTooLarge);
    assert_eq!(LAPError::from(LAPMODError::MaximalCostNotFinite), LAPError::MaximalCostNotFinite);
    assert_eq!(
        LAPError::from(LAPMODError::MaximalCostNotPositive),
        LAPError::MaximalCostNotPositive
    );
    assert_eq!(LAPError::from(LAPMODError::InfeasibleAssignment), LAPError::InfeasibleAssignment);
}

// ============================================================================
// From<LAPJVError> for LAPError
// ============================================================================

#[test]
fn test_from_lapjv_to_lap_error_all_variants() {
    assert_eq!(LAPError::from(LAPJVError::NonSquareMatrix), LAPError::NonSquareMatrix);
    assert_eq!(LAPError::from(LAPJVError::EmptyMatrix), LAPError::EmptyMatrix);
    assert_eq!(LAPError::from(LAPJVError::ZeroValues), LAPError::ZeroValues);
    assert_eq!(LAPError::from(LAPJVError::NegativeValues), LAPError::NegativeValues);
    assert_eq!(LAPError::from(LAPJVError::NonFiniteValues), LAPError::NonFiniteValues);
    assert_eq!(LAPError::from(LAPJVError::ValueTooLarge), LAPError::ValueTooLarge);
    assert_eq!(LAPError::from(LAPJVError::MaximalCostNotFinite), LAPError::MaximalCostNotFinite);
    assert_eq!(
        LAPError::from(LAPJVError::MaximalCostNotPositive),
        LAPError::MaximalCostNotPositive
    );
    assert_eq!(LAPError::from(LAPJVError::PaddingValueNotFinite), LAPError::PaddingValueNotFinite);
    assert_eq!(
        LAPError::from(LAPJVError::PaddingValueNotPositive),
        LAPError::PaddingValueNotPositive
    );
}
