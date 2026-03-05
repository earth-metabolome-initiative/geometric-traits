//! Tests for LAPError Display impl.
#![cfg(feature = "std")]

use std::error::Error;

use geometric_traits::traits::LAPError;

#[test]
fn test_lap_error_variant_display_messages() {
    let cases = [
        (
            LAPError::NonFractionalValueTypeUnsupported,
            "The matrix value type is non-fractional and is not supported by LAP algorithms.",
        ),
        (LAPError::NonSquareMatrix, "The matrix is not square."),
        (LAPError::EmptyMatrix, "The matrix is empty."),
        (LAPError::ZeroValues, "The matrix contains zero values."),
        (LAPError::NegativeValues, "The matrix contains negative values."),
        (LAPError::NonFiniteValues, "The matrix contains non-finite values."),
        (LAPError::ValueTooLarge, "The matrix contains a value larger than the maximum cost."),
        (LAPError::MaximalCostNotFinite, "The provided maximal cost is not a finite number."),
        (LAPError::MaximalCostNotPositive, "The provided maximal cost is not a positive number."),
        (LAPError::PaddingValueNotFinite, "The provided padding value is not a finite number."),
        (LAPError::PaddingValueNotPositive, "The provided padding value is not a positive number."),
        (
            LAPError::PaddingCostTooSmall,
            "The padding cost is too small: padding_cost / 2 must be strictly greater than the maximum sparse value.",
        ),
        (
            LAPError::ExpandedMatrixBuildFailed,
            "Failed to build the expanded sparse matrix from the provided sparse structure.",
        ),
        (
            LAPError::IndexConversionFailed,
            "Internal index conversion failed while processing the sparse wrapper.",
        ),
        (
            LAPError::InfeasibleAssignment,
            "The sparse structure has no perfect matching (infeasible assignment).",
        ),
    ];

    for (error, expected) in cases {
        assert_eq!(format!("{error}"), expected);
    }
}

#[test]
fn test_lap_error_is_std_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPError::NonSquareMatrix);
}

#[test]
fn test_lap_error_debug_clone_eq() {
    let error = LAPError::EmptyMatrix;
    let cloned = error.clone();
    assert_eq!(error, cloned);
    assert!(format!("{error:?}").contains("EmptyMatrix"));
}
