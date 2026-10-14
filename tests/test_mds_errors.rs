//! Display and trait tests for MdsError variants.
#![cfg(feature = "std")]

use std::error::Error;

use geometric_traits::prelude::MdsError;

#[test]
fn test_non_square_matrix() {
    let err = MdsError::NonSquareMatrix { rows: 3, columns: 5 };
    assert_eq!(
        format!("{err}"),
        "The distance matrix must be square, but has 3 rows and 5 columns."
    );
}

#[test]
fn test_non_symmetric_matrix() {
    let err = MdsError::NonSymmetricMatrix { row: 1, column: 2 };
    assert_eq!(
        format!("{err}"),
        "The distance matrix is not symmetric: value at (1, 2) differs from (2, 1)."
    );
}

#[test]
fn test_non_finite_value() {
    let err = MdsError::NonFiniteValue { row: 0, column: 3 };
    assert_eq!(format!("{err}"), "Found a non-finite value at (0, 3).");
}

#[test]
fn test_negative_distance() {
    let err = MdsError::NegativeDistance { row: 2, column: 4 };
    assert_eq!(format!("{err}"), "Found a negative distance at (2, 4).");
}

#[test]
fn test_empty_matrix() {
    let err = MdsError::EmptyMatrix;
    assert_eq!(format!("{err}"), "The distance matrix is empty.");
}

#[test]
fn test_invalid_dimensions() {
    let err = MdsError::InvalidDimensions(0);
    assert_eq!(
        format!("{err}"),
        "The number of embedding dimensions must be at least 1, but got 0."
    );
}

#[test]
fn test_dimensions_exceed_points() {
    let err = MdsError::DimensionsExceedPoints { dimensions: 5, num_points: 3 };
    assert_eq!(
        format!("{err}"),
        "Requested 5 embedding dimensions, but the matrix has only 3 points."
    );
}

#[test]
fn test_distance_too_large() {
    let err = MdsError::DistanceTooLarge(1e155);
    let msg = format!("{err}");
    assert!(msg.contains("too large"), "expected 'too large' in: {msg}");
    assert!(msg.contains("overflow"), "expected 'overflow' in: {msg}");
}

#[test]
fn test_diagonal_not_zero() {
    let err = MdsError::DiagonalNotZero { index: 2, value: 0.5 };
    assert_eq!(format!("{err}"), "The diagonal entry at (2, 2) must be zero, but is 0.5.");
}

#[test]
fn test_too_few_points() {
    let err = MdsError::TooFewPoints(1);
    assert_eq!(format!("{err}"), "Need at least 2 points for MDS, but got 1.");
}

#[test]
fn test_jacobi_error_wrapping() {
    let inner = geometric_traits::prelude::JacobiError::InvalidTolerance;
    let err = MdsError::from(inner.clone());
    assert!(matches!(err, MdsError::JacobiError(_)));
    assert_eq!(format!("{err}"), format!("{inner}"));
}

#[test]
fn test_mds_error_debug() {
    let err = MdsError::EmptyMatrix;
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("EmptyMatrix"));
}

#[test]
fn test_mds_error_clone() {
    let err = MdsError::NonSquareMatrix { rows: 2, columns: 3 };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_mds_error_is_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(MdsError::EmptyMatrix);
}
