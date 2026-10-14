//! Tests for the dense GTH stationary-distribution algorithm.
#![cfg(feature = "std")]

mod common;

use common::dense_gth_residual_l1;
use geometric_traits::{impls::VecMatrix2D, prelude::*};

#[test]
fn test_1x1_matrix() {
    let matrix = VecMatrix2D::new(1, 1, vec![1.0]);
    let result = matrix.gth(&GthConfig::default()).unwrap();
    assert_eq!(result.order(), 1);
    assert_eq!(result.stationary(), &[1.0]);
}

#[test]
fn test_2x2_known_stationary_distribution() {
    let matrix = VecMatrix2D::new(2, 2, vec![0.10, 0.90, 0.60, 0.40]);
    let result = matrix.gth(&GthConfig::default()).unwrap();
    assert!((result.stationary()[0] - 0.4).abs() < 1e-12);
    assert!((result.stationary()[1] - 0.6).abs() < 1e-12);
}

#[test]
fn test_3x3_known_stationary_distribution() {
    let matrix = VecMatrix2D::new(3, 3, vec![0.15, 0.55, 0.30, 0.35, 0.20, 0.45, 0.25, 0.50, 0.25]);
    let result = matrix.gth(&GthConfig::default()).unwrap();
    let expected = [0.2631578947368421, 0.39473684210526316, 0.34210526315789475];
    for (got, want) in result.stationary().iter().zip(expected) {
        assert!((got - want).abs() < 1e-12, "expected {want}, got {got}");
    }
}

#[test]
fn test_output_sums_to_one_and_is_nonnegative() {
    let matrix = VecMatrix2D::new(
        4,
        4,
        vec![
            0.55, 0.20, 0.15, 0.10, 0.10, 0.65, 0.10, 0.15, 0.20, 0.10, 0.50, 0.20, 0.25, 0.15,
            0.20, 0.40,
        ],
    );
    let result = matrix.gth(&GthConfig::default()).unwrap();
    let sum: f64 = result.stationary().iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
    assert!(result.stationary().iter().all(|value| *value >= 0.0));
    assert!(dense_gth_residual_l1(&matrix, result.stationary()) < 1e-12);
}

#[test]
fn test_determinism() {
    let matrix = VecMatrix2D::new(3, 3, vec![0.15, 0.55, 0.30, 0.35, 0.20, 0.45, 0.25, 0.50, 0.25]);
    let config = GthConfig::default();
    let first = matrix.gth(&config).unwrap();
    let second = matrix.gth(&config).unwrap();
    assert_eq!(first, second);
}

#[test]
fn test_workspace_path_matches_allocating_path() {
    let matrix = VecMatrix2D::new(
        4,
        4,
        vec![
            0.55, 0.20, 0.15, 0.10, 0.10, 0.65, 0.10, 0.15, 0.20, 0.10, 0.50, 0.20, 0.25, 0.15,
            0.20, 0.40,
        ],
    );
    let config = GthConfig::default();
    let mut workspace = GthWorkspace::with_capacity(4);
    let capacity = workspace.capacity();

    let from_workspace = matrix.gth_with_workspace(&config, &mut workspace).unwrap();
    let from_allocating = matrix.gth(&config).unwrap();

    assert_eq!(from_workspace, from_allocating);
    assert!(workspace.capacity() >= capacity);
}

#[test]
fn test_reducible_matrix_returns_reference_style_distribution() {
    let matrix = VecMatrix2D::new(
        4,
        4,
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.5, 0.0, 0.5, 0.0, //
            0.0, 0.5, 0.0, 0.5,
        ],
    );
    let result = matrix.gth(&GthConfig::default()).unwrap();
    assert_eq!(result.stationary(), &[1.0, 0.0, 0.0, 0.0]);
    assert!(dense_gth_residual_l1(&matrix, result.stationary()) < 1e-12);
}

#[test]
fn test_reducible_matrix_prefers_lowest_closed_class() {
    let matrix = VecMatrix2D::new(
        4,
        4,
        vec![
            0.0, 1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    let result = matrix.gth(&GthConfig::default()).unwrap();
    assert_eq!(result.stationary(), &[0.0, 1.0, 0.0, 0.0]);
    assert!(dense_gth_residual_l1(&matrix, result.stationary()) < 1e-12);
}

#[test]
fn test_reducible_matrix_returns_multistate_closed_class_distribution() {
    let matrix = VecMatrix2D::new(
        4,
        4,
        vec![
            0.0, 1.0, 0.0, 0.0, //
            0.25, 0.75, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    let result = matrix.gth(&GthConfig::default()).unwrap();
    let expected = [0.2, 0.8, 0.0, 0.0];
    for (got, want) in result.stationary().iter().zip(expected) {
        assert!((got - want).abs() < 1e-12, "expected {want}, got {got}");
    }
    assert!(dense_gth_residual_l1(&matrix, result.stationary()) < 1e-12);
}

#[test]
fn test_non_square_matrix_error() {
    let matrix = VecMatrix2D::new(2, 3, vec![0.4, 0.3, 0.3, 0.2, 0.4, 0.4]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::NonSquareMatrix { rows: 2, columns: 3 })));
}

#[test]
fn test_empty_matrix_error() {
    let matrix = VecMatrix2D::<f64>::new(0, 0, vec![]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::EmptyMatrix)));
}

#[test]
fn test_non_finite_nan_error() {
    let matrix = VecMatrix2D::new(2, 2, vec![1.0, 0.0, f64::NAN, 1.0]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::NonFiniteValue { row: 1, column: 0 })));
}

#[test]
fn test_negative_value_error() {
    let matrix = VecMatrix2D::new(2, 2, vec![1.0, 0.0, -0.1, 1.1]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::NegativeValue { row: 1, column: 0 })));
}

#[test]
fn test_non_stochastic_row_error() {
    let matrix = VecMatrix2D::new(2, 2, vec![0.4, 0.5, 0.2, 0.8]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::NonStochasticRow { row: 0, .. })));
}

#[test]
fn test_invalid_tolerance_error() {
    let matrix = VecMatrix2D::new(2, 2, vec![0.4, 0.6, 0.6, 0.4]);
    let result = matrix.gth(&GthConfig { stochastic_tolerance: 0.0 });
    assert!(matches!(result, Err(GthError::InvalidStochasticTolerance)));
}

#[test]
fn test_small_negative_is_rejected_even_with_row_sum_within_tolerance() {
    let matrix = VecMatrix2D::new(2, 2, vec![1.0 + 5e-13, -5e-13, 0.6, 0.4]);
    let result = matrix.gth(&GthConfig::default());
    assert!(matches!(result, Err(GthError::NegativeValue { row: 0, column: 1 })));
}
