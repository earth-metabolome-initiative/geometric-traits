//! Tests for error Display impls: LAPError, ConnectedComponentsError,
//! SortedError, BipartiteAlgorithmError, MonopartiteAlgorithmError,
//! and error From conversions.
#![cfg(feature = "std")]

use geometric_traits::{
    errors::{
        bipartite_graph_error::algorithms::BipartiteAlgorithmError,
        monopartite_graph_error::algorithms::MonopartiteAlgorithmError, sorted_error::SortedError,
    },
    traits::{LAPError, connected_components::ConnectedComponentsError},
};

// ============================================================================
// LAPError Display
// ============================================================================

#[test]
fn test_lapjv_non_fractional_value_type_unsupported() {
    let e = LAPError::NonFractionalValueTypeUnsupported;
    assert_eq!(
        format!("{e}"),
        "The matrix value type is non-fractional and is not supported by LAP algorithms."
    );
}

#[test]
fn test_lapjv_non_square_matrix() {
    let e = LAPError::NonSquareMatrix;
    assert_eq!(format!("{e}"), "The matrix is not square.");
}

#[test]
fn test_lapjv_empty_matrix() {
    let e = LAPError::EmptyMatrix;
    assert_eq!(format!("{e}"), "The matrix is empty.");
}

#[test]
fn test_lapjv_zero_values() {
    let e = LAPError::ZeroValues;
    assert_eq!(format!("{e}"), "The matrix contains zero values.");
}

#[test]
fn test_lapjv_negative_values() {
    let e = LAPError::NegativeValues;
    assert_eq!(format!("{e}"), "The matrix contains negative values.");
}

#[test]
fn test_lapjv_non_finite_values() {
    let e = LAPError::NonFiniteValues;
    assert_eq!(format!("{e}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lapjv_value_too_large() {
    let e = LAPError::ValueTooLarge;
    assert_eq!(format!("{e}"), "The matrix contains a value larger than the maximum cost.");
}

#[test]
fn test_lapjv_maximal_cost_not_finite() {
    let e = LAPError::MaximalCostNotFinite;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lapjv_maximal_cost_not_positive() {
    let e = LAPError::MaximalCostNotPositive;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lapjv_padding_value_not_finite() {
    let e = LAPError::PaddingValueNotFinite;
    assert_eq!(format!("{e}"), "The provided padding value is not a finite number.");
}

#[test]
fn test_lapjv_padding_value_not_positive() {
    let e = LAPError::PaddingValueNotPositive;
    assert_eq!(format!("{e}"), "The provided padding value is not a positive number.");
}

#[test]
fn test_lapjv_debug() {
    let e = LAPError::NonSquareMatrix;
    let debug = format!("{e:?}");
    assert!(debug.contains("NonSquareMatrix"));
}

#[test]
fn test_lapjv_clone_eq() {
    let e = LAPError::EmptyMatrix;
    let e2 = e.clone();
    assert_eq!(e, e2);
}

// ============================================================================
// ConnectedComponentsError Display
// ============================================================================

#[test]
fn test_connected_components_error_display() {
    let e = ConnectedComponentsError::TooManyComponents;
    let msg = format!("{e}");
    assert!(msg.contains("too many connected components"));
}

#[test]
fn test_connected_components_error_debug() {
    let e = ConnectedComponentsError::TooManyComponents;
    let debug = format!("{e:?}");
    assert!(debug.contains("TooManyComponents"));
}

// ============================================================================
// SortedError Display
// ============================================================================

#[test]
fn test_sorted_error_display() {
    let e = SortedError::UnsortedEntry(42_usize);
    let msg = format!("{e}");
    assert!(msg.contains("Unsorted entry"));
    assert!(msg.contains("42"));
}

#[test]
fn test_sorted_error_debug() {
    let e = SortedError::UnsortedEntry(42_usize);
    let debug = format!("{e:?}");
    assert!(debug.contains("UnsortedEntry"));
}

#[test]
fn test_sorted_error_clone_eq() {
    let e = SortedError::UnsortedEntry(42_usize);
    let e2 = e.clone();
    assert_eq!(e, e2);
}

// ============================================================================
// BipartiteAlgorithmError Display + From
// ============================================================================

#[test]
fn test_bipartite_algorithm_error_display() {
    let inner = LAPError::NonSquareMatrix;
    let e = BipartiteAlgorithmError::LAPMOD(inner);
    let msg = format!("{e}");
    assert!(msg.contains("not square"));
}

#[test]
fn test_bipartite_algorithm_error_debug() {
    let inner = LAPError::EmptyMatrix;
    let e = BipartiteAlgorithmError::LAPMOD(inner);
    let debug = format!("{e:?}");
    assert!(debug.contains("LAPMOD"));
}

#[test]
fn test_bipartite_algorithm_error_clone_eq() {
    let e = BipartiteAlgorithmError::LAPMOD(LAPError::ZeroValues);
    let e2 = e.clone();
    assert_eq!(e, e2);
}

// ============================================================================
// MonopartiteAlgorithmError Display
// ============================================================================

#[test]
fn test_monopartite_algorithm_error_display() {
    let inner = ConnectedComponentsError::TooManyComponents;
    let e = MonopartiteAlgorithmError::ConnectedComponentsError(inner);
    let msg = format!("{e}");
    assert!(msg.contains("too many connected components"));
}

#[test]
fn test_monopartite_algorithm_error_debug() {
    let inner = ConnectedComponentsError::TooManyComponents;
    let e = MonopartiteAlgorithmError::ConnectedComponentsError(inner);
    let debug = format!("{e:?}");
    assert!(debug.contains("ConnectedComponentsError"));
}

#[test]
fn test_monopartite_algorithm_error_clone_eq() {
    let e = MonopartiteAlgorithmError::ConnectedComponentsError(
        ConnectedComponentsError::TooManyComponents,
    );
    let e2 = e.clone();
    assert_eq!(e, e2);
}
