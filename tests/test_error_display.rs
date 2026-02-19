//! Tests for error Display impls: LAPJVError, ConnectedComponentsError,
//! SortedError, BipartiteAlgorithmError, MonopartiteAlgorithmError,
//! and error From conversions.
#![cfg(feature = "std")]

use geometric_traits::{
    errors::{
        bipartite_graph_error::algorithms::BipartiteAlgorithmError,
        monopartite_graph_error::algorithms::MonopartiteAlgorithmError, sorted_error::SortedError,
    },
    traits::{LAPJVError, connected_components::ConnectedComponentsError},
};

// ============================================================================
// LAPJVError Display (10 variants, all manual Display impl)
// ============================================================================

#[test]
fn test_lapjv_non_square_matrix() {
    let e = LAPJVError::NonSquareMatrix;
    assert_eq!(format!("{e}"), "The matrix is not square.");
}

#[test]
fn test_lapjv_empty_matrix() {
    let e = LAPJVError::EmptyMatrix;
    assert_eq!(format!("{e}"), "The matrix is empty.");
}

#[test]
fn test_lapjv_zero_values() {
    let e = LAPJVError::ZeroValues;
    assert_eq!(format!("{e}"), "The matrix contains zero values.");
}

#[test]
fn test_lapjv_negative_values() {
    let e = LAPJVError::NegativeValues;
    assert_eq!(format!("{e}"), "The matrix contains negative values.");
}

#[test]
fn test_lapjv_non_finite_values() {
    let e = LAPJVError::NonFiniteValues;
    assert_eq!(format!("{e}"), "The matrix contains non-finite values.");
}

#[test]
fn test_lapjv_value_too_large() {
    let e = LAPJVError::ValueTooLarge;
    assert_eq!(format!("{e}"), "The matrix contains a value larger than the maximum cost.");
}

#[test]
fn test_lapjv_maximal_cost_not_finite() {
    let e = LAPJVError::MaximalCostNotFinite;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a finite number.");
}

#[test]
fn test_lapjv_maximal_cost_not_positive() {
    let e = LAPJVError::MaximalCostNotPositive;
    assert_eq!(format!("{e}"), "The provided maximal cost is not a positive number.");
}

#[test]
fn test_lapjv_padding_value_not_finite() {
    let e = LAPJVError::PaddingValueNotFinite;
    assert_eq!(format!("{e}"), "The provided padding value is not a finite number.");
}

#[test]
fn test_lapjv_padding_value_not_positive() {
    let e = LAPJVError::PaddingValueNotPositive;
    assert_eq!(format!("{e}"), "The provided padding value is not a positive number.");
}

#[test]
fn test_lapjv_debug() {
    let e = LAPJVError::NonSquareMatrix;
    let debug = format!("{e:?}");
    assert!(debug.contains("NonSquareMatrix"));
}

#[test]
fn test_lapjv_clone_eq() {
    let e = LAPJVError::EmptyMatrix;
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
    let inner = LAPJVError::NonSquareMatrix;
    let e = BipartiteAlgorithmError::LAPMOD(inner);
    let msg = format!("{e}");
    assert!(msg.contains("not square"));
}

#[test]
fn test_bipartite_algorithm_error_debug() {
    let inner = LAPJVError::EmptyMatrix;
    let e = BipartiteAlgorithmError::LAPMOD(inner);
    let debug = format!("{e:?}");
    assert!(debug.contains("LAPMOD"));
}

#[test]
fn test_bipartite_algorithm_error_clone_eq() {
    let e = BipartiteAlgorithmError::LAPMOD(LAPJVError::ZeroValues);
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
