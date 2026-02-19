//! Tests for error types: SortedError, KahnError, ConnectedComponentsError,
//! InformationContentError, MonopartiteAlgorithmError.
#![cfg(feature = "std")]

use geometric_traits::traits::{
    KahnError, connected_components::ConnectedComponentsError,
    information_content::InformationContentError,
};

// ============================================================================
// KahnError
// ============================================================================

#[test]
fn test_kahn_error_debug() {
    let err = KahnError::Cycle;
    let debug = format!("{err:?}");
    assert!(debug.contains("Cycle"));
}

#[test]
fn test_kahn_error_clone() {
    let err = KahnError::Cycle;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_kahn_error_partial_eq() {
    assert_eq!(KahnError::Cycle, KahnError::Cycle);
}

// ============================================================================
// ConnectedComponentsError
// ============================================================================

#[test]
fn test_connected_components_error_debug() {
    let err = ConnectedComponentsError::TooManyComponents;
    let debug = format!("{err:?}");
    assert!(debug.contains("TooManyComponents"));
}

#[test]
fn test_connected_components_error_display() {
    let err = ConnectedComponentsError::TooManyComponents;
    let display = format!("{err}");
    assert!(display.contains("too many"), "Display should mention 'too many': {display}");
}

#[test]
fn test_connected_components_error_clone() {
    let err = ConnectedComponentsError::TooManyComponents;
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

// ============================================================================
// InformationContentError
// ============================================================================

#[test]
fn test_ic_error_debug() {
    let err = InformationContentError::NotDag;
    let debug = format!("{err:?}");
    assert!(debug.contains("NotDag"));
}

#[test]
fn test_ic_error_display_not_dag() {
    let err = InformationContentError::NotDag;
    let display = format!("{err}");
    assert!(display.contains("DAG"));
}

#[test]
fn test_ic_error_display_unequal() {
    let err = InformationContentError::UnequalOccurrenceSize { expected: 10, found: 5 };
    let display = format!("{err}");
    assert!(display.contains("10"));
    assert!(display.contains('5'));
}

#[test]
fn test_ic_error_display_sink() {
    let err = InformationContentError::SinkNodeZeroOccurrence;
    let display = format!("{err}");
    assert!(display.contains("Sink"));
}

#[test]
fn test_ic_error_from_kahn_error() {
    let kahn_err = KahnError::Cycle;
    let ic_err: InformationContentError = kahn_err.into();
    assert_eq!(ic_err, InformationContentError::NotDag);
}

#[test]
fn test_ic_error_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(InformationContentError::NotDag);
    assert!(!err.to_string().is_empty());
}

// ============================================================================
// SortedError
// ============================================================================

#[test]
fn test_sorted_error_debug() {
    use geometric_traits::errors::SortedError;
    let err = SortedError::UnsortedEntry(42usize);
    let debug = format!("{err:?}");
    assert!(debug.contains("UnsortedEntry"));
    assert!(debug.contains("42"));
}

#[test]
fn test_sorted_error_display() {
    use geometric_traits::errors::SortedError;
    let err = SortedError::UnsortedEntry(7usize);
    let display = format!("{err}");
    assert!(display.contains('7'));
}

#[test]
fn test_sorted_error_clone_and_eq() {
    use geometric_traits::errors::SortedError;
    let err = SortedError::UnsortedEntry(3usize);
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_sorted_error_ne() {
    use geometric_traits::errors::SortedError;
    assert_ne!(SortedError::UnsortedEntry(1usize), SortedError::UnsortedEntry(2usize));
}
