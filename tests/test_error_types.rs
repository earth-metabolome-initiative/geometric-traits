//! Tests for error types exposed by traits and builders.
#![cfg(feature = "std")]

use geometric_traits::{
    errors::{builder::vocabulary::VocabularyBuilderError, nodes::NodeError},
    impls::SortedVec,
    prelude::GenericVocabularyBuilder,
    traits::{
        KahnError, VocabularyBuilder, connected_components::ConnectedComponentsError,
        information_content::InformationContentError,
    },
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

// ============================================================================
// VocabularyBuilderError / NodeError
// ============================================================================

#[test]
fn test_vocabulary_builder_error_missing_attribute() {
    // Calling build() without .symbols() triggers MissingAttribute.
    let result: Result<SortedVec<usize>, _> =
        GenericVocabularyBuilder::<std::iter::Empty<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(3)
            .build();
    assert!(result.is_err(), "build() without symbols should fail");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VocabularyBuilderError::MissingAttribute(_)),
        "Expected MissingAttribute error, got: {err:?}"
    );
}

#[test]
fn test_vocabulary_builder_error_number_of_symbols() {
    // Setting expected_number_of_symbols(5) but providing 3 symbols.
    let result: Result<SortedVec<usize>, _> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(5)
        .symbols(vec![(0usize, 0usize), (1, 1), (2, 2)].into_iter())
        .build();
    assert!(result.is_err(), "Mismatched symbol count should fail");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VocabularyBuilderError::NumberOfSymbols { expected: 5, actual: 3 }),
        "Expected NumberOfSymbols error, got: {err:?}"
    );
}

#[test]
fn test_vocabulary_builder_error_missing_attribute_display() {
    let err: VocabularyBuilderError<SortedVec<usize>> =
        VocabularyBuilderError::MissingAttribute("symbols");
    let msg = format!("{err}");
    assert!(
        msg.contains("symbols"),
        "Display message should contain the attribute name, got: {msg}"
    );
}

#[test]
fn test_node_error_unknown_node_id_display() {
    let err: NodeError<SortedVec<usize>> = NodeError::UnknownNodeId(42usize);
    let msg = format!("{err}");
    assert!(msg.contains("42"), "Display message should contain the node id, got: {msg}");
}

#[test]
fn test_node_error_unknown_node_symbol_display() {
    let err: NodeError<SortedVec<usize>> = NodeError::UnknownNodeSymbol(99usize);
    let msg = format!("{err}");
    assert!(msg.contains("99"), "Display message should contain the node symbol, got: {msg}");
}
