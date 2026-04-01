//! Consolidated tests for error types exposed by traits and builders.
#![cfg(feature = "std")]

use geometric_traits::{
    errors::{
        bipartite_graph_error::algorithms::BipartiteAlgorithmError,
        builder::vocabulary::VocabularyBuilderError,
        monopartite_graph_error::algorithms::MonopartiteAlgorithmError, nodes::NodeError,
        sorted_error::SortedError,
    },
    impls::SortedVec,
    prelude::GenericVocabularyBuilder,
    traits::{
        KahnError, LAPError, VocabularyBuilder, connected_components::ConnectedComponentsError,
        information_content::InformationContentError,
    },
};

#[test]
fn test_lap_error_display_variants() {
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
    ];

    for (error, expected_message) in cases {
        assert_eq!(format!("{error}"), expected_message);
    }
}

#[test]
fn test_lap_error_traits() {
    let error = LAPError::EmptyMatrix;
    assert!(format!("{:?}", LAPError::NonSquareMatrix).contains("NonSquareMatrix"));
    assert_eq!(error, error.clone());
}

#[test]
fn test_connected_components_error_traits() {
    let error = ConnectedComponentsError::TooManyComponents;
    assert!(format!("{error}").contains("too many connected components"));
    assert!(format!("{error:?}").contains("TooManyComponents"));
    assert_eq!(error, error.clone());
}

#[test]
fn test_sorted_error_traits() {
    let error = SortedError::UnsortedEntry(42usize);
    assert!(format!("{error}").contains("42"));
    assert!(format!("{error:?}").contains("UnsortedEntry"));
    assert_eq!(error, error.clone());
    assert_ne!(SortedError::UnsortedEntry(1usize), SortedError::UnsortedEntry(2usize));
}

#[test]
fn test_bipartite_algorithm_error_traits() {
    let error = BipartiteAlgorithmError::LAPMOD(LAPError::NonSquareMatrix);
    assert!(format!("{error}").contains("not square"));
    assert!(
        format!("{:?}", BipartiteAlgorithmError::LAPMOD(LAPError::EmptyMatrix)).contains("LAPMOD")
    );
    assert_eq!(
        BipartiteAlgorithmError::LAPMOD(LAPError::ZeroValues),
        BipartiteAlgorithmError::LAPMOD(LAPError::ZeroValues).clone()
    );
}

#[test]
fn test_monopartite_algorithm_error_traits() {
    let error = MonopartiteAlgorithmError::ConnectedComponentsError(
        ConnectedComponentsError::TooManyComponents,
    );
    assert!(format!("{error}").contains("too many connected components"));
    assert!(format!("{error:?}").contains("ConnectedComponentsError"));
    assert_eq!(error, error.clone());
}

#[test]
fn test_kahn_error_traits() {
    let error = KahnError::Cycle;
    assert!(format!("{error:?}").contains("Cycle"));
    assert_eq!(error, error.clone());
    assert_eq!(KahnError::Cycle, KahnError::Cycle);
}

#[test]
fn test_information_content_error_display_and_conversions() {
    let not_dag = InformationContentError::NotDag;
    assert!(format!("{not_dag:?}").contains("NotDag"));
    assert!(format!("{not_dag}").contains("DAG"));
    assert_eq!(InformationContentError::NotDag, KahnError::Cycle.into());

    let unequal = InformationContentError::UnequalOccurrenceSize { expected: 10, found: 5 };
    let unequal_display = format!("{unequal}");
    assert!(unequal_display.contains("10"));
    assert!(unequal_display.contains('5'));

    let sink = InformationContentError::SinkNodeZeroOccurrence;
    assert!(format!("{sink}").contains("Sink"));

    let dyn_error: Box<dyn std::error::Error> = Box::new(InformationContentError::NotDag);
    assert!(!dyn_error.to_string().is_empty());
}

#[test]
fn test_vocabulary_builder_missing_symbols_error() {
    let result: Result<SortedVec<usize>, _> =
        GenericVocabularyBuilder::<std::iter::Empty<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(3)
            .build();
    assert!(result.is_err(), "build() without symbols should fail");
    assert!(matches!(result.unwrap_err(), VocabularyBuilderError::MissingAttribute(_)));
}

#[test]
fn test_vocabulary_builder_number_of_symbols_error() {
    let result: Result<SortedVec<usize>, _> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(5)
        .symbols(vec![(0usize, 0usize), (1, 1), (2, 2)].into_iter())
        .build();
    assert!(result.is_err(), "mismatched symbol count should fail");
    assert!(matches!(
        result.unwrap_err(),
        VocabularyBuilderError::NumberOfSymbols { expected: 5, actual: 3 }
    ));
}

#[test]
fn test_vocabulary_builder_missing_attribute_display() {
    let error: VocabularyBuilderError<SortedVec<usize>> =
        VocabularyBuilderError::MissingAttribute("symbols");
    assert!(format!("{error}").contains("symbols"));
}

#[test]
fn test_node_error_display_variants() {
    let unknown_id: NodeError<SortedVec<usize>> = NodeError::UnknownNodeId(42usize);
    assert!(format!("{unknown_id}").contains("42"));

    let unknown_symbol: NodeError<SortedVec<usize>> = NodeError::UnknownNodeSymbol(99usize);
    assert!(format!("{unknown_symbol}").contains("99"));
}
