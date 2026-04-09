//! Consolidated tests for error types exposed by traits and builders.
#![cfg(feature = "std")]

use std::error::Error;

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
        BiconnectedComponentsError, K23HomeomorphError, KahnError, LAPError, OuterplanarityError,
        PlanarityError, VocabularyBuilder, connected_components::ConnectedComponentsError,
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

    for (error, expected_message) in cases {
        assert_eq!(format!("{error}"), expected_message);
    }
}

#[test]
fn test_lap_error_traits() {
    let error = LAPError::EmptyMatrix;
    assert!(format!("{:?}", LAPError::NonSquareMatrix).contains("NonSquareMatrix"));
    assert_eq!(error, error.clone());
    assert_ne!(LAPError::EmptyMatrix, LAPError::NonSquareMatrix);
}

#[test]
fn test_lap_error_is_std_error() {
    fn check_is_error<E: Error>(_: E) {}
    check_is_error(LAPError::NonSquareMatrix);
}

#[test]
fn test_connected_components_error_traits() {
    let error = ConnectedComponentsError::TooManyComponents;
    assert!(format!("{error}").contains("too many connected components"));
    assert!(format!("{error:?}").contains("TooManyComponents"));
    assert_eq!(error, error.clone());
}

#[test]
fn test_biconnected_components_error_traits() {
    let error = BiconnectedComponentsError::SelfLoopsUnsupported;
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("SelfLoopsUnsupported"));
    assert_eq!(error, error.clone());
    assert_ne!(
        BiconnectedComponentsError::SelfLoopsUnsupported,
        BiconnectedComponentsError::ParallelEdgesUnsupported
    );
}

#[test]
fn test_planarity_error_traits() {
    let error = PlanarityError::SelfLoopsUnsupported;
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("SelfLoopsUnsupported"));
    assert_eq!(error, error.clone());
    assert_ne!(PlanarityError::SelfLoopsUnsupported, PlanarityError::ParallelEdgesUnsupported);

    let malformed = PlanarityError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 };
    assert!(format!("{malformed}").contains("endpoint 9"));
    assert!(format!("{malformed}").contains("node_count=4"));
    assert_eq!(malformed, malformed.clone());
}

#[test]
fn test_outerplanarity_error_traits() {
    let error = OuterplanarityError::SelfLoopsUnsupported;
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("SelfLoopsUnsupported"));
    assert_eq!(error, error.clone());
    assert_ne!(
        OuterplanarityError::SelfLoopsUnsupported,
        OuterplanarityError::ParallelEdgesUnsupported
    );

    let malformed = OuterplanarityError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 };
    assert!(format!("{malformed}").contains("endpoint 9"));
    assert!(format!("{malformed}").contains("node_count=4"));
    assert_eq!(malformed, malformed.clone());
}

#[test]
fn test_k23_homeomorph_error_traits() {
    let error = K23HomeomorphError::SelfLoopsUnsupported;
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("SelfLoopsUnsupported"));
    assert_eq!(error, error.clone());
    assert_ne!(
        K23HomeomorphError::SelfLoopsUnsupported,
        K23HomeomorphError::ParallelEdgesUnsupported
    );

    let malformed = K23HomeomorphError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 };
    assert!(format!("{malformed}").contains("endpoint 9"));
    assert!(format!("{malformed}").contains("node_count=4"));
    assert_eq!(malformed, malformed.clone());
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

    let error = MonopartiteAlgorithmError::BiconnectedComponentsError(
        BiconnectedComponentsError::SelfLoopsUnsupported,
    );
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("BiconnectedComponentsError"));
    assert_eq!(error, error.clone());

    let error = MonopartiteAlgorithmError::PlanarityError(PlanarityError::SelfLoopsUnsupported);
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("PlanarityError"));
    assert_eq!(error, error.clone());

    let error =
        MonopartiteAlgorithmError::OuterplanarityError(OuterplanarityError::SelfLoopsUnsupported);
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("OuterplanarityError"));
    assert_eq!(error, error.clone());

    let error =
        MonopartiteAlgorithmError::K23HomeomorphError(K23HomeomorphError::SelfLoopsUnsupported);
    assert!(format!("{error}").contains("self-loops"));
    assert!(format!("{error:?}").contains("K23HomeomorphError"));
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
