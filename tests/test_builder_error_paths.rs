//! Tests for builder error paths in GenericEdgesBuilder and
//! GenericVocabularyBuilder.
#![cfg(feature = "std")]

use std::collections::HashMap;

use geometric_traits::{
    impls::{CSR2D, SortedVec},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

type TestCSR = CSR2D<usize, usize, usize>;
type EdgeIter = std::vec::IntoIter<(usize, usize)>;
type SymbolIter = std::vec::IntoIter<(usize, u32)>;

// ============================================================================
// GenericEdgesBuilder error paths
// ============================================================================

#[test]
fn test_edges_builder_missing_edges() {
    let result = GenericEdgesBuilder::<EdgeIter, TestCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((2, 2))
        .build();
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Missing builder attribute"));
}

#[test]
fn test_edges_builder_number_of_edges_mismatch() {
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(5)
        .expected_shape((3, 3))
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Unexpected number of edges"));
}

#[test]
fn test_edges_builder_duplicate_edge_error() {
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(3)
        .expected_shape((3, 3))
        .edges(vec![(0, 1), (0, 1), (1, 2)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Matrix error"));
}

#[test]
fn test_edges_builder_ignore_duplicates() {
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(2)
        .expected_shape((3, 3))
        .ignore_duplicates()
        .edges(vec![(0, 1), (0, 1), (1, 2)].into_iter())
        .build();
    assert!(result.is_ok());
    let csr = result.unwrap();
    assert_eq!(csr.number_of_defined_values(), 2);
}

#[test]
fn test_edges_builder_no_expected_shape() {
    // Build without expected_shape (uses with_capacity only)
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(2)
        .edges(vec![(0, 1), (1, 0)].into_iter())
        .build();
    assert!(result.is_ok());
}

#[test]
fn test_edges_builder_no_expected_count() {
    // Build without expected_number_of_edges (uses with_shape only)
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_shape((3, 3))
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build();
    assert!(result.is_ok());
}

#[test]
fn test_edges_builder_no_expected_count_or_shape() {
    // Build without either hint
    let result = GenericEdgesBuilder::<_, TestCSR>::default()
        .edges(vec![(0, 1), (1, 2)].into_iter())
        .build();
    assert!(result.is_ok());
}

// ============================================================================
// GenericVocabularyBuilder error paths
// ============================================================================

#[test]
fn test_vocabulary_builder_missing_symbols() {
    let result = GenericVocabularyBuilder::<SymbolIter, SortedVec<u32>>::default()
        .expected_number_of_symbols(3)
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Missing builder attribute"));
}

#[test]
fn test_vocabulary_builder_number_of_symbols_mismatch() {
    let result = GenericVocabularyBuilder::<_, SortedVec<u32>>::default()
        .expected_number_of_symbols(5)
        .symbols(vec![(0, 10), (1, 20)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Unexpected number of symbols"));
}

#[test]
fn test_vocabulary_builder_duplicate_source_hashmap() {
    // HashMap gives RepeatedSourceSymbol for duplicate keys
    let result = GenericVocabularyBuilder::<_, HashMap<i32, i32>>::default()
        .expected_number_of_symbols(2)
        .symbols(vec![(0, 10), (0, 20)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Repeated source symbol"));
}

#[test]
fn test_vocabulary_builder_duplicate_destination_hashmap() {
    let result = GenericVocabularyBuilder::<_, HashMap<i32, i32>>::default()
        .expected_number_of_symbols(2)
        .symbols(vec![(0, 10), (1, 10)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Repeated destination symbol"));
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_hashmap() {
    let result = GenericVocabularyBuilder::<_, HashMap<i32, i32>>::default()
        .expected_number_of_symbols(1)
        .ignore_duplicates()
        .symbols(vec![(0, 10), (0, 20)].into_iter())
        .build();
    assert!(result.is_ok());
    let map = result.unwrap();
    assert_eq!(Vocabulary::len(&map), 1);
}

#[test]
fn test_vocabulary_builder_sparse_source_sorted_vec() {
    // SortedVec requires sequential sources (0, 1, 2, ...)
    let result = GenericVocabularyBuilder::<_, SortedVec<u32>>::default()
        .expected_number_of_symbols(2)
        .symbols(vec![(0, 10), (0, 20)].into_iter())
        .build();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Sparse source node"));
}

#[test]
fn test_vocabulary_builder_no_expected_count() {
    // Build without expected_number_of_symbols
    let result = GenericVocabularyBuilder::<_, SortedVec<u32>>::default()
        .symbols(vec![(0, 10), (1, 20)].into_iter())
        .build();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

// ============================================================================
// EdgesBuilder / VocabularyBuilder trait accessor tests
// ============================================================================

#[test]
fn test_edges_builder_get_accessors() {
    let builder = GenericEdgesBuilder::<EdgeIter, TestCSR>::default()
        .expected_number_of_edges(10)
        .expected_shape((5, 5));

    assert_eq!(builder.get_expected_number_of_edges(), Some(10));
    assert_eq!(builder.get_expected_shape(), Some((5, 5)));
}

#[test]
fn test_edges_builder_get_accessors_unset() {
    let builder = GenericEdgesBuilder::<EdgeIter, TestCSR>::default();

    assert_eq!(builder.get_expected_number_of_edges(), None);
    assert_eq!(builder.get_expected_shape(), None);
    assert!(!builder.should_ignore_duplicates());
}

#[test]
fn test_vocabulary_builder_get_accessors() {
    let builder = GenericVocabularyBuilder::<SymbolIter, SortedVec<u32>>::default()
        .expected_number_of_symbols(5);

    assert_eq!(builder.get_expected_number_of_symbols(), Some(5));
    assert!(!builder.should_ignore_duplicates());
}
