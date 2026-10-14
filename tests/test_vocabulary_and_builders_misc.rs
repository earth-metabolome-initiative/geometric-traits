//! Miscellaneous vocabulary and builder coverage by domain.
#![cfg(feature = "std")]

use std::collections::HashMap;

use geometric_traits::{
    impls::SortedVec,
    prelude::*,
    traits::{BidirectionalVocabularyRef, EdgesBuilder, VocabularyBuilder, VocabularyRef},
};

#[test]
fn test_vocabulary_ref_delegation_through_reference() {
    let sv: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0usize, 10u8), (1, 20), (2, 30)].into_iter())
        .build()
        .unwrap();

    let sv_ref: &SortedVec<u8> = &sv;
    assert_eq!(sv_ref.convert_ref(&0), Some(&10u8));
    assert_eq!(sv_ref.convert_ref(&1), Some(&20u8));
    assert_eq!(sv_ref.convert_ref(&99), None);

    let dest_refs: Vec<&u8> = sv_ref.destination_refs().collect();
    assert_eq!(dest_refs.len(), 3);

    let map: HashMap<usize, u8> = HashMap::from([(0, 10), (1, 20), (2, 30)]);
    let map_ref: &HashMap<usize, u8> = &map;
    assert_eq!(map_ref.invert_ref(&10u8), Some(&0usize));
    assert_eq!(map_ref.invert_ref(&20u8), Some(&1usize));
    assert_eq!(map_ref.invert_ref(&99u8), None);
    let src_refs: Vec<&usize> = map_ref.source_refs().collect();
    assert_eq!(src_refs.len(), 3);
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_with_data() {
    let map: HashMap<usize, u8> = GenericVocabularyBuilder::default()
        .ignore_duplicates()
        .symbols(vec![(0usize, 10u8), (0, 20), (1, 30)].into_iter())
        .build()
        .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&0), Some(&10));
    assert_eq!(map.get(&1), Some(&30));
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_dest() {
    let map: HashMap<usize, u8> = GenericVocabularyBuilder::default()
        .ignore_duplicates()
        .symbols(vec![(0usize, 10u8), (1, 10), (2, 30)].into_iter())
        .build()
        .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&0), Some(&10));
    assert_eq!(map.get(&2), Some(&30));
    assert_eq!(map.get(&1), None);
}

#[test]
fn test_vocabulary_builder_missing_symbols() {
    let result =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(3)
            .build();
    assert!(result.is_err());
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_flag() {
    let builder =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default();
    assert!(!builder.should_ignore_duplicates());

    let builder = builder.ignore_duplicates();
    assert!(builder.should_ignore_duplicates());
}

#[test]
fn test_vocabulary_builder_wrong_count() {
    let result =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(5)
            .symbols(vec![(0, 10), (1, 20)].into_iter())
            .build();
    assert!(result.is_err());
}

#[test]
fn test_undirected_edges_builder_delegation() {
    let builder: UndiEdgesBuilder<std::vec::IntoIter<(usize, usize)>> = UndiEdgesBuilder::default();
    assert!(builder.get_expected_number_of_edges().is_none());
    assert!(!builder.should_ignore_duplicates());
    assert!(builder.get_expected_shape().is_none());

    let builder = builder.expected_number_of_edges(5).expected_shape(4).ignore_duplicates();

    assert_eq!(builder.get_expected_number_of_edges(), Some(5));
    assert!(builder.should_ignore_duplicates());
    assert_eq!(builder.get_expected_shape(), Some(4));
}

#[test]
fn test_sorted_vec_duplicate_source_error() {
    let result = GenericVocabularyBuilder::<_, SortedVec<usize>>::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 10), (1, 20), (1, 30)].into_iter())
        .build();
    assert!(result.is_err());
}

#[test]
fn test_sorted_vec_duplicate_destination_error() {
    let result = GenericVocabularyBuilder::<_, SortedVec<usize>>::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 10), (1, 10), (2, 30)].into_iter())
        .build();
    assert!(result.is_err());
}

#[test]
fn test_sorted_vec_no_capacity_build() {
    let sv: SortedVec<usize> = GenericVocabularyBuilder::default()
        .symbols(vec![(0, 10), (1, 20)].into_iter())
        .build()
        .unwrap();
    assert_eq!(sv.len(), 2);
}
