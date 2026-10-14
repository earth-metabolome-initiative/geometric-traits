//! Extended tests for Vocabulary traits including GrowableVocabulary,
//! BidirectionalVocabulary on SortedVec, and wrapper types.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::SortedVec,
    prelude::*,
    traits::{BidirectionalVocabulary, GrowableVocabulary, Vocabulary, VocabularyBuilder},
};

// ============================================================================
// GrowableVocabulary on SortedVec
// ============================================================================

#[test]
fn test_growable_vocabulary_new() {
    let sv: SortedVec<usize> = GrowableVocabulary::new();
    assert!(sv.is_empty());
    assert_eq!(sv.len(), 0);
}

#[test]
fn test_growable_vocabulary_with_capacity() {
    let sv: SortedVec<usize> = GrowableVocabulary::with_capacity(10);
    assert!(sv.is_empty());
    assert_eq!(sv.len(), 0);
}

#[test]
fn test_growable_vocabulary_add() {
    let mut sv: SortedVec<usize> = GrowableVocabulary::new();
    sv.add(0, 10).unwrap();
    sv.add(1, 20).unwrap();
    sv.add(2, 30).unwrap();

    assert_eq!(sv.len(), 3);
    assert_eq!(sv.convert(&0), Some(10));
    assert_eq!(sv.convert(&1), Some(20));
    assert_eq!(sv.convert(&2), Some(30));
}

#[test]
fn test_growable_vocabulary_add_error_duplicate_source() {
    let mut sv: SortedVec<usize> = GrowableVocabulary::new();
    sv.add(0, 10).unwrap();
    // Adding the same source should fail
    assert!(sv.add(0, 20).is_err());
}

// ============================================================================
// GrowableVocabulary on Vec
// ============================================================================

#[test]
fn test_growable_vocabulary_vec_new() {
    let v: Vec<usize> = GrowableVocabulary::new();
    assert!(Vocabulary::is_empty(&v));
}

#[test]
fn test_growable_vocabulary_vec_add() {
    let mut v: Vec<usize> = GrowableVocabulary::new();
    GrowableVocabulary::add(&mut v, 0, 10).unwrap();
    GrowableVocabulary::add(&mut v, 1, 20).unwrap();

    assert_eq!(Vocabulary::len(&v), 2);
    assert_eq!(v.convert(&0), Some(10));
    assert_eq!(v.convert(&1), Some(20));
}

// ============================================================================
// BidirectionalVocabulary on SortedVec
// ============================================================================

#[test]
fn test_sorted_vec_invert() {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 100), (1, 200), (2, 300)].into_iter())
        .build()
        .unwrap();

    assert_eq!(nodes.invert(&100), Some(0));
    assert_eq!(nodes.invert(&200), Some(1));
    assert_eq!(nodes.invert(&300), Some(2));
    assert_eq!(nodes.invert(&999), None);
}

#[test]
fn test_sorted_vec_sources_destinations() {
    let nodes: SortedVec<u32> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(4)
        .symbols(vec![(0, 10), (1, 20), (2, 30), (3, 40)].into_iter())
        .build()
        .unwrap();

    let sources: Vec<usize> = nodes.sources().collect();
    assert_eq!(sources, vec![0, 1, 2, 3]);

    let dests: Vec<u32> = nodes.destinations().collect();
    assert_eq!(dests, vec![10, 20, 30, 40]);
}

#[test]
fn test_sorted_vec_convert() {
    let nodes: SortedVec<u32> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(2)
        .symbols(vec![(0, 42), (1, 99)].into_iter())
        .build()
        .unwrap();

    assert_eq!(nodes.convert(&0), Some(42));
    assert_eq!(nodes.convert(&1), Some(99));
    assert_eq!(nodes.convert(&2), None);
}

#[test]
fn test_sorted_vec_is_empty() {
    let empty: SortedVec<usize> = SortedVec::new();
    assert!(Vocabulary::is_empty(&empty));

    let non_empty: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(1)
        .symbols(vec![(0, 1)].into_iter())
        .build()
        .unwrap();
    assert!(!Vocabulary::is_empty(&non_empty));
}

// ============================================================================
// SortedVec::try_from tests
// ============================================================================

#[test]
fn test_sorted_vec_try_from_sorted() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![1, 2, 3, 4, 5]).unwrap();
    assert_eq!(sv.len(), 5);
    assert_eq!(sv.get(0), Some(&1));
    assert_eq!(sv.get(4), Some(&5));
}

#[test]
fn test_sorted_vec_try_from_unsorted() {
    let result = SortedVec::try_from(vec![3, 1, 2]);
    assert!(result.is_err());
}

#[test]
fn test_sorted_vec_try_from_empty() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![]).unwrap();
    assert!(sv.is_empty());
}

// ============================================================================
// SortedVec Index, iter, clone, debug
// ============================================================================

#[test]
fn test_sorted_vec_index() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![10, 20, 30]).unwrap();
    assert_eq!(sv[0], 10);
    assert_eq!(sv[1], 20);
    assert_eq!(sv[2], 30);
}

#[test]
fn test_sorted_vec_iter() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![5, 10, 15]).unwrap();
    let items: Vec<&u32> = sv.iter().collect();
    assert_eq!(items, vec![&5, &10, &15]);
}

#[test]
fn test_sorted_vec_debug() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![1, 2]).unwrap();
    let debug = format!("{sv:?}");
    assert!(debug.contains("SortedVec"));
}

#[test]
fn test_sorted_vec_clone() {
    let sv: SortedVec<u32> = SortedVec::try_from(vec![1, 2, 3]).unwrap();
    let cloned = sv.clone();
    assert_eq!(sv, cloned);
}

#[test]
fn test_sorted_vec_default() {
    let sv: SortedVec<u32> = SortedVec::default();
    assert!(sv.is_empty());
}
