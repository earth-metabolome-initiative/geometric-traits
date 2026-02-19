//! Tests for vocabulary trait reference wrapper impls:
//! &V for Vocabulary, VocabularyRef, BidirectionalVocabulary,
//! BidirectionalVocabularyRef.
#![cfg(feature = "std")]

use std::collections::HashMap;

use geometric_traits::{
    impls::SortedVec,
    prelude::*,
    traits::{
        BidirectionalVocabulary, BidirectionalVocabularyRef, GrowableVocabulary, Vocabulary,
        VocabularyBuilder, VocabularyRef,
    },
};

fn build_sorted_vec() -> SortedVec<u32> {
    GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 10), (1, 20), (2, 30)].into_iter())
        .build()
        .unwrap()
}

fn build_hashmap() -> HashMap<i32, i32> {
    let mut map: HashMap<i32, i32> = GrowableVocabulary::new();
    map.add(0, 100).unwrap();
    map.add(1, 200).unwrap();
    map.add(2, 300).unwrap();
    map
}

// ============================================================================
// &V impl for Vocabulary (via SortedVec)
// ============================================================================

#[test]
fn test_vocabulary_convert_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    assert_eq!(sv_ref.convert(&0), Some(10));
    assert_eq!(sv_ref.convert(&1), Some(20));
    assert_eq!(sv_ref.convert(&2), Some(30));
    assert_eq!(sv_ref.convert(&3), None);
}

#[test]
fn test_vocabulary_len_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    assert_eq!(Vocabulary::len(sv_ref), 3);
    assert!(!Vocabulary::is_empty(sv_ref));
}

#[test]
fn test_vocabulary_sources_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    let sources: Vec<usize> = sv_ref.sources().collect();
    assert_eq!(sources, vec![0, 1, 2]);
}

#[test]
fn test_vocabulary_destinations_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    let dests: Vec<u32> = sv_ref.destinations().collect();
    assert_eq!(dests, vec![10, 20, 30]);
}

// ============================================================================
// &V impl for VocabularyRef (via SortedVec)
// ============================================================================

#[test]
fn test_vocabulary_ref_convert_ref_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    assert_eq!(sv_ref.convert_ref(&0), Some(&10));
    assert_eq!(sv_ref.convert_ref(&1), Some(&20));
    assert_eq!(sv_ref.convert_ref(&2), Some(&30));
    assert_eq!(sv_ref.convert_ref(&3), None);
}

#[test]
fn test_vocabulary_ref_destination_refs_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    let refs: Vec<&u32> = sv_ref.destination_refs().collect();
    assert_eq!(refs, vec![&10, &20, &30]);
}

// ============================================================================
// &V impl for BidirectionalVocabulary (via SortedVec)
// ============================================================================

#[test]
fn test_bidirectional_vocabulary_invert_via_ref() {
    let sv = build_sorted_vec();
    let sv_ref: &SortedVec<u32> = &sv;

    assert_eq!(sv_ref.invert(&10), Some(0));
    assert_eq!(sv_ref.invert(&20), Some(1));
    assert_eq!(sv_ref.invert(&30), Some(2));
    assert_eq!(sv_ref.invert(&999), None);
}

// ============================================================================
// &V impl for BidirectionalVocabularyRef (via HashMap)
// ============================================================================

#[test]
fn test_bidirectional_vocabulary_ref_invert_ref_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    assert_eq!(map_ref.invert_ref(&100), Some(&0));
    assert_eq!(map_ref.invert_ref(&200), Some(&1));
    assert_eq!(map_ref.invert_ref(&300), Some(&2));
    assert_eq!(map_ref.invert_ref(&999), None);
}

#[test]
fn test_bidirectional_vocabulary_ref_source_refs_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    let mut refs: Vec<&i32> = map_ref.source_refs().collect();
    refs.sort_unstable();
    assert_eq!(refs, vec![&0, &1, &2]);
}

// ============================================================================
// &V impl for Vocabulary (via HashMap)
// ============================================================================

#[test]
fn test_hashmap_vocabulary_convert_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    assert_eq!(map_ref.convert(&0), Some(100));
    assert_eq!(map_ref.convert(&1), Some(200));
    assert_eq!(map_ref.convert(&99), None);
}

#[test]
fn test_hashmap_vocabulary_len_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    assert_eq!(Vocabulary::len(map_ref), 3);
}

#[test]
fn test_hashmap_bidirectional_invert_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    assert_eq!(BidirectionalVocabulary::invert(map_ref, &100), Some(0));
    assert_eq!(BidirectionalVocabulary::invert(map_ref, &200), Some(1));
}

// ============================================================================
// &V impl for VocabularyRef (via HashMap)
// ============================================================================

#[test]
fn test_hashmap_vocabulary_ref_via_ref() {
    let map = build_hashmap();
    let map_ref: &HashMap<i32, i32> = &map;

    assert_eq!(VocabularyRef::convert_ref(map_ref, &0), Some(&100));
    let refs: Vec<&i32> = map_ref.destination_refs().collect();
    assert_eq!(refs.len(), 3);
}
