//! Tests for Vocabulary/BidirectionalVocabulary blanket impls on Box<V> and
//! Rc<V>, VocabularyRef for &V, BidirectionalVocabularyRef for &V,
//! and implicit numeric vocabularies for u16, u32, u64 (uncovered methods).
#![cfg(feature = "std")]

use std::rc::Rc;

use geometric_traits::traits::{
    BidirectionalVocabulary, BidirectionalVocabularyRef, Vocabulary, VocabularyRef,
};

// ============================================================================
// Box<V> blanket impl for Vocabulary
// ============================================================================

#[test]
fn test_box_vocabulary_convert() {
    let vocab: Box<Vec<&str>> = Box::new(vec!["a", "b", "c"]);
    assert_eq!(Vocabulary::convert(&*vocab, &0), Some("a"));
    assert_eq!(Vocabulary::convert(&*vocab, &1), Some("b"));
    assert_eq!(Vocabulary::convert(&*vocab, &3), None);
}

#[test]
fn test_box_vocabulary_len() {
    let vocab: Box<Vec<&str>> = Box::new(vec!["a", "b"]);
    assert_eq!(Vocabulary::len(&*vocab), 2);
}

#[test]
fn test_box_vocabulary_sources() {
    let vocab: Box<Vec<&str>> = Box::new(vec!["a", "b", "c"]);
    let sources: Vec<usize> = Vocabulary::sources(&*vocab).collect();
    assert_eq!(sources, vec![0, 1, 2]);
}

#[test]
fn test_box_vocabulary_destinations() {
    let vocab: Box<Vec<&str>> = Box::new(vec!["x", "y"]);
    let dests: Vec<&str> = Vocabulary::destinations(&*vocab).collect();
    assert_eq!(dests, vec!["x", "y"]);
}

// ============================================================================
// Rc<V> blanket impl for Vocabulary
// ============================================================================

#[test]
fn test_rc_vocabulary_convert() {
    let vocab: Rc<Vec<&str>> = Rc::new(vec!["a", "b", "c"]);
    assert_eq!(Vocabulary::convert(&*vocab, &0), Some("a"));
    assert_eq!(Vocabulary::convert(&*vocab, &2), Some("c"));
}

#[test]
fn test_rc_vocabulary_len() {
    let vocab: Rc<Vec<&str>> = Rc::new(vec!["a", "b", "c"]);
    assert_eq!(Vocabulary::len(&*vocab), 3);
}

#[test]
fn test_rc_vocabulary_sources() {
    let vocab: Rc<Vec<&str>> = Rc::new(vec!["a", "b"]);
    let sources: Vec<usize> = Vocabulary::sources(&*vocab).collect();
    assert_eq!(sources, vec![0, 1]);
}

#[test]
fn test_rc_vocabulary_destinations() {
    let vocab: Rc<Vec<&str>> = Rc::new(vec!["x", "y", "z"]);
    let dests: Vec<&str> = Vocabulary::destinations(&*vocab).collect();
    assert_eq!(dests, vec!["x", "y", "z"]);
}

// ============================================================================
// Box<V> BidirectionalVocabulary
// ============================================================================

#[test]
fn test_box_bidirectional_vocabulary() {
    let vocab: Box<usize> = Box::new(10);
    assert_eq!(BidirectionalVocabulary::invert(&*vocab, &5), Some(5));
}

// ============================================================================
// Rc<V> BidirectionalVocabulary
// ============================================================================

#[test]
fn test_rc_bidirectional_vocabulary() {
    let vocab: Rc<usize> = Rc::new(10);
    assert_eq!(BidirectionalVocabulary::invert(&*vocab, &5), Some(5));
}

// ============================================================================
// VocabularyRef for &V
// ============================================================================

#[test]
fn test_ref_vocabulary_ref_convert_ref() {
    let vocab: Vec<String> = vec!["apple".to_string(), "banana".to_string()];
    let r = &vocab;
    assert_eq!(VocabularyRef::convert_ref(r, &0), Some(&"apple".to_string()));
    assert_eq!(VocabularyRef::convert_ref(r, &1), Some(&"banana".to_string()));
}

#[test]
fn test_ref_vocabulary_ref_destination_refs() {
    let vocab: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let r = &vocab;
    let refs: Vec<&String> = VocabularyRef::destination_refs(r).collect();
    assert_eq!(refs, vec![&"x".to_string(), &"y".to_string()]);
}

// ============================================================================
// BidirectionalVocabularyRef for &V (HashMap implements this)
// ============================================================================

#[test]
fn test_ref_bidirectional_vocabulary_ref_invert_ref() {
    use std::collections::HashMap;
    let mut vocab: HashMap<usize, String> = HashMap::new();
    vocab.insert(0, "apple".to_string());
    vocab.insert(1, "banana".to_string());
    let r = &vocab;
    assert_eq!(BidirectionalVocabularyRef::invert_ref(r, &"apple".to_string()), Some(&0));
}

#[test]
fn test_ref_bidirectional_vocabulary_ref_source_refs() {
    use std::collections::HashMap;
    let mut vocab: HashMap<usize, String> = HashMap::new();
    vocab.insert(0, "x".to_string());
    vocab.insert(1, "y".to_string());
    let r = &vocab;
    let refs: Vec<&usize> = BidirectionalVocabularyRef::source_refs(r).collect();
    assert_eq!(refs.len(), 2);
}

// ============================================================================
// Implicit numeric vocabularies: uncovered methods for u16, u32, u64
// ============================================================================

#[test]
fn test_u16_vocabulary_convert() {
    let v: u16 = 10;
    assert_eq!(Vocabulary::convert(&v, &3), Some(3));
}

#[test]
fn test_u16_vocabulary_destinations() {
    let v: u16 = 3;
    let dests: Vec<u16> = Vocabulary::destinations(&v).collect();
    assert_eq!(dests, vec![0, 1, 2]);
}

#[test]
fn test_u32_vocabulary_destinations() {
    let v: u32 = 3;
    let dests: Vec<u32> = Vocabulary::destinations(&v).collect();
    assert_eq!(dests, vec![0, 1, 2]);
}

#[test]
fn test_u64_vocabulary_convert() {
    let v: u64 = 10;
    assert_eq!(Vocabulary::convert(&v, &5), Some(5));
}

#[test]
fn test_u64_vocabulary_destinations() {
    let v: u64 = 3;
    let dests: Vec<u64> = Vocabulary::destinations(&v).collect();
    assert_eq!(dests, vec![0, 1, 2]);
}
