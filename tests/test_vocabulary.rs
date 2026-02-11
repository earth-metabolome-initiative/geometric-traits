//! Simple test for vocabulary.

use ::geometric_traits::prelude::*;
use geometric_traits::impls::SortedVec;

#[test]
/// First simple test for vocabulary.
pub fn test_vocabulary() {
    let nodes: Vec<usize> = vec![1, 2, 3, 4, 5];
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(nodes.len())
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    assert_eq!(nodes.len(), 5);
    assert_eq!(nodes.get(0), Some(&1));
    assert_eq!(nodes.get(1), Some(&2));
    assert_eq!(nodes.get(2), Some(&3));
    assert_eq!(nodes.get(3), Some(&4));
    assert_eq!(nodes.get(4), Some(&5));
    assert_eq!(nodes.get(5), None);
    assert_eq!(nodes.get(6), None);
    assert_eq!(nodes.get(7), None);
    assert_eq!(nodes.sources().collect::<Vec<usize>>(), vec![0, 1, 2, 3, 4]);
    assert_eq!(nodes.destinations().collect::<Vec<usize>>(), vec![1, 2, 3, 4, 5]);
}

#[test]
/// Test vocabulary trait methods on Vec.
pub fn test_vec_vocabulary() {
    use geometric_traits::traits::{BidirectionalVocabulary, Vocabulary, VocabularyRef};

    let vocab: Vec<&str> = vec!["apple", "banana", "cherry"];

    // Test Vocabulary methods
    assert_eq!(vocab.len(), 3);
    assert!(!vocab.is_empty());
    assert_eq!(vocab.convert(&0), Some("apple"));
    assert_eq!(vocab.convert(&1), Some("banana"));
    assert_eq!(vocab.convert(&2), Some("cherry"));
    assert_eq!(vocab.convert(&3), None);

    // Test sources iterator
    let sources: Vec<usize> = vocab.sources().collect();
    assert_eq!(sources, vec![0, 1, 2]);

    // Test destinations iterator
    let destinations: Vec<&str> = vocab.destinations().collect();
    assert_eq!(destinations, vec!["apple", "banana", "cherry"]);

    // Test VocabularyRef methods
    assert_eq!(vocab.convert_ref(&0), Some(&"apple"));
    assert_eq!(vocab.convert_ref(&1), Some(&"banana"));
    let refs: Vec<&&str> = vocab.destination_refs().collect();
    assert_eq!(refs.len(), 3);

    // Test BidirectionalVocabulary
    assert_eq!(vocab.invert(&"apple"), Some(0));
    assert_eq!(vocab.invert(&"banana"), Some(1));
    assert_eq!(vocab.invert(&"cherry"), Some(2));
    assert_eq!(vocab.invert(&"date"), None);
}

#[test]
/// Test empty vocabulary.
pub fn test_empty_vocabulary() {
    use geometric_traits::traits::Vocabulary;

    let empty: Vec<i32> = vec![];
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.convert(&0), None);
}

#[test]
/// Test vocabulary through reference.
pub fn test_vocabulary_ref_impl() {
    use geometric_traits::traits::Vocabulary;

    let vocab: Vec<u32> = vec![10, 20, 30];
    let vocab_ref: &Vec<u32> = &vocab;

    assert_eq!(vocab_ref.len(), 3);
    assert_eq!(vocab_ref.convert(&0), Some(10));
    assert_eq!(vocab_ref.convert(&1), Some(20));
    assert_eq!(vocab_ref.convert(&2), Some(30));

    let sources: Vec<usize> = vocab_ref.sources().collect();
    assert_eq!(sources, vec![0, 1, 2]);
}

// ============================================================================
// Box wrapper tests
// ============================================================================

#[test]
fn test_vocabulary_box_wrapper() {
    use std::boxed::Box;

    use geometric_traits::traits::Vocabulary;

    let vocab: Vec<&str> = vec!["one", "two", "three"];
    let boxed: Box<Vec<&str>> = Box::new(vocab);

    assert_eq!(boxed.len(), 3);
    assert!(!boxed.is_empty());
    assert_eq!(boxed.convert(&0), Some("one"));
    assert_eq!(boxed.convert(&1), Some("two"));
    assert_eq!(boxed.convert(&2), Some("three"));
    assert_eq!(boxed.convert(&3), None);

    let sources: Vec<usize> = boxed.sources().collect();
    assert_eq!(sources, vec![0, 1, 2]);

    let destinations: Vec<&str> = boxed.destinations().collect();
    assert_eq!(destinations, vec!["one", "two", "three"]);
}

#[test]
fn test_bidirectional_vocabulary_box_wrapper() {
    use std::boxed::Box;

    use geometric_traits::traits::BidirectionalVocabulary;

    let vocab: Vec<&str> = vec!["alpha", "beta", "gamma"];
    let boxed: Box<Vec<&str>> = Box::new(vocab);

    assert_eq!(boxed.invert(&"alpha"), Some(0));
    assert_eq!(boxed.invert(&"beta"), Some(1));
    assert_eq!(boxed.invert(&"gamma"), Some(2));
    assert_eq!(boxed.invert(&"delta"), None);
}

// ============================================================================
// Rc wrapper tests
// ============================================================================

#[test]
fn test_vocabulary_rc_wrapper() {
    use std::rc::Rc;

    use geometric_traits::traits::Vocabulary;

    let vocab: Vec<i32> = vec![100, 200, 300];
    let rc: Rc<Vec<i32>> = Rc::new(vocab);

    assert_eq!(rc.len(), 3);
    assert!(!rc.is_empty());
    assert_eq!(rc.convert(&0), Some(100));
    assert_eq!(rc.convert(&1), Some(200));
    assert_eq!(rc.convert(&2), Some(300));
    assert_eq!(rc.convert(&3), None);

    let sources: Vec<usize> = rc.sources().collect();
    assert_eq!(sources, vec![0, 1, 2]);

    let destinations: Vec<i32> = rc.destinations().collect();
    assert_eq!(destinations, vec![100, 200, 300]);
}

#[test]
fn test_bidirectional_vocabulary_rc_wrapper() {
    use std::rc::Rc;

    use geometric_traits::traits::BidirectionalVocabulary;

    let vocab: Vec<&str> = vec!["red", "green", "blue"];
    let rc: Rc<Vec<&str>> = Rc::new(vocab);

    assert_eq!(rc.invert(&"red"), Some(0));
    assert_eq!(rc.invert(&"green"), Some(1));
    assert_eq!(rc.invert(&"blue"), Some(2));
    assert_eq!(rc.invert(&"yellow"), None);
}

// ============================================================================
// VocabularyRef wrapper tests
// ============================================================================

#[test]
fn test_vocabulary_ref_convert_ref_via_reference() {
    use geometric_traits::traits::VocabularyRef;

    let vocab: Vec<String> = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let vocab_ref: &Vec<String> = &vocab;

    assert_eq!(vocab_ref.convert_ref(&0), Some(&"x".to_string()));
    assert_eq!(vocab_ref.convert_ref(&1), Some(&"y".to_string()));
    assert_eq!(vocab_ref.convert_ref(&2), Some(&"z".to_string()));
    assert_eq!(vocab_ref.convert_ref(&3), None);
}

#[test]
fn test_vocabulary_ref_destination_refs_via_reference() {
    use geometric_traits::traits::VocabularyRef;

    let vocab: Vec<u64> = vec![1, 2, 3];
    let vocab_ref: &Vec<u64> = &vocab;

    let refs: Vec<&u64> = vocab_ref.destination_refs().collect();
    assert_eq!(refs, vec![&1, &2, &3]);
}

// ============================================================================
// BidirectionalVocabularyRef tests (using HashMap which implements this trait)
// ============================================================================

#[test]
fn test_bidirectional_vocabulary_ref_invert_ref() {
    use std::collections::HashMap;

    use geometric_traits::traits::BidirectionalVocabularyRef;

    let mut vocab: HashMap<usize, &str> = HashMap::new();
    vocab.insert(0, "first");
    vocab.insert(1, "second");
    vocab.insert(2, "third");

    assert_eq!(vocab.invert_ref(&"first"), Some(&0));
    assert_eq!(vocab.invert_ref(&"second"), Some(&1));
    assert_eq!(vocab.invert_ref(&"third"), Some(&2));
    assert_eq!(vocab.invert_ref(&"fourth"), None);
}

#[test]
fn test_bidirectional_vocabulary_ref_source_refs() {
    use std::collections::HashMap;

    use geometric_traits::traits::BidirectionalVocabularyRef;

    let mut vocab: HashMap<usize, i32> = HashMap::new();
    vocab.insert(0, 10);
    vocab.insert(1, 20);
    vocab.insert(2, 30);
    vocab.insert(3, 40);

    let source_refs: Vec<&usize> = vocab.source_refs().collect();
    assert_eq!(source_refs.len(), 4);
    // HashMap doesn't guarantee order, so just check the values are present
    assert!(source_refs.contains(&&0));
    assert!(source_refs.contains(&&1));
    assert!(source_refs.contains(&&2));
    assert!(source_refs.contains(&&3));
}

#[test]
fn test_bidirectional_vocabulary_ref_via_reference() {
    use std::collections::HashMap;

    use geometric_traits::traits::BidirectionalVocabularyRef;

    let mut vocab: HashMap<usize, &str> = HashMap::new();
    vocab.insert(0, "a");
    vocab.insert(1, "b");
    let vocab_ref: &HashMap<usize, &str> = &vocab;

    assert_eq!(vocab_ref.invert_ref(&"a"), Some(&0));
    assert_eq!(vocab_ref.invert_ref(&"b"), Some(&1));

    let source_refs: Vec<&usize> = vocab_ref.source_refs().collect();
    assert_eq!(source_refs.len(), 2);
}
