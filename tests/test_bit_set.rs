//! Integration tests for [`BitSet`].
#![cfg(feature = "alloc")]

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use geometric_traits::prelude::*;

const WORD: usize = usize::BITS as usize;

fn hash_of(set: &BitSet) -> u64 {
    let mut hasher = DefaultHasher::new();
    set.hash(&mut hasher);
    hasher.finish()
}

fn from_ones(len: usize, ones: &[usize]) -> BitSet {
    let mut set = BitSet::zeros(len);
    for &index in ones {
        set.insert(index);
    }
    set
}

#[test]
fn test_zeros_and_ones_across_a_word_boundary() {
    let len = WORD + 3;
    let zeros = BitSet::zeros(len);
    let ones = BitSet::ones(len);
    assert_eq!(zeros.len(), len);
    assert_eq!(ones.len(), len);
    assert_eq!(zeros.count_ones(), 0);
    assert_eq!(ones.count_ones(), len);
    assert!((0..len).all(|index| !zeros.contains(index)));
    assert!((0..len).all(|index| ones.contains(index)));
    assert_eq!(zeros.as_words().len(), 2);
    assert_eq!(ones.as_words().len(), 2);
}

#[test]
fn test_ones_leaves_no_bits_past_the_length() {
    let len = WORD + 6;
    let all: Vec<usize> = (0..len).collect();
    let inserted = from_ones(len, &all);
    let ones = BitSet::ones(len);
    assert_eq!(ones, inserted);
    assert_eq!(hash_of(&ones), hash_of(&inserted));
    assert_eq!(ones.as_words()[1], (1 << 6) - 1);
    assert_eq!(ones.last_one(), Some(len - 1));
}

#[test]
fn test_ones_of_an_exact_word_multiple_fills_every_word() {
    let ones = BitSet::ones(2 * WORD);
    assert_eq!(ones.as_words(), &[usize::MAX, usize::MAX]);
    assert_eq!(ones.count_ones(), 2 * WORD);
}

#[test]
fn test_empty_set() {
    let empty = BitSet::zeros(0);
    assert!(empty.is_empty());
    assert_eq!(empty, BitSet::ones(0));
    assert_eq!(empty, BitSet::default());
    assert_eq!(empty.as_words(), &[] as &[usize]);
    assert_eq!(empty.iter_ones().count(), 0);
    assert_eq!(empty.last_one(), None);
}

#[test]
fn test_insert_and_remove_report_whether_the_set_changed() {
    let mut set = BitSet::zeros(WORD + 1);
    assert!(set.insert(WORD));
    assert!(!set.insert(WORD));
    assert!(set.contains(WORD));
    assert_eq!(set.count_ones(), 1);
    assert!(set.remove(WORD));
    assert!(!set.remove(WORD));
    assert!(!set.contains(WORD));
    assert_eq!(set, BitSet::zeros(WORD + 1));
}

#[test]
fn test_contains_past_the_length_is_false() {
    let set = BitSet::ones(4);
    assert!(!set.contains(4));
    assert!(!set.contains(WORD - 1));
    assert!(!set.contains(WORD));
    assert!(!set.contains(usize::MAX));
}

#[test]
#[should_panic(expected = "index 4 out of range for a BitSet of length 4")]
fn test_insert_past_the_length_panics() {
    BitSet::zeros(4).insert(4);
}

#[test]
#[should_panic(expected = "index 4 out of range for a BitSet of length 4")]
fn test_remove_past_the_length_panics() {
    BitSet::zeros(4).remove(4);
}

#[test]
fn test_iter_ones_in_both_directions() {
    let ones = [0, 3, WORD - 1, WORD, WORD + 2, 2 * WORD + 1];
    let set = from_ones(2 * WORD + 5, &ones);
    let forward: Vec<usize> = set.iter_ones().collect();
    let backward: Vec<usize> = set.iter_ones().rev().collect();
    assert_eq!(forward, ones);
    assert_eq!(backward, ones.iter().rev().copied().collect::<Vec<_>>());
}

#[test]
fn test_iter_ones_interleaved_never_yields_twice() {
    let ones = [1, WORD - 1, WORD, WORD + 7, 2 * WORD];
    let set = from_ones(2 * WORD + 1, &ones);
    let mut iter = set.iter_ones();
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(2 * WORD));
    assert_eq!(iter.next_back(), Some(WORD + 7));
    assert_eq!(iter.next(), Some(WORD - 1));
    assert_eq!(iter.next(), Some(WORD));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_iter_ones_meeting_inside_one_word() {
    let set = from_ones(WORD, &[2, 5, 9]);
    let mut iter = set.iter_ones();
    assert_eq!(iter.next_back(), Some(9));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next_back(), Some(5));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_iter_ones_len_tracks_both_ends() {
    let ones = [0, WORD - 1, WORD + 1, 2 * WORD, 3 * WORD - 1];
    let set = from_ones(3 * WORD, &ones);
    let mut iter = set.iter_ones();
    assert_eq!(iter.len(), 5);
    iter.next();
    assert_eq!(iter.len(), 4);
    iter.next_back();
    assert_eq!(iter.len(), 3);
    iter.next();
    iter.next();
    assert_eq!(iter.len(), 1);
    assert_eq!(iter.next_back(), Some(2 * WORD));
    assert_eq!(iter.len(), 0);
}

#[test]
fn test_last_one_finds_the_highest_set_bit() {
    assert_eq!(BitSet::zeros(3 * WORD).last_one(), None);
    assert_eq!(from_ones(3 * WORD, &[4, WORD + 1]).last_one(), Some(WORD + 1));
    assert_eq!(from_ones(3 * WORD, &[3 * WORD - 1]).last_one(), Some(3 * WORD - 1));
}

#[test]
fn test_and_count_counts_common_bits() {
    let left = from_ones(WORD + 4, &[0, 2, WORD, WORD + 3]);
    let right = from_ones(WORD + 4, &[2, 3, WORD + 3]);
    assert_eq!(left.and_count(&right), 2);
    assert_eq!(right.and_count(&left), 2);
}

#[test]
fn test_and_count_treats_bits_past_the_shorter_length_as_absent() {
    let long = BitSet::ones(WORD + 4);
    let short = BitSet::ones(3);
    assert_eq!(long.and_count(&short), 3);
    assert_eq!(short.and_count(&long), 3);
}

#[test]
fn test_and_assign_intersects_and_keeps_the_length() {
    let mut left = from_ones(WORD + 4, &[0, 2, WORD, WORD + 3]);
    left &= &from_ones(WORD + 4, &[2, WORD + 3]);
    assert_eq!(left, from_ones(WORD + 4, &[2, WORD + 3]));
}

#[test]
fn test_and_assign_clears_bits_past_the_shorter_length() {
    let mut long = BitSet::ones(WORD + 4);
    long &= &BitSet::ones(3);
    assert_eq!(long.len(), WORD + 4);
    assert_eq!(long, from_ones(WORD + 4, &[0, 1, 2]));
}

#[test]
fn test_and_assign_with_a_longer_mask_keeps_the_tail_clear() {
    let mut short = BitSet::ones(3);
    short &= &BitSet::ones(WORD + 4);
    assert_eq!(short, BitSet::ones(3));
    assert_eq!(short.as_words(), &[0b111]);
}
