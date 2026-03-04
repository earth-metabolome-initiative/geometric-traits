//! Tests for small uncovered paths in sorted wrappers and `TotalOrd`.
#![cfg(feature = "std")]

use core::cmp::Ordering;

use geometric_traits::{
    impls::SortedVec,
    traits::{TotalOrd, TransmuteFrom},
};

#[test]
fn test_sorted_vec_transmute_from_sorted_input() {
    let sorted =
        unsafe { <SortedVec<i32> as TransmuteFrom<Vec<i32>>>::transmute_from(vec![1, 2, 3]) };
    assert_eq!(sorted.len(), 3);
}

#[test]
#[should_panic(expected = "source vector is not sorted")]
fn test_sorted_vec_transmute_from_unsorted_panics_in_debug() {
    let _ = unsafe { <SortedVec<i32> as TransmuteFrom<Vec<i32>>>::transmute_from(vec![2, 1, 3]) };
}

#[test]
fn test_total_ord_f32_impl() {
    assert_eq!(TotalOrd::total_cmp(&1.0_f32, &2.0_f32), Ordering::Less);
    assert_eq!(TotalOrd::total_cmp(&2.0_f32, &1.0_f32), Ordering::Greater);
    assert_eq!(TotalOrd::total_cmp(&f32::NAN, &f32::NAN), Ordering::Equal);
}
