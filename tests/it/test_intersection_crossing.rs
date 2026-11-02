//! Tests specifically targeting the crossing detection paths in
//! Intersection::next() and Intersection::next_back() (intersection.rs).
//! These paths check if front/back candidates have crossed each other.
#![cfg(feature = "std")]

use geometric_traits::impls::generic_iterators::Intersection;

// ============================================================================
// Test crossing detection in next() when back candidates exist
// (intersection.rs lines 54-63)
// ============================================================================

#[test]
fn test_intersection_next_detects_crossing_with_back1() {
    // Set up: after consuming from back, the front values exceed back values.
    // This tests the `if val1 > back1` crossing check in next().
    let iter1 = [1, 3, 5, 7].into_iter();
    let iter2 = [1, 3, 5, 7].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from back first
    assert_eq!(intersection.next_back(), Some(7));
    assert_eq!(intersection.next_back(), Some(5));
    // Now consume from front
    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(3));
    // Next front candidate would be > back candidates; crossing detected
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_next_detects_crossing_with_back2() {
    // Different lengths to test the back2 crossing path
    let iter1 = [1, 2, 3, 4, 5, 6, 7, 8].into_iter();
    let iter2 = [2, 4, 6, 8].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from back
    assert_eq!(intersection.next_back(), Some(8));
    assert_eq!(intersection.next_back(), Some(6));
    // Consume from front
    assert_eq!(intersection.next(), Some(2));
    assert_eq!(intersection.next(), Some(4));
    // Crossing: front > back
    assert_eq!(intersection.next(), None);
}

// ============================================================================
// Test crossing detection in next_back() when front candidates exist
// (intersection.rs lines 100-109)
// ============================================================================

#[test]
fn test_intersection_next_back_detects_crossing_with_front1() {
    let iter1 = [1, 3, 5, 7].into_iter();
    let iter2 = [1, 3, 5, 7].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from front first
    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(3));
    assert_eq!(intersection.next(), Some(5));
    // Consume from back
    assert_eq!(intersection.next_back(), Some(7));
    // No more matches between front and back
    assert_eq!(intersection.next_back(), None);
}

#[test]
fn test_intersection_next_back_detects_crossing_with_front2() {
    let iter1 = [10, 20, 30].into_iter();
    let iter2 = [10, 20, 30].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from front
    assert_eq!(intersection.next(), Some(10));
    assert_eq!(intersection.next(), Some(20));
    // Now back
    assert_eq!(intersection.next_back(), Some(30));
    // Both front and back are exhausted
    assert_eq!(intersection.next_back(), None);
    assert_eq!(intersection.next(), None);
}

// ============================================================================
// Test Greater branch in next() (intersection.rs line 68)
// ============================================================================

#[test]
fn test_intersection_next_greater_branch() {
    // iter1 has values that are all greater than iter2's first values
    // This exercises the Greater => discard val2 path
    let iter1 = [5, 10].into_iter();
    let iter2 = [1, 2, 5, 10].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    // iter1 starts at 5, iter2 starts at 1. 5 > 1, so iter2 advances (Greater
    // branch)
    assert_eq!(intersection.next(), Some(5));
    assert_eq!(intersection.next(), Some(10));
    assert_eq!(intersection.next(), None);
}

// ============================================================================
// Test that next() falls through to back items (line 42-48 item refill)
// ============================================================================

#[test]
fn test_intersection_next_uses_back_items() {
    let iter1 = [1, 2, 3].into_iter();
    let iter2 = [1, 2, 3].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume one from back
    assert_eq!(intersection.next_back(), Some(3));
    // Now front iteration should still work for remaining items
    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(2));
    // No more
    assert_eq!(intersection.next(), None);
}

// ============================================================================
// Test next_back Equal branch (line 114)
// ============================================================================

#[test]
fn test_intersection_next_back_equal_match() {
    let iter1 = [10, 20, 30].into_iter();
    let iter2 = [10, 30].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // next_back should find 30 as common element
    assert_eq!(intersection.next_back(), Some(30));
    // Forward should find 10
    assert_eq!(intersection.next(), Some(10));
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_alternating_extensively() {
    let iter1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter();
    let iter2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next_back(), Some(10));
    assert_eq!(intersection.next(), Some(2));
    assert_eq!(intersection.next_back(), Some(9));
    assert_eq!(intersection.next(), Some(3));
    assert_eq!(intersection.next_back(), Some(8));
    assert_eq!(intersection.next(), Some(4));
    assert_eq!(intersection.next_back(), Some(7));
    assert_eq!(intersection.next(), Some(5));
    assert_eq!(intersection.next_back(), Some(6));
    assert_eq!(intersection.next(), None);
    assert_eq!(intersection.next_back(), None);
}
