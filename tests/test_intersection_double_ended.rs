//! Tests for Intersection DoubleEndedIterator with crossing front/back
//! candidates, exercising the uncovered paths in the next() and next_back()
//! methods.
#![cfg(feature = "std")]

use geometric_traits::impls::generic_iterators::Intersection;

// ============================================================================
// Intersection: crossing front and back candidates
// ============================================================================

#[test]
fn test_intersection_mixed_crossing() {
    // Exercise the crossing detection in next()
    let iter1 = [1, 2, 3, 4, 5].into_iter();
    let iter2 = [1, 2, 3, 4, 5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from front
    assert_eq!(intersection.next(), Some(1));
    // Consume from back
    assert_eq!(intersection.next_back(), Some(5));
    // Continue from front
    assert_eq!(intersection.next(), Some(2));
    // Continue from back
    assert_eq!(intersection.next_back(), Some(4));
    // Middle element
    assert_eq!(intersection.next(), Some(3));
    // Should be exhausted
    assert_eq!(intersection.next(), None);
    assert_eq!(intersection.next_back(), None);
}

#[test]
fn test_intersection_back_only_partial_overlap() {
    let iter1 = [2, 4, 6, 8].into_iter();
    let iter2 = [1, 2, 7, 8].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next_back(), Some(8));
    assert_eq!(intersection.next(), Some(2));
    // No more common elements between the remaining items
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_crossing_with_unequal_iters() {
    let iter1 = [1, 3, 5, 7, 9].into_iter();
    let iter2 = [3, 5, 7].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next(), Some(3));
    assert_eq!(intersection.next_back(), Some(7));
    assert_eq!(intersection.next(), Some(5));
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_back_exhaustion() {
    let iter1 = [1, 2, 3].into_iter();
    let iter2 = [3].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Back should find 3
    assert_eq!(intersection.next_back(), Some(3));
    // No more matches
    assert_eq!(intersection.next_back(), None);
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_front_then_back_then_front() {
    let iter1 = [10, 20, 30, 40, 50].into_iter();
    let iter2 = [10, 20, 30, 40, 50].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next(), Some(10));
    assert_eq!(intersection.next_back(), Some(50));
    assert_eq!(intersection.next(), Some(20));
    assert_eq!(intersection.next_back(), Some(40));
    assert_eq!(intersection.next(), Some(30));
    assert_eq!(intersection.next_back(), None);
}

// ============================================================================
// Intersection: edge case where back items fall through to front
// ============================================================================

#[test]
fn test_intersection_single_common_element() {
    let iter1 = [5].into_iter();
    let iter2 = [5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next(), Some(5));
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_back_no_common() {
    let iter1 = [1, 3, 5].into_iter();
    let iter2 = [2, 4, 6].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next_back(), None);
}
