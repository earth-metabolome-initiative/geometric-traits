//! Additional coverage for intersection iterator edge cases.
#![cfg(feature = "std")]

use geometric_traits::impls::Intersection;

#[test]
fn test_intersection_next_front_crossing_item1_back() {
    let iter1 = [1, 2, 3, 4, 5].into_iter();
    let iter2 = [1, 2, 3, 4, 5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(4));
    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(2));
    assert_eq!(intersection.next(), Some(3));
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_next_back_crossing_front() {
    let iter1 = [1, 2, 3, 4, 5].into_iter();
    let iter2 = [1, 2, 3, 4, 5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(2));
    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(4));
    assert_eq!(intersection.next_back(), Some(3));
    assert_eq!(intersection.next_back(), None);
}

#[test]
fn test_intersection_next_back_no_overlap() {
    let iter1 = [1, 3, 5].into_iter();
    let iter2 = [2, 4, 6].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next_back(), None);
}

#[test]
fn test_intersection_next_back_partial_overlap() {
    let iter1 = [1, 3, 5, 7].into_iter();
    let iter2 = [3, 5, 8].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(3));
    assert_eq!(intersection.next_back(), None);
}
