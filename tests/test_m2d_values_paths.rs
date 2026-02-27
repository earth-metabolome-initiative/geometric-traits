//! Targeted tests for `M2DValues` forward/backward path coverage.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
};

type TestCsr = ValuedCSR2D<u8, u8, u8, f64>;
type TestPadded = PaddedMatrix2D<TestCsr, fn((u8, u8)) -> f64>;

fn padded_value(_: (u8, u8)) -> f64 {
    999.0
}

fn padded_3x3() -> TestPadded {
    let matrix: TestCsr =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    PaddedMatrix2D::new(matrix, padded_value as fn((u8, u8)) -> f64).unwrap()
}

#[test]
fn test_m2d_values_next_back_crosses_to_previous_back_row() {
    let padded = padded_3x3();
    let mut iter = padded.sparse_values();

    // `next_back` consumes the back row from its front in this iterator.
    assert_eq!(iter.next_back(), Some(7.0));
    assert_eq!(iter.next_back(), Some(8.0));
    assert_eq!(iter.next_back(), Some(9.0));

    // After exhausting row 2, it steps to row 1.
    assert_eq!(iter.next_back(), Some(4.0));
}

#[test]
fn test_m2d_values_next_back_falls_back_to_front_iterator_when_rows_meet() {
    let padded = padded_3x3();
    let mut iter = padded.sparse_values();

    // Move the front iterator into row 1.
    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next(), Some(3.0));
    assert_eq!(iter.next(), Some(4.0));

    // Exhaust back row 2, then `next_back` falls back to `next.next()`.
    assert_eq!(iter.next_back(), Some(7.0));
    assert_eq!(iter.next_back(), Some(8.0));
    assert_eq!(iter.next_back(), Some(9.0));
    assert_eq!(iter.next_back(), Some(5.0));
}

#[test]
fn test_m2d_values_len_updates_with_front_and_back_consumption() {
    let matrix: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    let mut iter = matrix.sparse_values();

    assert_eq!(iter.len(), 9);

    assert_eq!(iter.next(), Some(1.0));
    assert_eq!(iter.len(), 8);

    assert_eq!(iter.next_back(), Some(9.0));
    assert_eq!(iter.len(), 7);

    assert_eq!(iter.next(), Some(2.0));
    assert_eq!(iter.next_back(), Some(8.0));
    assert_eq!(iter.len(), 5);
}
