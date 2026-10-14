//! Tests for numeric helper traits.
#![cfg(feature = "std")]

use geometric_traits::traits::{Finite, Number, PositiveInteger, TryFromUsize};

fn assert_positive_integer<T: PositiveInteger>() {}
fn assert_number<T: Number>() {}

#[test]
fn test_try_from_usize_default_method() {
    assert_eq!(<u8 as TryFromUsize>::try_from_usize(42).unwrap(), 42);
    assert!(<u8 as TryFromUsize>::try_from_usize(300).is_err());
}

#[test]
fn test_finite_for_floats() {
    assert!(Finite::is_finite(&1.5f32));
    assert!(!Finite::is_finite(&f32::INFINITY));
    assert!(!Finite::is_finite(&f32::NEG_INFINITY));

    assert!(Finite::is_finite(&2.5f64));
    assert!(!Finite::is_finite(&f64::NAN));
}

#[test]
fn test_finite_for_integers() {
    assert!(Finite::is_finite(&0u8));
    assert!(Finite::is_finite(&123usize));
    assert!(Finite::is_finite(&-7i32));
}

#[test]
fn test_numeric_trait_bounds_compile() {
    assert_positive_integer::<u8>();
    assert_positive_integer::<usize>();
    assert_number::<f64>();
    assert_number::<i32>();
}
