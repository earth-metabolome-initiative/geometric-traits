//! Tests for Coordinates trait implementations.
#![cfg(feature = "std")]

use geometric_traits::traits::Coordinates;

// ============================================================================
// Unit type tests
// ============================================================================

#[test]
fn test_unit_dimensions() {
    assert_eq!(<()>::dimensions(), 0);
}

// ============================================================================
// Tuple tests
// ============================================================================

#[test]
fn test_tuple_1_dimensions() {
    assert_eq!(<(i32,)>::dimensions(), 1);
    assert_eq!(<(f64,)>::dimensions(), 1);
    assert_eq!(<(String,)>::dimensions(), 1);
}

#[test]
fn test_tuple_2_dimensions() {
    assert_eq!(<(i32, i32)>::dimensions(), 2);
    assert_eq!(<(f64, f64)>::dimensions(), 2);
    assert_eq!(<(i32, f64)>::dimensions(), 2);
}

#[test]
fn test_tuple_3_dimensions() {
    assert_eq!(<(i32, i32, i32)>::dimensions(), 3);
    assert_eq!(<(f64, f64, f64)>::dimensions(), 3);
    assert_eq!(<(i32, f32, f64)>::dimensions(), 3);
}

#[test]
fn test_tuple_4_dimensions() {
    assert_eq!(<(i32, i32, i32, i32)>::dimensions(), 4);
    assert_eq!(<(f64, f64, f64, f64)>::dimensions(), 4);
}

// ============================================================================
// Array tests
// ============================================================================

#[test]
fn test_array_0_dimensions() {
    assert_eq!(<[i32; 0]>::dimensions(), 0);
    assert_eq!(<[f64; 0]>::dimensions(), 0);
}

#[test]
fn test_array_1_dimensions() {
    assert_eq!(<[i32; 1]>::dimensions(), 1);
    assert_eq!(<[f64; 1]>::dimensions(), 1);
}

#[test]
fn test_array_2_dimensions() {
    assert_eq!(<[i32; 2]>::dimensions(), 2);
    assert_eq!(<[f64; 2]>::dimensions(), 2);
}

#[test]
fn test_array_3_dimensions() {
    assert_eq!(<[i32; 3]>::dimensions(), 3);
    assert_eq!(<[f64; 3]>::dimensions(), 3);
}

#[test]
fn test_array_4_dimensions() {
    assert_eq!(<[i32; 4]>::dimensions(), 4);
    assert_eq!(<[f64; 4]>::dimensions(), 4);
}

// ============================================================================
// Consistency checks
// ============================================================================

#[test]
fn test_tuple_array_consistency() {
    assert_eq!(<()>::dimensions(), <[i32; 0]>::dimensions());
    assert_eq!(<(i32,)>::dimensions(), <[i32; 1]>::dimensions());
    assert_eq!(<(i32, i32)>::dimensions(), <[i32; 2]>::dimensions());
    assert_eq!(<(i32, i32, i32)>::dimensions(), <[i32; 3]>::dimensions());
    assert_eq!(<(i32, i32, i32, i32)>::dimensions(), <[i32; 4]>::dimensions());
}
