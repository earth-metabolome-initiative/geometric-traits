//! Tests for basic traits: AsPrimitive, TotalOrd, Coordinates, Vector, Symbol.
#![cfg(feature = "std")]

use core::cmp::Ordering;

use geometric_traits::traits::{Coordinates, Symbol, TotalOrd, Vector};
use num_traits::AsPrimitive;

// ============================================================================
// AsPrimitive<usize> tests
// ============================================================================

#[test]
fn test_as_usize_u8() {
    assert_eq!(AsPrimitive::<usize>::as_(0_u8), 0_usize);
    assert_eq!(AsPrimitive::<usize>::as_(42_u8), 42_usize);
    assert_eq!(AsPrimitive::<usize>::as_(255_u8), 255_usize);
}

#[test]
fn test_as_usize_u16() {
    assert_eq!(AsPrimitive::<usize>::as_(0_u16), 0_usize);
    assert_eq!(AsPrimitive::<usize>::as_(1000_u16), 1000_usize);
    assert_eq!(AsPrimitive::<usize>::as_(65535_u16), 65535_usize);
}

#[test]
fn test_as_usize_u32() {
    assert_eq!(AsPrimitive::<usize>::as_(0_u32), 0_usize);
    assert_eq!(AsPrimitive::<usize>::as_(100_000_u32), 100_000_usize);
    assert_eq!(AsPrimitive::<usize>::as_(u32::MAX), u32::MAX as usize);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_as_usize_u64() {
    assert_eq!(AsPrimitive::<usize>::as_(0_u64), 0_usize);
    assert_eq!(AsPrimitive::<usize>::as_(1_000_000_000_u64), 1_000_000_000_usize);
}

#[test]
fn test_as_usize_usize() {
    assert_eq!(AsPrimitive::<usize>::as_(0_usize), 0_usize);
    assert_eq!(AsPrimitive::<usize>::as_(42_usize), 42_usize);
    assert_eq!(AsPrimitive::<usize>::as_(usize::MAX), usize::MAX);
}

// ============================================================================
// TotalOrd tests
// ============================================================================

#[test]
fn test_total_ord_f32() {
    assert_eq!(1.0_f32.total_cmp(&2.0), Ordering::Less);
    assert_eq!(2.0_f32.total_cmp(&1.0), Ordering::Greater);
    assert_eq!(1.0_f32.total_cmp(&1.0), Ordering::Equal);
    assert_eq!(0.0_f32.total_cmp(&-0.0), Ordering::Greater);
    // NaN handling
    assert_eq!(f32::NAN.total_cmp(&f32::NAN), Ordering::Equal);
}

#[test]
fn test_total_ord_f64() {
    assert_eq!(1.0_f64.total_cmp(&2.0), Ordering::Less);
    assert_eq!(2.0_f64.total_cmp(&1.0), Ordering::Greater);
    assert_eq!(1.0_f64.total_cmp(&1.0), Ordering::Equal);
    assert_eq!(f64::INFINITY.total_cmp(&f64::MAX), Ordering::Greater);
    assert_eq!(f64::NEG_INFINITY.total_cmp(&f64::MIN), Ordering::Less);
}

#[test]
fn test_total_ord_integers() {
    // Unsigned
    assert_eq!(1_u8.total_cmp(&2), Ordering::Less);
    assert_eq!(100_u16.total_cmp(&50), Ordering::Greater);
    assert_eq!(42_u32.total_cmp(&42), Ordering::Equal);
    assert_eq!(0_u64.total_cmp(&1), Ordering::Less);
    assert_eq!(5_usize.total_cmp(&5), Ordering::Equal);

    // Signed
    assert_eq!((-1_i8).total_cmp(&1), Ordering::Less);
    assert_eq!(0_i16.total_cmp(&-100), Ordering::Greater);
    assert_eq!((-42_i32).total_cmp(&-42), Ordering::Equal);
    assert_eq!(i64::MIN.total_cmp(&i64::MAX), Ordering::Less);
    assert_eq!(0_isize.total_cmp(&0), Ordering::Equal);
}

// ============================================================================
// Coordinates tests
// ============================================================================

#[test]
fn test_coordinates_unit() {
    assert_eq!(<()>::dimensions(), 0);
}

#[test]
fn test_coordinates_tuples() {
    assert_eq!(<(i32,)>::dimensions(), 1);
    assert_eq!(<(i32, i32)>::dimensions(), 2);
    assert_eq!(<(i32, i32, i32)>::dimensions(), 3);
    assert_eq!(<(f64, f64, f64, f64)>::dimensions(), 4);
}

#[test]
fn test_coordinates_arrays() {
    assert_eq!(<[f64; 0]>::dimensions(), 0);
    assert_eq!(<[f64; 1]>::dimensions(), 1);
    assert_eq!(<[f64; 2]>::dimensions(), 2);
    assert_eq!(<[f64; 3]>::dimensions(), 3);
    assert_eq!(<[i32; 4]>::dimensions(), 4);
}

// ============================================================================
// Vector tests
// ============================================================================

#[test]
fn test_vector_vec() {
    let v: Vec<i32> = vec![10, 20, 30, 40];

    assert_eq!(v.len(), 4);
    assert!(!v.is_empty());

    let items: Vec<&i32> = v.iter().collect();
    assert_eq!(items, vec![&10, &20, &30, &40]);

    let indices: Vec<usize> = v.indices().collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);

    // Test indexing
    assert_eq!(v[0], 10);
    assert_eq!(v[3], 40);
}

#[test]
fn test_vector_empty_vec() {
    let v: Vec<i32> = vec![];

    assert_eq!(v.len(), 0);
    assert!(v.is_empty());

    let items: Vec<&i32> = v.iter().collect();
    assert!(items.is_empty());

    let indices: Vec<usize> = v.indices().collect();
    assert!(indices.is_empty());
}

#[test]
fn test_vector_array() {
    let arr: [i32; 3] = [100, 200, 300];

    assert_eq!(arr.len(), 3);
    // Note: is_empty on fixed-size array is known at compile time

    let items: Vec<&i32> = arr.iter().collect();
    assert_eq!(items, vec![&100, &200, &300]);

    let indices: Vec<usize> = arr.indices().collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn test_vector_slice() {
    let data = [1, 2, 3, 4, 5];
    let slice: &[i32] = &data[1..4];

    assert_eq!(slice.len(), 3);
    assert!(!slice.is_empty());

    let items: Vec<&i32> = slice.iter().collect();
    assert_eq!(items, vec![&2, &3, &4]);
}

// ============================================================================
// Symbol tests
// ============================================================================

fn accepts_symbol<S: Symbol>(s: &S) -> S {
    s.clone()
}

#[test]
fn test_symbol_primitives() {
    assert_eq!(accepts_symbol(&42_u32), 42);
    assert_eq!(accepts_symbol(&42_i64), 42);
    assert_eq!(accepts_symbol(&"hello"), "hello");
}

#[test]
fn test_symbol_string() {
    let s = String::from("world");
    assert_eq!(accepts_symbol(&s), "world");
}

#[test]
fn test_symbol_tuple() {
    let t = (1, 2);
    assert_eq!(accepts_symbol(&t), (1, 2));
}
