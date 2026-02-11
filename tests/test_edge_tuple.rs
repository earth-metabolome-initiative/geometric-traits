//! Tests for Edge and AttributedEdge implementations on tuples.

use geometric_traits::traits::{AttributedEdge, Edge};

// ============================================================================
// Edge tests for 2-tuple
// ============================================================================

#[test]
fn test_edge_tuple_2_usize() {
    let edge: (usize, usize) = (5, 10);
    assert_eq!(edge.source(), 5);
    assert_eq!(edge.destination(), 10);
}

#[test]
fn test_edge_tuple_2_u8() {
    let edge: (u8, u8) = (100, 200);
    assert_eq!(edge.source(), 100);
    assert_eq!(edge.destination(), 200);
}

#[test]
fn test_edge_tuple_2_u16() {
    let edge: (u16, u16) = (1000, 2000);
    assert_eq!(edge.source(), 1000);
    assert_eq!(edge.destination(), 2000);
}

#[test]
fn test_edge_tuple_2_mixed_u8_u16() {
    let edge: (u8, u16) = (10, 20);
    assert_eq!(edge.source(), 10_u8);
    assert_eq!(edge.destination(), 20_u16);
}

#[test]
fn test_edge_tuple_2_mixed_u16_usize() {
    let edge: (u16, usize) = (100, 200);
    assert_eq!(edge.source(), 100_u16);
    assert_eq!(edge.destination(), 200_usize);
}

// ============================================================================
// Edge tests for 3-tuple (with weight)
// ============================================================================

#[test]
fn test_edge_tuple_3_usize_f64() {
    let edge: (usize, usize, f64) = (1, 2, 3.5);
    assert_eq!(edge.source(), 1);
    assert_eq!(edge.destination(), 2);
}

#[test]
fn test_edge_tuple_3_u8_f32() {
    let edge: (u8, u8, f32) = (10, 20, 1.5);
    assert_eq!(edge.source(), 10);
    assert_eq!(edge.destination(), 20);
}

#[test]
fn test_edge_tuple_3_u16_i32() {
    let edge: (u16, u16, i32) = (5, 10, -100);
    assert_eq!(edge.source(), 5);
    assert_eq!(edge.destination(), 10);
}

// ============================================================================
// AttributedEdge tests for 3-tuple
// ============================================================================

#[test]
fn test_attributed_edge_tuple_3_f64() {
    let edge: (usize, usize, f64) = (1, 2, 3.5);
    assert!((edge.attribute() - 3.5).abs() < f64::EPSILON);
}

#[test]
fn test_attributed_edge_tuple_3_f32() {
    let edge: (u8, u8, f32) = (10, 20, 1.5);
    assert!((edge.attribute() - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_attributed_edge_tuple_3_i32() {
    let edge: (u16, u16, i32) = (5, 10, -100);
    assert_eq!(edge.attribute(), -100);
}

#[test]
fn test_attributed_edge_tuple_3_u64_weight() {
    let edge: (usize, usize, u64) = (0, 1, 12345);
    assert_eq!(edge.attribute(), 12345);
}

#[test]
fn test_attributed_edge_tuple_3_zero_weight() {
    let edge: (usize, usize, f64) = (0, 0, 0.0);
    assert!((edge.attribute()).abs() < f64::EPSILON);
}

#[test]
fn test_attributed_edge_tuple_3_negative_weight() {
    let edge: (usize, usize, i64) = (1, 2, -999);
    assert_eq!(edge.attribute(), -999);
}

// ============================================================================
// is_self_loop tests
// ============================================================================

#[test]
fn test_is_self_loop_true() {
    let edge: (usize, usize) = (5, 5);
    assert!(edge.is_self_loop());
}

#[test]
fn test_is_self_loop_false() {
    let edge: (usize, usize) = (5, 10);
    assert!(!edge.is_self_loop());
}

#[test]
fn test_is_self_loop_true_u8() {
    let edge: (u8, u8) = (0, 0);
    assert!(edge.is_self_loop());
}

#[test]
fn test_is_self_loop_weighted() {
    let edge: (usize, usize, f64) = (3, 3, 1.5);
    assert!(edge.is_self_loop());

    let edge2: (usize, usize, f64) = (3, 4, 1.5);
    assert!(!edge2.is_self_loop());
}
