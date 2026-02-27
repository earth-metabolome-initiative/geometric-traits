//! Tests designed to exercise deeper algorithmic paths in LAPJV and LAPMOD.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{LAPJV, SparseLAPJV, SparseLAPMOD},
};

// ============================================================================
// LAPJV: Dense matrices with conflicts during column reduction
// ============================================================================

#[test]
fn test_lapjv_all_rows_prefer_same_column() {
    // All rows have minimum cost at column 0: creates conflicts
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 50.0, 50.0],
        [1.0, 50.0, 50.0],
        [50.0, 50.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lapjv_all_rows_identical() {
    // All rows identical — maximum conflicts
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 3);
    // Each row must be assigned a unique column
    let mut cols: Vec<u8> = result.iter().map(|&(_, c)| c).collect();
    cols.sort();
    assert_eq!(cols, vec![0, 1, 2]);
}

#[test]
fn test_lapjv_perfect_match_no_augmentation() {
    // Diagonal matrix — column reduction solves everything, no augmentation needed
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 100.0],
        [100.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 2);
    assert!(result.contains(&(0, 0)));
    assert!(result.contains(&(1, 1)));
}

#[test]
fn test_lapjv_5x5_complex_augmentation() {
    // 5x5 matrix designed to exercise find_path and scan
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [10.0, 1.0, 50.0, 50.0, 50.0],
        [50.0, 10.0, 1.0, 50.0, 50.0],
        [50.0, 50.0, 10.0, 1.0, 50.0],
        [50.0, 50.0, 50.0, 10.0, 1.0],
        [1.0, 50.0, 50.0, 50.0, 10.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_lapjv_4x4_ties() {
    // Multiple ties requiring augmenting row reduction
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 1.0, 50.0, 50.0],
        [1.0, 1.0, 50.0, 50.0],
        [50.0, 50.0, 1.0, 1.0],
        [50.0, 50.0, 1.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 4);
}

// ============================================================================
// LAPMOD: Sparse matrices exercising scan_sparse and find_path_sparse
// ============================================================================

#[test]
fn test_lapmod_diamond_pattern() {
    // Diamond: Row 0→{C0,C1}, Row 1→{C0,C2}, Row 2→{C1,C2}
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 5.0, 99.0],
        [2.0, 99.0, 2.0],
        [99.0, 3.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lapmod_chain_augmentation() {
    // Sparse matrix requiring multi-hop augmenting path
    // R0→C0, R1→{C0,C1}, R2→{C1,C2}, R3→{C2,C3}
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0, 99.0],
        [2.0, 1.0, 99.0, 99.0],
        [99.0, 2.0, 1.0, 99.0],
        [99.0, 99.0, 2.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_lapmod_crossing_preferences() {
    // R0 prefers C1, R1 prefers C0 — crossing
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [5.0, 1.0, 99.0],
        [1.0, 5.0, 99.0],
        [99.0, 99.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn test_lapmod_5x5_sparse_augmenting() {
    // 5x5 sparse matrix with overlapping preferences
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 3.0, 99.0, 99.0, 99.0],
        [99.0, 1.0, 3.0, 99.0, 99.0],
        [99.0, 99.0, 1.0, 3.0, 99.0],
        [99.0, 99.0, 99.0, 1.0, 3.0],
        [3.0, 99.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_lapmod_6x6_complex() {
    // 6x6 sparse with non-trivial augmentation paths
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 99.0, 99.0, 99.0, 99.0],
        [2.0, 1.0, 3.0, 99.0, 99.0, 99.0],
        [99.0, 3.0, 1.0, 2.0, 99.0, 99.0],
        [99.0, 99.0, 2.0, 1.0, 3.0, 99.0],
        [99.0, 99.0, 99.0, 3.0, 1.0, 2.0],
        [99.0, 99.0, 99.0, 99.0, 2.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapmod(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 6);
}

// ============================================================================
// SparseLAPJV: Exercises the PaddedMatrix2D wrapper path
// ============================================================================

#[test]
fn test_sparse_lapjv_3x3_with_padding() {
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 5.0, 99.0],
        [99.0, 1.0, 5.0],
        [5.0, 99.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapjv(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn test_sparse_lapjv_rectangular_3x2() {
    // Non-square: 3 rows, 2 columns — padding fills to 3x3
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 5.0],
        [5.0, 1.0],
        [2.0, 3.0],
    ])
    .unwrap();
    let result = m.sparse_lapjv(900.0, 1000.0).unwrap();
    // Only real assignments (not padding)
    assert!(result.len() <= 3);
}

#[test]
fn test_sparse_lapjv_4x4_conflicts() {
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 1.0, 99.0, 99.0],
        [99.0, 1.0, 1.0, 99.0],
        [99.0, 99.0, 1.0, 1.0],
        [1.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let result = m.sparse_lapjv(900.0, 1000.0).unwrap();
    assert_eq!(result.len(), 4);
}
