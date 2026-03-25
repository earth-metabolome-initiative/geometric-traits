//! Tests for valued trait delegation through SquareCSR2D and SymmetricCSR2D.
#![cfg(feature = "alloc")]

use geometric_traits::{
    impls::{SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::MatrixMut,
};

/// Build a SquareCSR2D wrapping a ValuedCSR2D from sorted symmetric entries.
fn build_valued_square(
    n: usize,
    edges: &[(usize, usize, i32)],
) -> SquareCSR2D<ValuedCSR2D<usize, usize, usize, i32>> {
    let mut valued: ValuedCSR2D<usize, usize, usize, i32> =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), edges.len());
    // Add all entries (must be in row-major sorted order).
    let mut sorted: Vec<(usize, usize, i32)> = edges.to_vec();
    sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut diag = 0usize;
    for (r, c, v) in sorted {
        MatrixMut::add(&mut valued, (r, c, v)).unwrap();
        if r == c {
            diag += 1;
        }
    }
    SquareCSR2D::from_parts(valued, diag)
}

/// Build a SymmetricCSR2D wrapping a ValuedCSR2D from upper-triangular edges.
fn build_valued_symmetric(
    n: usize,
    upper_edges: &[(usize, usize, i32)],
) -> SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, i32>> {
    let mut all: Vec<(usize, usize, i32)> = Vec::new();
    for &(r, c, v) in upper_edges {
        all.push((r, c, v));
        if r != c {
            all.push((c, r, v));
        }
    }
    all.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut valued: ValuedCSR2D<usize, usize, usize, i32> =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), all.len());
    for (r, c, v) in all {
        MatrixMut::add(&mut valued, (r, c, v)).unwrap();
    }
    SymmetricCSR2D::from_parts(SquareCSR2D::from_parts(valued, 0))
}

fn to_i32(index: usize) -> i32 {
    i32::try_from(index).expect("test indices fit in i32")
}

fn coord_sum_to_i32(row: usize, column: usize) -> i32 {
    i32::try_from(row + column).expect("test indices fit in i32")
}

// ============================================================================
// SquareCSR2D<ValuedCSR2D> tests
// ============================================================================

#[test]
fn test_square_valued_satisfies_traits() {
    let sq = build_valued_square(3, &[(0, 1, 10), (1, 0, 20), (1, 2, 30)]);

    // SquareMatrix
    assert_eq!(sq.order(), 3);

    // SparseSquareMatrix
    assert_eq!(sq.number_of_defined_diagonal_values(), 0);

    // SparseValuedMatrix2D
    let row0_vals: Vec<i32> = sq.sparse_row_values(0).collect();
    assert_eq!(row0_vals, vec![10]);

    let row1_vals: Vec<i32> = sq.sparse_row_values(1).collect();
    assert_eq!(row1_vals, vec![20, 30]);

    // sparse_value_at
    assert_eq!(sq.sparse_value_at(0, 1), Some(10));
    assert_eq!(sq.sparse_value_at(1, 0), Some(20));
    assert_eq!(sq.sparse_value_at(0, 0), None);
}

#[test]
fn test_square_valued_sparse_values() {
    let sq = build_valued_square(2, &[(0, 0, 1), (0, 1, 2), (1, 1, 3)]);
    let vals: Vec<i32> = sq.sparse_values().collect();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_square_valued_select_value() {
    let sq = build_valued_square(2, &[(0, 0, 100), (0, 1, 200), (1, 1, 300)]);
    assert_eq!(sq.select_value(0), 100);
    assert_eq!(sq.select_value(1), 200);
    assert_eq!(sq.select_value(2), 300);
}

#[test]
fn test_square_valued_diagonal_count() {
    let sq = build_valued_square(3, &[(0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 2, 4)]);
    assert_eq!(sq.number_of_defined_diagonal_values(), 3);
}

#[test]
fn test_square_valued_try_rank() {
    let sq = build_valued_square(3, &[(0, 0, 10), (0, 2, 20), (1, 1, 30)]);
    assert_eq!(sq.try_rank(0, 0), Some(0));
    assert_eq!(sq.try_rank(0, 2), Some(1));
    assert_eq!(sq.try_rank(1, 1), Some(2));
    assert_eq!(sq.try_rank(2, 2), None);
}

#[test]
fn test_square_valued_reference_accessors() {
    let sq = build_valued_square(3, &[(0, 0, 10), (0, 2, 20), (1, 1, 30)]);

    let values: Vec<&i32> = sq.sparse_values_ref().collect();
    assert_eq!(values, vec![&10, &20, &30]);

    assert_eq!(sq.select_value_ref(1), &20);

    let row0: Vec<&i32> = sq.sparse_row_values_ref(0).collect();
    assert_eq!(row0, vec![&10, &20]);

    assert_eq!(sq.sparse_value_at_ref(0, 2), Some(&20));
    assert_eq!(sq.sparse_value_at_ref(2, 2), None);
}

#[test]
fn test_square_valued_mutable_accessors() {
    let mut sq = build_valued_square(3, &[(0, 0, 10), (0, 2, 20), (1, 1, 30)]);

    for value in sq.sparse_values_mut() {
        *value += 1;
    }

    for ((row, column), value) in sq.sparse_entries_mut() {
        *value += coord_sum_to_i32(row, column);
    }

    *sq.select_value_mut(0) = 100;

    for value in sq.sparse_row_values_mut(0) {
        *value += 5;
    }

    if let Some(value) = sq.sparse_value_at_mut(1, 1) {
        *value *= 2;
    }

    for (column, value) in sq.sparse_row_entries_mut(0) {
        *value += to_i32(column);
    }

    assert_eq!(sq.sparse_values().collect::<Vec<i32>>(), vec![105, 30, 66]);
    assert_eq!(sq.sparse_value_at(0, 0), Some(105));
    assert_eq!(sq.sparse_value_at(0, 2), Some(30));
    assert_eq!(sq.sparse_value_at(1, 1), Some(66));
}

// ============================================================================
// SymmetricCSR2D<ValuedCSR2D> tests
// ============================================================================

#[test]
fn test_symmetric_valued_satisfies_traits() {
    let sym = build_valued_symmetric(3, &[(0, 1, 10), (0, 2, 20), (1, 2, 30)]);

    // SquareMatrix
    assert_eq!(sym.order(), 3);

    // SparseValuedMatrix2D
    let row0_vals: Vec<i32> = sym.sparse_row_values(0).collect();
    assert_eq!(row0_vals, vec![10, 20]);

    // Symmetric access
    assert_eq!(sym.sparse_value_at(0, 1), Some(10));
    assert_eq!(sym.sparse_value_at(1, 0), Some(10));
    assert_eq!(sym.sparse_value_at(0, 2), Some(20));
    assert_eq!(sym.sparse_value_at(2, 0), Some(20));
    assert_eq!(sym.sparse_value_at(1, 2), Some(30));
    assert_eq!(sym.sparse_value_at(2, 1), Some(30));
}

#[test]
fn test_symmetric_valued_sparse_values() {
    let sym = build_valued_symmetric(3, &[(0, 1, 5), (1, 2, 7)]);
    let vals: Vec<i32> = sym.sparse_values().collect();
    // Both directions stored: (0,1)=5, (1,0)=5, (1,2)=7, (2,1)=7
    assert_eq!(vals, vec![5, 5, 7, 7]);
}

#[test]
fn test_symmetric_valued_select_value() {
    let sym = build_valued_symmetric(2, &[(0, 1, 42)]);
    // Two entries: (0,1)=42, (1,0)=42
    assert_eq!(sym.select_value(0), 42);
    assert_eq!(sym.select_value(1), 42);
}

#[test]
fn test_symmetric_valued_empty() {
    let sym = build_valued_symmetric(3, &[]);
    assert_eq!(sym.order(), 3);
    assert!(sym.is_empty());
    assert_eq!(sym.sparse_value_at(0, 1), None);
}

#[test]
fn test_symmetric_valued_try_rank() {
    let sym = build_valued_symmetric(3, &[(0, 1, 10), (1, 2, 20)]);
    assert_eq!(sym.try_rank(0, 1), Some(0));
    assert_eq!(sym.try_rank(1, 0), Some(1));
    assert_eq!(sym.try_rank(1, 2), Some(2));
    assert_eq!(sym.try_rank(2, 0), None);
}

#[test]
fn test_symmetric_valued_reference_accessors() {
    let sym = build_valued_symmetric(3, &[(0, 1, 10), (1, 2, 20)]);

    let values: Vec<&i32> = sym.sparse_values_ref().collect();
    assert_eq!(values, vec![&10, &10, &20, &20]);

    assert_eq!(sym.select_value_ref(2), &20);

    let row1: Vec<&i32> = sym.sparse_row_values_ref(1).collect();
    assert_eq!(row1, vec![&10, &20]);

    assert_eq!(sym.sparse_value_at_ref(2, 1), Some(&20));
    assert_eq!(sym.sparse_value_at_ref(2, 2), None);
}

#[test]
fn test_symmetric_valued_mutable_accessors() {
    let mut sym = build_valued_symmetric(3, &[(0, 1, 10), (1, 2, 20)]);

    for value in sym.sparse_values_mut() {
        *value += 1;
    }

    for ((row, column), value) in sym.sparse_entries_mut() {
        *value += coord_sum_to_i32(row, column);
    }

    *sym.select_value_mut(0) = 100;

    for value in sym.sparse_row_values_mut(1) {
        *value += 5;
    }

    if let Some(value) = sym.sparse_value_at_mut(2, 1) {
        *value *= 2;
    }

    for (column, value) in sym.sparse_row_entries_mut(1) {
        *value += to_i32(column);
    }

    assert_eq!(sym.sparse_values().collect::<Vec<i32>>(), vec![100, 17, 31, 48]);
    assert_eq!(sym.sparse_value_at(0, 1), Some(100));
    assert_eq!(sym.sparse_value_at(1, 0), Some(17));
    assert_eq!(sym.sparse_value_at(1, 2), Some(31));
    assert_eq!(sym.sparse_value_at(2, 1), Some(48));
}
