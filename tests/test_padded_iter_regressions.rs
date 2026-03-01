//! Regression tests for mixed front/back iteration on padded diagonal
//! iterators.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SparseMatrix, SparseMatrix2D, SparseValuedMatrix2D},
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

fn build_padded(
    edges: Vec<(usize, usize, f64)>,
    rows: usize,
    cols: usize,
) -> GenericMatrix2DWithPaddedDiagonal<TestValCSR, impl Fn(usize) -> f64 + Clone> {
    let inner: TestValCSR = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((rows, cols))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericMatrix2DWithPaddedDiagonal::new(inner, |row: usize| {
        f64::from(u32::try_from(row + 1).expect("test row should fit in u32")) * 100.0
    })
    .unwrap()
}

fn sort_f64(values: &mut [f64]) {
    values.sort_by(|left, right| left.partial_cmp(right).expect("test values are finite"));
}

#[test]
fn test_padded_sparse_columns_mixed_iteration_does_not_lose_elements() {
    let padded = build_padded(
        vec![
            (0, 1, 10.0),
            (2, 0, 20.0),
            (2, 1, 21.0),
            (3, 1, 31.0),
            (3, 2, 32.0),
            (4, 1, 41.0),
            (4, 2, 42.0),
        ],
        5,
        3,
    );
    let ops = [false, true, false, true, true, false, true, false, true, true, true, true];

    let expected: Vec<usize> = padded.sparse_columns().collect();
    let mut iter = padded.sparse_columns();
    let mut got = Vec::new();
    for front in ops {
        let value = if front { iter.next() } else { iter.next_back() };
        if let Some(value) = value {
            got.push(value);
        }
    }

    assert_eq!(got.len(), expected.len());
    let mut got_sorted = got.clone();
    got_sorted.sort_unstable();
    let mut expected_sorted = expected.clone();
    expected_sorted.sort_unstable();
    assert_eq!(got_sorted, expected_sorted);
}

#[test]
fn test_padded_sparse_coordinates_mixed_iteration_does_not_lose_elements() {
    let padded = build_padded(
        vec![
            (0, 1, 10.0),
            (2, 0, 20.0),
            (2, 1, 21.0),
            (3, 1, 31.0),
            (3, 2, 32.0),
            (4, 1, 41.0),
            (4, 2, 42.0),
        ],
        5,
        3,
    );
    let ops = [false, true, false, true, true, false, true, false, true, true, true, true];

    let expected: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&padded).collect();
    let mut iter = SparseMatrix::sparse_coordinates(&padded);
    let mut got = Vec::new();
    for front in ops {
        let value = if front { iter.next() } else { iter.next_back() };
        if let Some(value) = value {
            got.push(value);
        }
    }

    assert_eq!(got.len(), expected.len());
    let mut got_sorted = got.clone();
    got_sorted.sort_unstable();
    let mut expected_sorted = expected.clone();
    expected_sorted.sort_unstable();
    assert_eq!(got_sorted, expected_sorted);
}

#[test]
fn test_padded_sparse_row_mixed_iteration_does_not_lose_elements() {
    let padded = build_padded(vec![(3, 1, 31.0), (3, 2, 32.0)], 5, 3);

    let expected: Vec<usize> = padded.sparse_row(3).collect();
    let mut iter = padded.sparse_row(3);
    let got: Vec<usize> =
        [iter.next_back(), iter.next(), iter.next(), iter.next()].into_iter().flatten().collect();

    assert_eq!(got.len(), expected.len());
    let mut got_sorted = got.clone();
    got_sorted.sort_unstable();
    let mut expected_sorted = expected.clone();
    expected_sorted.sort_unstable();
    assert_eq!(got_sorted, expected_sorted);
}

#[test]
fn test_padded_sparse_row_values_mixed_iteration_does_not_lose_elements() {
    let padded = build_padded(vec![(1, 2, 12.0), (1, 3, 13.0)], 4, 4);

    let expected: Vec<f64> = padded.sparse_row_values(1).collect();
    let mut iter = padded.sparse_row_values(1);
    let got: Vec<f64> =
        [iter.next_back(), iter.next(), iter.next(), iter.next()].into_iter().flatten().collect();

    assert_eq!(got.len(), expected.len());
    let mut got_sorted = got.clone();
    sort_f64(&mut got_sorted);
    let mut expected_sorted = expected.clone();
    sort_f64(&mut expected_sorted);
    assert_eq!(got_sorted, expected_sorted);
}

#[test]
fn test_padded_sparse_row_values_reverse_matches_reversed_forward() {
    let padded = build_padded(vec![(3, 1, 31.0), (3, 2, 32.0)], 5, 3);

    let forward: Vec<f64> = padded.sparse_row_values(3).collect();
    let expected_reverse: Vec<f64> = forward.iter().rev().copied().collect();
    let reversed: Vec<f64> = padded.sparse_row_values(3).rev().collect();

    assert_eq!(reversed, expected_reverse);
}

#[test]
fn test_padded_sparse_row_values_front_then_back_keeps_saved_front_value() {
    let padded = build_padded(vec![(0, 1, 11.0), (0, 2, 12.0)], 1, 3);

    let expected: Vec<f64> = padded.sparse_row_values(0).collect();
    let mut iter = padded.sparse_row_values(0);
    let got: Vec<f64> = [iter.next(), iter.next_back(), iter.next_back(), iter.next_back()]
        .into_iter()
        .flatten()
        .collect();

    assert_eq!(got.len(), expected.len());
    let mut got_sorted = got.clone();
    sort_f64(&mut got_sorted);
    let mut expected_sorted = expected.clone();
    sort_f64(&mut expected_sorted);
    assert_eq!(got_sorted, expected_sorted);
}
