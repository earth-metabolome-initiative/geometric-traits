//! Additional branch-focused tests for `test_utils` LAP/Louvain helpers.
#![cfg(all(feature = "std", feature = "arbitrary"))]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    test_utils::{
        check_lap_sparse_wrapper_invariants, check_lap_square_invariants, check_louvain_invariants,
    },
};

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn build_csr(shape: (u8, u8), entries: &[(u8, u8, f64)]) -> Csr {
    let mut csr = Csr::with_sparse_shaped_capacity(shape, u16::try_from(entries.len()).unwrap());
    let mut sorted = entries.to_vec();
    sorted.sort_unstable_by_key(|&(row, column, _)| (row, column));
    for (row, column, value) in sorted {
        MatrixMut::add(&mut csr, (row, column, value)).expect("insert edge");
    }
    csr
}

#[test]
fn test_check_lap_sparse_wrapper_invariants_shared_error_path() {
    // Contains a negative edge, so both wrappers should reject the input.
    let csr = build_csr((2, 2), &[(0, 0, -1.0), (1, 1, 1.0)]);
    check_lap_sparse_wrapper_invariants(&csr);
}

#[test]
fn test_check_lap_sparse_wrapper_invariants_returns_when_padding_overflows() {
    // max value * 2.1 overflows to inf, hitting the early-return guard.
    let csr = build_csr((1, 1), &[(0, 0, f64::MAX)]);
    check_lap_sparse_wrapper_invariants(&csr);
}

#[test]
fn test_check_lap_square_invariants_returns_when_max_cost_overflows() {
    // max value * 2.1 overflows to inf, hitting the early-return guard.
    let csr = build_csr((1, 1), &[(0, 0, f64::MAX)]);
    check_lap_square_invariants(&csr);
}

#[test]
fn test_check_lap_square_invariants_unstable_numeric_range_path() {
    // Numerically unstable range should trigger the unstable branch and return
    // after validating any produced assignments.
    let csr = build_csr((2, 2), &[(0, 0, 1.0e-20), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0e20)]);
    check_lap_square_invariants(&csr);
}

#[test]
fn test_check_louvain_invariants_returns_for_non_square_input() {
    let csr = build_csr((2, 3), &[(0, 1, 1.0), (1, 2, 1.0)]);
    check_louvain_invariants(&csr);
}

#[test]
fn test_check_louvain_invariants_returns_for_empty_input() {
    let csr = Csr::with_sparse_shaped_capacity((0, 0), 0);
    check_louvain_invariants(&csr);
}
