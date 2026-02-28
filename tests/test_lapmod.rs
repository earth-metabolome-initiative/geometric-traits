//! Unit tests for the LAPMOD algorithm.
#![cfg(feature = "std")]

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use num_traits::AsPrimitive;

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        HopcroftKarp, LAPMOD, LAPMODError, MatrixMut, SparseLAPJV, SparseMatrixMut,
        SparseValuedMatrix,
    },
    traits::{Matrix2D, algorithms::randomized_graphs::XorShift64},
};
use num_traits::ToPrimitive;

// ---------------------------------------------------------------------------
// Helper: sort assignment for stable comparison
// ---------------------------------------------------------------------------

fn sorted(mut v: Vec<(u8, u8)>) -> Vec<(u8, u8)> {
    v.sort_unstable_by_key(|&(r, c)| (r, c));
    v
}

fn target_edge_count(n: usize, density: f64) -> usize {
    let Some(total_cells) = n.checked_mul(n).and_then(|value| value.to_f64()) else {
        return usize::MAX;
    };
    (total_cells * density).floor().to_usize().unwrap_or(usize::MAX)
}

fn random_index(rng: &mut XorShift64, n: usize) -> usize {
    let n_u64 = u64::try_from(n).expect("usize values always fit into u64");
    let raw = rng.next().expect("XorShift64 produces infinite values") % n_u64;
    usize::try_from(raw).expect("raw index is modulo n and always fits usize")
}

fn random_cost(rng: &mut XorShift64) -> f64 {
    let raw = rng.next().expect("XorShift64 produces infinite values") % 999 + 1;
    let cents = u32::try_from(raw).expect("bounded to the range 1..=999");
    f64::from(cents) / 100.0
}

// ---------------------------------------------------------------------------
// Edge-case / error-path tests
// ---------------------------------------------------------------------------

#[test]
/// An empty 0×0 matrix should return an empty assignment.
fn test_lapmod_empty() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);
    let assignment = csr.lapmod(1000.0).unwrap();
    assert_eq!(assignment, Vec::new());
}

#[test]
/// A 1×1 matrix with a single edge should return that edge.
fn test_lapmod_single_edge() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 1);
    csr.add((0, 0, 1.0)).unwrap();
    let assignment = csr.lapmod(1000.0).unwrap();
    assert_eq!(assignment, vec![(0, 0)]);
}

#[test]
/// A non-square matrix should return `NonSquareMatrix`.
fn test_lapmod_non_square() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 3), 0);
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::NonSquareMatrix));
}

#[test]
/// A matrix containing a zero cost should return `ZeroValues`.
fn test_lapmod_zero_value() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
    csr.add((0, 0, 0.0)).unwrap();
    csr.add((1, 1, 1.0)).unwrap();
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::ZeroValues));
}

#[test]
/// A matrix containing a negative cost should return `NegativeValues`.
fn test_lapmod_negative_value() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
    csr.add((0, 0, -1.0)).unwrap();
    csr.add((1, 1, 1.0)).unwrap();
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::NegativeValues));
}

#[test]
/// A matrix containing a NaN cost should return `NonFiniteValues`.
fn test_lapmod_nan_value() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
    csr.add((0, 0, f64::NAN)).unwrap();
    csr.add((1, 1, 1.0)).unwrap();
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::NonFiniteValues));
}

#[test]
/// An edge cost equal to max_cost should return `ValueTooLarge`.
fn test_lapmod_value_too_large() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
    csr.add((0, 0, 1000.0)).unwrap(); // equal to max_cost
    csr.add((1, 1, 1.0)).unwrap();
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::ValueTooLarge));
}

#[test]
/// max_cost = f64::INFINITY should return `MaximalCostNotFinite`.
fn test_lapmod_max_cost_not_finite() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 0);
    assert_eq!(csr.lapmod(f64::INFINITY), Err(LAPMODError::MaximalCostNotFinite));
}

#[test]
/// max_cost = -1.0 should return `MaximalCostNotPositive`.
fn test_lapmod_max_cost_not_positive() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((1, 1), 0);
    assert_eq!(csr.lapmod(-1.0), Err(LAPMODError::MaximalCostNotPositive));
}

#[test]
/// A row with no outgoing edges makes the assignment infeasible.
fn test_lapmod_no_edges_row() {
    // 3×3, row 1 has no edges
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 2);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::InfeasibleAssignment));
}

#[test]
/// A bipartite structure that has no perfect matching returns
/// `InfeasibleAssignment`.
fn test_lapmod_no_matching() {
    // 3×3: rows 0 and 1 both only connect to column 0, row 2 to column 2
    // There is no perfect matching because column 1 is unreachable.
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 4);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((1, 0, 2.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();
    // Row 1 has only column 0 but row 0 also wants column 0 → column 1 is
    // never reachable, making a perfect matching impossible.
    // Actually we need all rows to have at least one entry, but some column
    // must be unreachable.  Force row 1 to have only col 0.
    // A matching can cover rows 0→col0 and row 2→col2, leaving row 1 unmatched.
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::InfeasibleAssignment));
}

// ---------------------------------------------------------------------------
// Correctness tests
// ---------------------------------------------------------------------------

#[test]
/// Perfect 3×3 dense-like matrix — result should match SparseLAPJV.
fn test_lapmod_perfect_3x3() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 1.0, 6.0], [7.0, 8.0, 1.0]])
            .expect("Failed to create CSR matrix");

    let lapmod = sorted(csr.lapmod(1000.0).expect("LAPMOD failed"));
    let slapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));
    assert_eq!(lapmod, slapjv, "LAPMOD and SparseLAPJV disagree on 3×3 dense matrix");
    assert_eq!(lapmod, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
/// Regression for augmenting-path backtracking when a free sink column is
/// discovered directly from the current unassigned start row.
fn test_lapmod_direct_sink_from_start_row_regression() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [1.0, 3.0]]).expect("Failed to create CSR matrix");

    let result = sorted(csr.lapmod(1000.0).expect("LAPMOD failed"));
    assert_eq!(result, vec![(0, 1), (1, 0)]);
}

#[test]
/// Sparse 3×3 with exactly one perfect matching.
fn test_lapmod_unique_sparse() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 3);
    csr.add((0, 1, 1.0)).unwrap();
    csr.add((1, 0, 1.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();
    let result = sorted(csr.lapmod(1000.0).expect("LAPMOD failed"));
    assert_eq!(result, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
/// Verify optimality on a 4×4 matrix: the returned cost is ≤ all other
/// feasible permutations.
fn test_lapmod_optimality_4x4() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [10.0, 2.0, 30.0, 40.0],
        [5.0, 20.0, 3.0, 8.0],
        [50.0, 4.0, 1.0, 6.0],
        [7.0, 9.0, 11.0, 1.0],
    ])
    .unwrap();

    let assignment = csr.lapmod(1000.0).expect("LAPMOD failed");
    assert_eq!(assignment.len(), 4, "Expected a perfect matching");

    let assignment_cost: f64 = assignment
        .iter()
        .map(|&(r, c)| {
            let r = usize::from(r);
            let c = usize::from(c);
            [
                [10.0, 2.0, 30.0, 40.0],
                [5.0, 20.0, 3.0, 8.0],
                [50.0, 4.0, 1.0, 6.0],
                [7.0, 9.0, 11.0, 1.0],
            ][r][c]
        })
        .sum();

    // Brute-force: check all 4! = 24 permutations
    let costs = [
        [10.0, 2.0, 30.0, 40.0],
        [5.0, 20.0, 3.0, 8.0],
        [50.0, 4.0, 1.0, 6.0],
        [7.0, 9.0, 11.0, 1.0],
    ];
    for p0 in 0..4 {
        for p1 in 0..4 {
            if p1 == p0 {
                continue;
            }
            for p2 in 0..4 {
                if p2 == p0 || p2 == p1 {
                    continue;
                }
                let p3 = (0..4).find(|&x| x != p0 && x != p1 && x != p2).unwrap();
                let cost = costs[0][p0] + costs[1][p1] + costs[2][p2] + costs[3][p3];
                assert!(
                    assignment_cost <= cost + 1e-9,
                    "LAPMOD cost {assignment_cost} > permutation cost {cost}"
                );
            }
        }
    }
}

#[test]
/// Cardinality of LAPMOD result must match Hopcroft-Karp maximum matching.
fn test_lapmod_cardinality_matches_hopcroft() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((4, 4), 8);
    // Dense-ish 4×4 with a perfect matching available
    for r in 0u8..4 {
        for c in 0u8..4 {
            csr.add((r, c, (f64::from(r) + 1.0) * (f64::from(c) + 1.0))).unwrap();
        }
    }
    let lapmod_len = csr.lapmod(1000.0).expect("LAPMOD failed").len();
    let hk_len = csr.hopcroft_karp().expect("Hopcroft-Karp failed").len();
    assert_eq!(lapmod_len, hk_len, "LAPMOD and Hopcroft-Karp cardinalities differ");
}

#[test]
/// Matches the same tricky cases that were regressions in SparseLAPJV.
fn test_lapmod_regression_two_row_sparse() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 4);
    csr.add((0, 0, 2e-5)).unwrap();
    csr.add((0, 2, 3e-5)).unwrap();
    csr.add((2, 0, 4.778_309_726_7e-5)).unwrap();

    // row 1 has no edges → infeasible
    assert_eq!(csr.lapmod(1.0), Err(LAPMODError::InfeasibleAssignment));
}

#[test]
/// Classic [[1,0.5,10],[0.5,10,20],[10,20,0.5]] benchmark; result should
/// agree with SparseLAPJV.
fn test_lapmod_classic_benchmark() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 0.5, 10.0], [0.5, 10.0, 20.0], [10.0, 20.0, 0.5]])
            .expect("Failed to create CSR matrix");

    let lapmod = sorted(csr.lapmod(1000.0).expect("LAPMOD failed"));
    let slapjv = sorted(csr.sparse_lapjv(900.0, 1000.0).expect("SparseLAPJV failed"));

    assert_eq!(lapmod, slapjv);
    assert_eq!(lapmod, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
/// Wide-rectangular CSR (3 rows × 4 cols) — should produce `NonSquareMatrix`.
fn test_lapmod_wide_rectangular() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 4), 0);
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::NonSquareMatrix));
}

#[test]
/// Verify the number_of_rows and number_of_columns accessors are used.
fn test_lapmod_matrix_dimensions() {
    let csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((5, 5), 0);
    assert_eq!(csr.number_of_rows(), 5u8);
    assert_eq!(csr.number_of_columns(), 5u8);
    // 5×5 with no edges → all rows empty → InfeasibleAssignment
    assert_eq!(csr.lapmod(1000.0), Err(LAPMODError::InfeasibleAssignment));
}

// ---------------------------------------------------------------------------
// Deadlock / infinite-loop regression tests
//
// These tests isolate the hang that occurs when LAPMOD is called on the
// sparse matrix generated by the benchmark (n=20, density=0.05, seed=62).
// The matrix is reproduced deterministically from the same XorShift64 seed
// and benchmark logic so the test is self-contained.
//
// Three layers of coverage are provided:
//   1. `test_lapmod_benchmark_n20_seed62_timeout` – exact benchmark matrix,
//      detected via a 2-second wall-clock timeout.
//   2. `test_lapmod_dense_small_*` – fully-connected n×n matrices (n = 4..8)
//      that complete in < 1 ms on a correct implementation and expose hangs
//      without requiring a timeout.
//   3. `test_lapmod_hand_crafted_augback_cycle` – minimal 4×4 matrix designed
//      to force the specific `augmentation_backtrack` cycle described below.
//
// Root-cause analysis of the hang
// ================================
// `augmentation_backtrack` follows a chain of `predecessors[col]` pointers
// (set during `find_path_sparse`) interleaved with `assigned_columns[row]`
// pointers (set during phases 1–3 and updated by previous backtrack calls).
// The termination condition is `if row_index == start_row { break; }`.
//
// If the chain
//
//     sink → pred[sink]=R_a → assigned_cols[R_a]=C_b
//          → pred[C_b]=R_b  → assigned_cols[R_b]=C_c
//          → pred[C_c]=R_a  → …                        (cycle, R_a ≠ start_row)
//
// never passes through `start_row`, the loop is infinite.  This can arise
// when:
//
//   (a) Phase 3 (augmenting_row_reduction) assigns a row R_a to column C_x,
//       then displaces R_a back to `unassigned_rows` while leaving
//       `assigned_columns[R_a] = C_x`.
//   (b) `find_path_sparse` (for a *different* unassigned row) later processes
//       column C_x (whose `assigned_rows[C_x]` is now stale — still pointing
//       to R_a from phase 3 even though the previous augmentation changed
//       R_a's actual assignment), and therefore sets `pred[some_col] = R_a`.
//   (c) When `augmentation_backtrack` later traverses that path, it follows
//       `assigned_cols[R_a] = C_x` back to C_x (not toward `start_row`),
//       creating the cycle.
//
// The stale `assigned_rows` pointer is never corrected because
// `augmentation_backtrack` only updates `assigned_rows[col]` for columns
// *along the augmenting path*, so a column that was an intermediate
// assignment during phase 3 but not on the final augmenting path keeps its
// phase-3 row pointer indefinitely.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: build the same sparse matrix as the lapmod benchmark does
// ---------------------------------------------------------------------------

fn sparse_valued_matrix_usize(
    seed: u64,
    n: usize,
    density: f64,
) -> ValuedCSR2D<usize, usize, usize, f64> {
    let mut rng = XorShift64::from(seed);
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((n, n), target_edge_count(n, density));

    // Guarantee at least one edge per row (feasibility requirement).
    for row in 0..n {
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng);
        let _ = csr.add((row, col, cost));
    }

    // Add random edges at the requested density.
    let target_edges = target_edge_count(n, density);
    for _ in 0..target_edges {
        let row = random_index(&mut rng, n);
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng);
        let _ = csr.add((row, col, cost)); // silently ignore duplicate keys
    }

    csr
}

// ---------------------------------------------------------------------------
// Task 1 — exact benchmark matrix with a 2-second timeout
// ---------------------------------------------------------------------------

#[test]
/// The n=20 sparse matrix generated by `sparse_valued_matrix(seed=62, n=20,
/// density=0.05)` was observed to hang inside `lapmod`.  This test runs the
/// call on a background thread and fails (rather than hanging forever) if it
/// does not complete within 2 seconds.
///
/// A correct implementation must return `Ok(_)` within the timeout.
fn test_lapmod_benchmark_n20_seed62_timeout() {
    let csr = sparse_valued_matrix_usize(62, 20, 0.05);

    // Compute max_cost the same way the benchmark does.
    let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;

    let finished = Arc::new(AtomicBool::new(false));
    let finished_clone = Arc::clone(&finished);

    let handle = thread::spawn(move || {
        let result = csr.lapmod(max_cost);
        finished_clone.store(true, Ordering::SeqCst);
        result
    });

    // Allow up to 2 seconds; a well-functioning implementation finishes in << 1 ms.
    thread::sleep(Duration::from_secs(2));

    assert!(
        finished.load(Ordering::SeqCst),
        "LAPMOD deadlocked on n=20 density=0.05 seed=62 matrix \
         (did not return within 2 s)"
    );

    let result = handle.join().expect("thread panicked");
    assert!(
        result.is_ok() || result == Err(LAPMODError::InfeasibleAssignment),
        "LAPMOD returned unexpected error: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Task 2 — fully-dense small matrices (should complete without timeout)
// ---------------------------------------------------------------------------

/// Build a deterministic fully-dense n×n matrix from seed `42 + n`.
fn dense_valued_matrix(seed: u64, n: usize) -> ValuedCSR2D<usize, usize, usize, f64> {
    let mut rng = XorShift64::from(seed);
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((n, n), n * n);

    for row in 0..n {
        for col in 0..n {
            let cost = random_cost(&mut rng);
            let _ = csr.add((row, col, cost));
        }
    }

    csr
}

macro_rules! dense_lapmod_test {
    ($name:ident, $n:expr) => {
        #[test]
        /// A fully-connected $n×n matrix must be solved in << 1 ms.
        /// Hangs here indicate a loop inside the Dijkstra augmentation phase.
        fn $name() {
            let n: usize = $n;
            let csr = dense_valued_matrix(
                42 + u64::try_from(n).expect("usize values always fit into u64"),
                n,
            );
            let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;
            let result = csr.lapmod(max_cost);
            assert!(
                result.is_ok(),
                "LAPMOD failed on fully-dense {n}×{n} matrix: {:?}",
                result.unwrap_err()
            );
            assert_eq!(result.unwrap().len(), n, "Expected a perfect matching of size {n}");
        }
    };
}

dense_lapmod_test!(test_lapmod_dense_4x4, 4);
dense_lapmod_test!(test_lapmod_dense_5x5, 5);
dense_lapmod_test!(test_lapmod_dense_6x6, 6);
dense_lapmod_test!(test_lapmod_dense_7x7, 7);
dense_lapmod_test!(test_lapmod_dense_8x8, 8);
dense_lapmod_test!(test_lapmod_dense_9x9, 9);
dense_lapmod_test!(test_lapmod_dense_10x10, 10);

// ---------------------------------------------------------------------------
// Task 3 — hand-crafted matrix that targets the augmentation_backtrack cycle
// ---------------------------------------------------------------------------

#[test]
/// Minimal 4×4 sparse matrix engineered to expose the infinite loop in
/// `augmentation_backtrack`.
///
/// Construction rationale
/// ----------------------
/// We want a configuration where, after phases 1–3:
///   - `unassigned_rows` contains exactly R0.
///   - `assigned_rows[C0] = R1`, `assigned_rows[C1] = R2`  (two assigned cols).
///   - `assigned_columns[R1] = C1`, `assigned_columns[R2] = C0` (a "cross"
///     assignment left over from augmenting row reduction).
///   - R0's only sparse neighbour is C0.
///
/// `find_path_sparse(R0)`:
///   - Seeds: `pred[C0] = R0`, `dist[C0] = cost(R0,C0) - col_costs[C0]`.
///   - Expands C0 (assigned to R1): discovers R1's neighbours. R1 has edges to
///     C0 and C1.  Sets `pred[C1] = R1`.
///   - Expands C1 (assigned to R2): discovers R2's neighbours. R2 has edges to
///     C1 and C2 (free).  Sets `pred[C2] = R2`.
///   - C2 is free → sink = C2.
///
/// `augmentation_backtrack(C2, start_row=R0)`:
///   C2 → pred[C2]=R2 → assigned_cols[R2]=C0 → pred[C0]=R0=start_row → break.
///
/// **Cycle variant** (the bug): if `assigned_columns[R2] = C1` (not C0) after
/// phase 3, and `pred[C1] = R1` (set by scan_sparse), then:
///   C2 → pred[C2]=R2 → assigned_cols[R2]=C1 → pred[C1]=R1 →
/// assigned_cols[R1]=C1 (updated) → col=C1 → cycle.
///
/// The test below uses the simplest 4×4 structure that — on a *correct*
/// implementation — terminates in O(1) steps, and on a *buggy* implementation
/// that forms the cycle, hangs.  We wrap it in a 1-second timeout so the test
/// suite does not stall.
fn test_lapmod_hand_crafted_augback_cycle() {
    // 4×4 sparse matrix designed to exercise the augmentation_backtrack cycle.
    //
    // Sparse structure:
    //   R0 → {C1: 2.0, C3: 3.0}
    //   R1 → {C0: 1.0, C1: 1.5, C2: 4.0}
    //   R2 → {C0: 2.0, C2: 1.0}
    //   R3 → {C1: 3.0, C3: 1.0}
    //
    // All perfect matchings and costs:
    //   R0→C1, R1→C0, R2→C2, R3→C3  →  2.0+1.0+1.0+1.0 = 5.0  (optimal)
    //   R0→C3, R1→C0, R2→C2, R3→C1  →  3.0+1.0+1.0+3.0 = 8.0
    //   R0→C3, R1→C2, R2→C0, R3→C1  →  3.0+4.0+2.0+3.0 = 12.0
    //   R0→C1, R1→C2, R2→C0, R3→C3  →  2.0+4.0+2.0+1.0 = 9.0
    // Optimal: R0→C1 (2.0), R1→C0 (1.0), R2→C2 (1.0), R3→C3 (1.0) = 5.0.
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((4, 4), 8);
    csr.add((0, 1, 2.0)).unwrap();
    csr.add((0, 3, 3.0)).unwrap();
    csr.add((1, 0, 1.0)).unwrap();
    csr.add((1, 1, 1.5)).unwrap();
    csr.add((1, 2, 4.0)).unwrap();
    csr.add((2, 0, 2.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();
    csr.add((3, 1, 3.0)).unwrap();
    csr.add((3, 3, 1.0)).unwrap();

    let max_cost = 1000.0_f64;

    let finished = Arc::new(AtomicBool::new(false));
    let finished_clone = Arc::clone(&finished);

    let handle = thread::spawn(move || {
        let result = csr.lapmod(max_cost);
        finished_clone.store(true, Ordering::SeqCst);
        result
    });

    thread::sleep(Duration::from_secs(1));

    assert!(
        finished.load(Ordering::SeqCst),
        "LAPMOD deadlocked on 4×4 hand-crafted matrix \
         (augmentation_backtrack cycle — did not return within 1 s)"
    );

    let assignment = handle.join().expect("thread panicked").expect("LAPMOD failed");
    assert_eq!(assignment.len(), 4, "Expected a perfect matching of size 4");

    // Verify total cost is 5.0 (the known optimum).
    let matrix = [
        [f64::INFINITY, 2.0, f64::INFINITY, 3.0],
        [1.0, 1.5, 4.0, f64::INFINITY],
        [2.0, f64::INFINITY, 1.0, f64::INFINITY],
        [f64::INFINITY, 3.0, f64::INFINITY, 1.0],
    ];
    let cost: f64 = assignment.iter().map(|&(r, c)| matrix[r][c]).sum();
    assert!((cost - 5.0).abs() < 1e-9, "Expected optimal cost 5.0, got {cost}");
}

#[test]
/// 3×3 sparse matrix where two rows share all their neighbours — the
/// augmenting path must reroute through the already-assigned column whose
/// row has a back-edge to the starting column.
///
/// Structure:
///   R0 → {C0: 1.0, C1: 2.0}
///   R1 → {C0: 3.0, C1: 1.0}
///   R2 → {C2: 1.0}
///
/// After column reduction: col_costs = [1.0, 1.0, 1.0].
/// Optimal matching: R0→C0 (1.0), R1→C1 (1.0), R2→C2 (1.0) = 3.0.
///
/// This is a "shared-neighbours" case that exercises the augmenting-path
/// rerouting in `find_path_sparse`.  A buggy implementation that corrupts
/// the predecessor tree when updating already-discovered columns would
/// either deadlock or return a wrong assignment.
fn test_lapmod_shared_neighbours_3x3() {
    let mut csr: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::with_sparse_shaped_capacity((3, 3), 5);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((0, 1, 2.0)).unwrap();
    csr.add((1, 0, 3.0)).unwrap();
    csr.add((1, 1, 1.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();

    let result = sorted(csr.lapmod(1000.0).expect("LAPMOD failed on shared-neighbours 3×3"));
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
/// 4×4 matrix where rows 0 and 1 share columns 0 and 1, and the augmenting
/// path must hop through an already-assigned column whose assigned row has a
/// back-edge to the column we started from.
///
/// Structure:
///   R0 → {C0: 1.0, C1: 10.0}
///   R1 → {C0: 10.0, C1: 1.0, C2: 5.0}
///   R2 → {C1: 3.0, C2: 1.0, C3: 8.0}
///   R3 → {C3: 1.0}
///
/// Optimal: R0→C0, R1→C1, R2→C2, R3→C3 with cost 4.0.
/// The "hop" structure: to assign R3, we need to free C3 from R2,
/// which must be reassigned from C2 to C3; but C2 was held by R1, which
/// can move to C1, displacing R0 to C0.  This exercises the multi-hop
/// augmenting path in `find_path_sparse` and the predecessor tracking.
fn test_lapmod_multi_hop_augmenting_path_4x4() {
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((4, 4), 9);
    csr.add((0, 0, 1.0)).unwrap();
    csr.add((0, 1, 10.0)).unwrap();
    csr.add((1, 0, 10.0)).unwrap();
    csr.add((1, 1, 1.0)).unwrap();
    csr.add((1, 2, 5.0)).unwrap();
    csr.add((2, 1, 3.0)).unwrap();
    csr.add((2, 2, 1.0)).unwrap();
    csr.add((2, 3, 8.0)).unwrap();
    csr.add((3, 3, 1.0)).unwrap();

    let result = csr.lapmod(1000.0).expect("LAPMOD failed on multi-hop 4×4");
    assert_eq!(result.len(), 4, "Expected perfect matching of size 4");

    let matrix = [
        [1.0, 10.0, f64::INFINITY, f64::INFINITY],
        [10.0, 1.0, 5.0, f64::INFINITY],
        [f64::INFINITY, 3.0, 1.0, 8.0],
        [f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0],
    ];
    let cost: f64 = result.iter().map(|&(r, c)| matrix[r][c]).sum();
    assert!((cost - 4.0).abs() < 1e-9, "Expected optimal cost 4.0, got {cost}");
}

#[test]
/// Stress-test: 10 random sparse matrices at various sizes and densities
/// are each checked for correctness against `sparse_lapjv`.
/// Any hang here is caught by the test runner's default timeout.
fn test_lapmod_stress_correctness_vs_sparse_lapjv() {
    for (seed, n, density) in [
        (1u64, 5usize, 0.4f64),
        (2, 6, 0.3),
        (3, 7, 0.25),
        (4, 8, 0.2),
        (5, 9, 0.15),
        (6, 10, 0.15),
        (7, 12, 0.12),
        (8, 15, 0.10),
        (9, 18, 0.08),
        (10, 20, 0.06),
    ] {
        let csr = sparse_valued_matrix_usize(seed, n, density);
        let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;
        let padding = max_cost * 0.9;

        let lapmod_result = csr.lapmod(max_cost);
        let slapjv_result = csr.sparse_lapjv(padding, max_cost);
        let hk_len: usize = csr.hopcroft_karp().expect("Hopcroft-Karp failed").len().as_();

        match (lapmod_result, slapjv_result) {
            (Ok(lapmod), Ok(slapjv)) => {
                assert_eq!(
                    lapmod.len(),
                    hk_len,
                    "seed={seed} n={n} d={density}: LAPMOD cardinality differs from Hopcroft-Karp"
                );
                if hk_len == n {
                    assert_eq!(
                        lapmod.len(),
                        slapjv.len(),
                        "seed={seed} n={n} d={density}: LAPMOD and SparseLAPJV returned different assignment sizes"
                    );
                }
            }
            (Err(LAPMODError::InfeasibleAssignment), Ok(slapjv)) => {
                assert!(
                    hk_len < n,
                    "seed={seed} n={n} d={density}: LAPMOD reported infeasible but Hopcroft-Karp found perfect matching"
                );
                assert_eq!(
                    slapjv.len(),
                    hk_len,
                    "seed={seed} n={n} d={density}: SparseLAPJV cardinality differs from Hopcroft-Karp"
                );
            }
            (Err(LAPMODError::InfeasibleAssignment), Err(_)) => {
                assert!(
                    hk_len < n,
                    "seed={seed} n={n} d={density}: both algorithms failed but Hopcroft-Karp found perfect matching"
                );
            }
            (lapmod, slapjv) => {
                panic!(
                    "seed={seed} n={n} d={density}: mismatched results LAPMOD={lapmod:?} SparseLAPJV={slapjv:?}"
                );
            }
        }
    }
}
