//! Fuzz harness for the Eades-Lin-Smyth (GR) greedy feedback-arc-set heuristic.
//!
//! The arbitrary `ValuedCSR2D` is sanitized into a small-to-medium weighted
//! *square* directed graph with finite, positive, numerically-normal weights,
//! so the call is always expected to succeed. The harness then asserts the
//! invariants that must hold for *any* such input (the fuzzing oracles):
//!
//! * the call never panics (running to completion is the baseline oracle);
//! * `order` is a permutation of `0..n` and `positions` is its inverse;
//! * `feedback_edges` are exactly the non-self-loop edges `(u, v)` with
//!   `positions[u] > positions[v]`, and `feedback_weight` is their summed
//!   weight (within an absolute floating-point tolerance);
//! * `0 <= feedback_weight <= total_weight` and `tangle_fraction in [0, 1]`;
//! * the residual graph (input edges minus feedback edges) is acyclic, cross
//!   checked against `Kahn` on the kept edges;
//! * `is_acyclic()` agrees with whether `Kahn` succeeds on the full graph;
//! * the result is deterministic across two consecutive calls.
//!
//! `eades_lin_smyth()` consumes a square bidirectional valued matrix, so the
//! sanitized graph is always square with valid weights and its `Err` arm is
//! unreachable; it is asserted as such.

use std::collections::HashSet;

use geometric_traits::prelude::*;
use honggfuzz::fuzz;

/// Arbitrary input matrix type, mirroring the other valued-graph fuzz targets.
type Input = ValuedCSR2D<u16, u8, u8, f64>;
/// Concrete weighted square matrix backing the bidirectional matrix the
/// algorithm runs on.
type Weighted = ValuedCSR2D<u8, u8, u8, f64>;
/// Bidirectional wrapper the algorithm actually consumes.
type WeightedBi = GenericBiMatrix2D<Weighted, Weighted>;
/// Unvalued square matrix used to cross-check acyclicity with `Kahn`,
/// mirroring the directed-graph builder used elsewhere in the fuzz suite.
type Unweighted = SquareCSR2D<CSR2D<usize, usize, usize>>;

/// Absolute tolerance for floating-point weight comparisons.
const TOLERANCE: f64 = 1e-9;

/// Float comparison robust to legitimate overflow. Both the heuristic and this
/// oracle accumulate positive weights, so near-`f64::MAX` inputs can overflow
/// both sums to the *same* infinity; `==` makes `inf == inf` pass (a naive
/// `(inf - inf).abs()` is `NaN`, which would spuriously fail), and the tolerance
/// covers the finite case.
fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= TOLERANCE
}

/// Builds a square directed graph from the arbitrary matrix, keeping only
/// finite, positive, numerically-normal weights and deduplicating parallel
/// edges. Returns `None` when no usable edge survives or the shape is
/// unsuitable (non-square, empty, or larger than `u8::MAX` nodes).
///
/// This deliberately mirrors `positive_directed_graph` in the crate's
/// `test_utils`, so the resulting matrix is always square with valid weights
/// and `eades_lin_smyth` is therefore expected to return `Ok`.
fn positive_square_graph(csr: &Input) -> Option<(u8, Vec<(u8, u8, f64)>)> {
    let rows: usize = usize::from(csr.number_of_rows());
    let cols: usize = usize::from(csr.number_of_columns());
    if rows != cols || rows == 0 || rows > u8::MAX as usize {
        return None;
    }

    let n = u8::try_from(rows).ok()?;

    let mut edges: Vec<(u8, u8, f64)> = Vec::new();
    for row in csr.row_indices() {
        let r: usize = usize::from(row);
        if r >= rows {
            continue;
        }
        for (col, val) in csr.sparse_row(row).zip(csr.sparse_row_values(row)) {
            let c: usize = usize::from(col);
            if c < cols && val.is_finite() && val.is_normal() && val > 0.0 {
                let (Ok(r8), Ok(c8)) = (u8::try_from(r), u8::try_from(c)) else {
                    continue;
                };
                edges.push((r8, c8, val));
            }
        }
    }

    if edges.is_empty() {
        return None;
    }

    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    edges.dedup_by(|(r1, c1, _), (r2, c2, _)| (*r1, *c1) == (*r2, *c2));

    Some((n, edges))
}

/// Builds the bidirectional weighted square matrix the heuristic consumes.
fn build_weighted(n: u8, edges: &[(u8, u8, f64)]) -> Option<WeightedBi> {
    let edge_count = u8::try_from(edges.len()).ok()?;
    let matrix: Weighted = GenericEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape((n, n))
        .edges(edges.iter().copied())
        .build()
        .ok()?;
    Some(GenericBiMatrix2D::new(matrix))
}

/// Builds an unvalued square matrix from `(source, destination)` pairs for the
/// `Kahn` acyclicity cross-check. The pairs must already be sorted and unique.
fn build_unweighted(n: u8, pairs: &[(usize, usize)]) -> Option<Unweighted> {
    DiEdgesBuilder::default()
        .expected_number_of_edges(pairs.len())
        .expected_shape(n as usize)
        .edges(pairs.iter().copied())
        .build()
        .ok()
}

/// Runs every invariant assertion on the sanitized graph.
fn check(n: u8, edges: &[(u8, u8, f64)]) {
    let nodes = n as usize;
    let Some(matrix) = build_weighted(n, edges) else {
        return;
    };

    let result = matrix
        .eades_lin_smyth()
        .expect("eades_lin_smyth must succeed on a square graph with finite positive weights");

    let order = result.order();
    let positions = result.positions();

    // `order` must be a permutation of `0..n` and `positions` its exact inverse.
    assert_eq!(order.len(), nodes, "order length must equal node count");
    assert_eq!(positions.len(), nodes, "positions length must equal node count");
    let mut seen = vec![false; nodes];
    for (position, &node) in order.iter().enumerate() {
        assert!(node < nodes, "order entry {node} out of range 0..{nodes}");
        assert!(!seen[node], "order entry {node} appears twice (not a permutation)");
        seen[node] = true;
        assert_eq!(positions[node], position, "positions must invert order");
    }

    // The non-self-loop edges of the (deduplicated) input graph, and the exact
    // feedback set / weight recomputed independently from `positions`.
    let mut expected_feedback: Vec<(usize, usize)> = Vec::new();
    let mut expected_feedback_weight = 0.0_f64;
    let mut expected_total_weight = 0.0_f64;
    let mut full_pairs: Vec<(usize, usize)> = Vec::new();
    for &(source, destination, weight) in edges {
        if source == destination {
            // Self-loops are dropped by the algorithm and excluded everywhere.
            continue;
        }
        let (s, d) = (source as usize, destination as usize);
        expected_total_weight += weight;
        full_pairs.push((s, d));
        if positions[s] > positions[d] {
            expected_feedback.push((s, d));
            expected_feedback_weight += weight;
        }
    }

    // `feedback_edges` must be exactly the backward edges. Both sides are sorted
    // before comparison because the algorithm emits them in row-major order,
    // which already matches our edge iteration, but sorting makes the oracle
    // independent of that emission order.
    let mut actual_feedback: Vec<(usize, usize)> = result.feedback_edges().to_vec();
    actual_feedback.sort_unstable();
    expected_feedback.sort_unstable();
    assert_eq!(actual_feedback, expected_feedback, "feedback edge set mismatch");

    assert!(
        close(result.feedback_weight(), expected_feedback_weight),
        "feedback weight {} != recomputed {}",
        result.feedback_weight(),
        expected_feedback_weight
    );
    assert!(
        close(result.total_weight(), expected_total_weight),
        "total weight {} != recomputed {}",
        result.total_weight(),
        expected_total_weight
    );

    // Weight and fraction bounds.
    let feedback_weight = result.feedback_weight();
    let total_weight = result.total_weight();
    assert!(feedback_weight >= -TOLERANCE, "feedback weight must be non-negative");
    assert!(
        feedback_weight <= total_weight + TOLERANCE,
        "feedback weight {feedback_weight} must not exceed total weight {total_weight}"
    );
    // The tangle fraction is only well-defined in [0, 1] when the total weight is
    // finite. With near-`f64::MAX` weights the sums can overflow to infinity, so
    // the fraction becomes inf/inf = NaN; that is an inherent f64 accumulation
    // limit on pathological input, not an algorithm defect, so the bound is only
    // asserted for finite totals.
    if total_weight.is_finite() {
        let fraction = result.tangle_fraction();
        assert!(
            (-TOLERANCE..=1.0 + TOLERANCE).contains(&fraction),
            "tangle fraction {fraction} out of [0, 1]"
        );
    }

    // `is_acyclic()` must match "no feedback edges".
    assert_eq!(
        result.is_acyclic(),
        result.feedback_edges().is_empty(),
        "is_acyclic must agree with an empty feedback set"
    );

    // Cross-check that removing the algorithm's OWN reported feedback edges leaves
    // an acyclic graph -- the defining property of a feedback arc set. The residual
    // is built from `result.feedback_edges()` (not from `positions`), so this can
    // genuinely fail on a cut that does not break every cycle.
    let cut: HashSet<(usize, usize)> = result.feedback_edges().iter().copied().collect();
    let mut residual_pairs: Vec<(usize, usize)> =
        full_pairs.iter().copied().filter(|pair| !cut.contains(pair)).collect();
    residual_pairs.sort_unstable();
    residual_pairs.dedup();
    let residual = build_unweighted(n, &residual_pairs)
        .expect("residual graph of a sanitized square input must build");
    assert!(
        residual.kahn().is_ok(),
        "the residual graph (input minus feedback edges) must be acyclic"
    );

    // Cross-check is_acyclic() against Kahn on the full non-self-loop graph:
    // GR reports zero feedback edges iff the graph is already a DAG.
    full_pairs.sort_unstable();
    full_pairs.dedup();
    let full = build_unweighted(n, &full_pairs)
        .expect("full graph of a sanitized square input must build");
    assert_eq!(
        result.is_acyclic(),
        full.kahn().is_ok(),
        "is_acyclic must agree with Kahn on the full graph"
    );

    // Determinism: a second run must reproduce the arrangement and feedback set.
    let second = matrix
        .eades_lin_smyth()
        .expect("eades_lin_smyth must remain successful on the second call");
    assert_eq!(result.order(), second.order(), "order must be deterministic");
    assert_eq!(
        result.feedback_edges(),
        second.feedback_edges(),
        "feedback edges must be deterministic"
    );
    assert!(
        close(result.feedback_weight(), second.feedback_weight()),
        "feedback weight must be deterministic"
    );
}

fn main() {
    loop {
        fuzz!(|csr: Input| {
            // The heuristic consumes a square bidirectional valued matrix, which
            // the raw arbitrary input is not, so it is sanitized into one first.
            // Running the full oracle to completion (no panic) is the baseline.
            if let Some((n, edges)) = positive_square_graph(&csr) {
                check(n, &edges);
            }
        });
    }
}
