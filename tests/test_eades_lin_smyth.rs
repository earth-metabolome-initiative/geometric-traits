//! Tests for the Eades-Lin-Smyth (GR) greedy feedback-arc-set heuristic.
//!
//! GR builds a linear arrangement of the vertices of a weighted directed graph
//! by peeling sinks to the right, sources to the left, and otherwise placing
//! the maximum `out_weight - in_weight` vertex next on the left. The feedback
//! (back) edges are then the edges that point backward in that arrangement, and
//! their total weight is the tangle measure.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericBiMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::{EadesLinSmyth, EadesLinSmythError, algorithms::eades_lin_smyth::EadesLinSmythResult},
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;
type WeightedBiMatrix = GenericBiMatrix2D<WeightedMatrix, WeightedMatrix>;

const TOLERANCE: f64 = 1e-12;

/// Builds a weighted bidirectional directed graph from `node_count` and
/// `(source, dest, weight)` triples.
fn build(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WeightedBiMatrix {
    let mut edges = edges;
    edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericBiMatrix2D::new(matrix)
}

fn arrange(node_count: usize, edges: Vec<(usize, usize, f64)>) -> EadesLinSmythResult {
    build(node_count, edges).eades_lin_smyth().unwrap()
}

#[test]
fn test_dag_has_no_feedback() {
    // A simple chain 0 -> 1 -> 2 -> 3 is already acyclic, so GR must leave no
    // backward edges and recover a topological order.
    let result = arrange(4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);

    assert!(result.feedback_edges().is_empty());
    assert!(result.feedback_weight().abs() < TOLERANCE);
    assert!((result.total_weight() - 3.0).abs() < TOLERANCE);
    assert!(result.tangle_fraction().abs() < TOLERANCE);
    assert!(result.is_acyclic());

    let positions = result.positions();
    assert!(positions[0] < positions[1]);
    assert!(positions[1] < positions[2]);
    assert!(positions[2] < positions[3]);
}

#[test]
fn test_single_cycle_leaves_one_back_edge() {
    // 0 -> 1 -> 2 -> 0 is a single 3-cycle. GR (smallest-id tie-break) peels
    // node 0 first, so the closing edge 2 -> 0 is the lone feedback edge.
    let result = arrange(3, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);

    assert_eq!(result.feedback_edges(), &[(2, 0)]);
    assert!((result.feedback_weight() - 1.0).abs() < TOLERANCE);
    assert!((result.total_weight() - 3.0).abs() < TOLERANCE);
    assert!((result.tangle_fraction() - 1.0 / 3.0).abs() < TOLERANCE);
    assert!(!result.is_acyclic());
}

#[test]
fn test_weighted_facade_only_dispatcher_edges_are_feedback() {
    // Facade crate: root (node 0) defines shared types that both leaves (1, 2)
    // reference heavily (leaf -> root, weight 10), while the root's dispatcher
    // calls each leaf lightly (root -> leaf, weight 1). GR places the leaves on
    // the left and the root last, so only the two light dispatcher calls point
    // backward: usage is free, only the genuine back-references are penalized.
    let result = arrange(3, vec![(1, 0, 10.0), (2, 0, 10.0), (0, 1, 1.0), (0, 2, 1.0)]);

    assert_eq!(result.feedback_edges(), &[(0, 1), (0, 2)]);
    assert!((result.feedback_weight() - 2.0).abs() < TOLERANCE);
    assert!((result.total_weight() - 22.0).abs() < TOLERANCE);
    assert!((result.tangle_fraction() - 2.0 / 22.0).abs() < TOLERANCE);
}

#[test]
fn test_source_feeding_a_cycle_is_peeled_left() {
    // Node 0 is a pure source (no in-edges) that feeds a downstream 2-cycle
    // 1 <-> 2. The cycle keeps 1 and 2 from draining as sinks, so GR can only
    // make progress by peeling the source 0 to the left. Its edge 0 -> 1 then
    // points forward and the lone back edge is the one closing the 1 <-> 2 cycle.
    let result = arrange(3, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 1, 1.0)]);

    assert_eq!(result.positions()[0], 0);
    assert_eq!(result.feedback_edges(), &[(2, 1)]);
    assert!((result.feedback_weight() - 1.0).abs() < TOLERANCE);
    assert!(!result.is_acyclic());
}

#[test]
fn test_pure_sink_is_free() {
    // A shared utility (node 0) that everyone calls but that calls nobody is a
    // pure sink: it can never produce a backward edge, so the graph is acyclic.
    let result = arrange(4, vec![(1, 0, 5.0), (2, 0, 3.0), (3, 0, 2.0)]);

    assert!(result.is_acyclic());
    assert!(result.feedback_edges().is_empty());
    assert!((result.total_weight() - 10.0).abs() < TOLERANCE);
}

#[test]
fn test_self_loops_are_ignored() {
    // A self-loop is intra-module noise: it can never point backward and must
    // not contribute to the total weight either.
    let result = arrange(2, vec![(0, 0, 5.0), (0, 1, 1.0)]);

    assert!(result.is_acyclic());
    assert!(result.feedback_edges().is_empty());
    assert!((result.total_weight() - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_self_loop_with_invalid_weight_is_still_ignored() {
    // Self-loops are dropped before weight validation, so a self-loop carrying a
    // non-finite or non-positive weight is ignored entirely rather than rejected.
    let nan_loop = build(2, vec![(0, 0, f64::NAN), (0, 1, 1.0)]).eades_lin_smyth().unwrap();
    assert!(nan_loop.is_acyclic());
    assert!((nan_loop.total_weight() - 1.0).abs() < TOLERANCE);

    let negative_loop = build(2, vec![(1, 1, -3.0), (0, 1, 1.0)]).eades_lin_smyth().unwrap();
    assert!(negative_loop.is_acyclic());
    assert!((negative_loop.total_weight() - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_single_node_graph_is_vacuously_acyclic() {
    let result = arrange(1, Vec::new());

    assert_eq!(result.order(), &[0]);
    assert!(result.feedback_edges().is_empty());
    assert!(result.total_weight().abs() < TOLERANCE);
    assert!(result.tangle_fraction().abs() < TOLERANCE);
    assert!(result.is_acyclic());
}

#[test]
fn test_arrangement_is_deterministic() {
    // Two disjoint 2-cycles. Repeated runs must yield byte-identical output.
    let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
    let first = arrange(4, edges.clone());
    let second = arrange(4, edges);

    assert_eq!(first.order(), second.order());
    assert_eq!(first.feedback_edges(), second.feedback_edges());
    assert!((first.feedback_weight() - second.feedback_weight()).abs() < TOLERANCE);
}

#[test]
fn test_complete_mutual_tangle_keeps_half_the_weight() {
    // Every ordered pair of four nodes has an edge in both directions: a maximal
    // mutual tangle. For any linear arrangement exactly one edge of each pair
    // points backward, so a genuine tangle stays pinned at a tangle fraction of
    // 0.5 no matter how GR orders it (the positive control the metric needs).
    let mut edges = Vec::new();
    for source in 0..4 {
        for destination in 0..4 {
            if source != destination {
                edges.push((source, destination, 1.0));
            }
        }
    }
    let result = arrange(4, edges);

    assert!((result.total_weight() - 12.0).abs() < TOLERANCE);
    assert_eq!(result.feedback_edges().len(), 6);
    assert!((result.tangle_fraction() - 0.5).abs() < TOLERANCE);
}

// Parity fixtures: expected cuts for the GR heuristic, cross-checked against an
// independent reference implementation of the same algorithm (same smallest-id
// max-delta tie-break). The feedback weight is invariant across any valid GR
// ordering; where our ordering is also uniquely determined we assert it too.

#[test]
fn test_parity_four_cycle() {
    // Reference: order [0, 1, 2, 3], cut {3 -> 0}, weight 1.
    let result = arrange(4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]);

    assert_eq!(result.order(), &[0, 1, 2, 3]);
    assert_eq!(result.feedback_edges(), &[(3, 0)]);
    assert!((result.feedback_weight() - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_parity_two_disjoint_three_cycles() {
    // One back edge per cycle is cut: {2 -> 0, 5 -> 3}, weight 2.
    let result = arrange(
        6,
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 3, 1.0)],
    );

    assert_eq!(result.feedback_edges(), &[(2, 0), (5, 3)]);
    assert!((result.feedback_weight() - 2.0).abs() < TOLERANCE);
}

#[test]
fn test_parity_weighted_facade_with_pure_source() {
    // Leaves (1, 2) reference the root (0) heavily (weight 5); the root's
    // dispatcher calls each leaf lightly (weight 1); node 3 is a pure source
    // feeding the root. Reference: order [3, 1, 2, 0], cut {0 -> 1, 0 -> 2},
    // weight 2.
    let result = arrange(4, vec![(1, 0, 5.0), (2, 0, 5.0), (3, 0, 5.0), (0, 1, 1.0), (0, 2, 1.0)]);

    assert_eq!(result.order(), &[3, 1, 2, 0]);
    assert_eq!(result.feedback_edges(), &[(0, 1), (0, 2)]);
    assert!((result.feedback_weight() - 2.0).abs() < TOLERANCE);
}

#[test]
fn test_parity_weighted_cycle_no_ties() {
    // A 3-cycle with one heavy edge: deltas are -9 / 0 / 9, so there are NO ties
    // and the ordering is unique regardless of tie-break rule. Reference: order
    // [2, 0, 1], cut {1 -> 2}, keeping the heavy 2 -> 0 (weight 10) forward.
    // The strongest cross-implementation check, since it is entirely
    // tie-break-insensitive.
    let result = arrange(3, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 10.0)]);

    assert_eq!(result.order(), &[2, 0, 1]);
    assert_eq!(result.feedback_edges(), &[(1, 2)]);
    assert!((result.feedback_weight() - 1.0).abs() < TOLERANCE);
    assert!((result.total_weight() - 12.0).abs() < TOLERANCE);
}

// Larger parity fixtures with distinct integer weights, so every max-delta
// choice is unique and the cut is tie-break-independent: our heuristic and an
// independent reference implementation must agree exactly.
struct ParityCase {
    nodes: usize,
    edges: &'static [(usize, usize, f64)],
    cut: &'static [(usize, usize)],
    feedback_weight: f64,
    total_weight: f64,
}

const RANDOM_PARITY_CASES: &[ParityCase] = &[
    // 12 nodes, 30 edges (seed 12).
    ParityCase {
        nodes: 12,
        edges: &[
            (0, 1, 14.0),
            (0, 2, 23.0),
            (0, 8, 11.0),
            (0, 10, 17.0),
            (1, 8, 30.0),
            (1, 10, 3.0),
            (2, 4, 22.0),
            (2, 5, 29.0),
            (2, 8, 4.0),
            (3, 2, 12.0),
            (3, 4, 26.0),
            (3, 11, 13.0),
            (4, 3, 21.0),
            (5, 1, 2.0),
            (5, 3, 9.0),
            (5, 7, 25.0),
            (6, 2, 18.0),
            (6, 5, 15.0),
            (6, 8, 27.0),
            (6, 11, 28.0),
            (7, 2, 8.0),
            (7, 5, 10.0),
            (7, 8, 24.0),
            (8, 0, 19.0),
            (8, 1, 20.0),
            (8, 10, 5.0),
            (9, 4, 6.0),
            (10, 0, 7.0),
            (11, 0, 16.0),
            (11, 4, 1.0),
        ],
        cut: &[(3, 2), (4, 3), (7, 2), (7, 5), (8, 0), (8, 1), (10, 0), (11, 0)],
        feedback_weight: 113.0,
        total_weight: 465.0,
    },
    // 20 nodes, 55 edges (seed 20).
    ParityCase {
        nodes: 20,
        edges: &[
            (0, 12, 42.0),
            (0, 14, 50.0),
            (0, 18, 11.0),
            (2, 0, 33.0),
            (2, 4, 8.0),
            (2, 5, 48.0),
            (2, 10, 53.0),
            (2, 11, 36.0),
            (2, 14, 5.0),
            (2, 15, 30.0),
            (3, 4, 40.0),
            (3, 8, 13.0),
            (4, 1, 44.0),
            (4, 8, 18.0),
            (4, 11, 21.0),
            (5, 2, 10.0),
            (5, 8, 23.0),
            (5, 9, 28.0),
            (5, 11, 26.0),
            (5, 12, 3.0),
            (5, 18, 37.0),
            (6, 5, 16.0),
            (6, 11, 55.0),
            (6, 16, 29.0),
            (6, 18, 9.0),
            (7, 0, 25.0),
            (7, 3, 1.0),
            (7, 13, 22.0),
            (8, 4, 51.0),
            (8, 11, 6.0),
            (8, 12, 43.0),
            (8, 16, 15.0),
            (8, 17, 2.0),
            (9, 0, 17.0),
            (10, 18, 47.0),
            (10, 19, 54.0),
            (11, 1, 31.0),
            (11, 9, 49.0),
            (12, 2, 46.0),
            (12, 15, 52.0),
            (13, 4, 38.0),
            (13, 19, 12.0),
            (15, 8, 27.0),
            (15, 12, 41.0),
            (16, 15, 20.0),
            (16, 17, 14.0),
            (16, 18, 34.0),
            (17, 2, 19.0),
            (18, 3, 35.0),
            (18, 6, 7.0),
            (18, 9, 32.0),
            (18, 11, 45.0),
            (18, 13, 39.0),
            (19, 0, 4.0),
            (19, 9, 24.0),
        ],
        cut: &[(0, 18), (4, 8), (5, 2), (12, 2), (15, 8), (15, 12), (16, 18), (17, 2), (18, 6)],
        feedback_weight: 213.0,
        total_weight: 1540.0,
    },
    // 30 nodes, 90 edges (seed 30).
    ParityCase {
        nodes: 30,
        edges: &[
            (0, 8, 71.0),
            (0, 21, 54.0),
            (0, 25, 25.0),
            (0, 27, 59.0),
            (1, 0, 2.0),
            (1, 2, 55.0),
            (1, 21, 58.0),
            (2, 12, 64.0),
            (2, 17, 80.0),
            (2, 25, 3.0),
            (2, 26, 79.0),
            (3, 11, 72.0),
            (3, 19, 66.0),
            (3, 28, 46.0),
            (4, 6, 13.0),
            (4, 17, 47.0),
            (4, 22, 68.0),
            (4, 26, 36.0),
            (5, 19, 16.0),
            (6, 10, 67.0),
            (6, 17, 61.0),
            (7, 13, 12.0),
            (8, 17, 90.0),
            (8, 18, 88.0),
            (8, 23, 73.0),
            (8, 27, 57.0),
            (8, 28, 89.0),
            (9, 2, 6.0),
            (9, 6, 22.0),
            (9, 25, 1.0),
            (9, 26, 53.0),
            (10, 6, 63.0),
            (10, 19, 18.0),
            (10, 27, 41.0),
            (11, 0, 17.0),
            (11, 3, 8.0),
            (11, 21, 49.0),
            (11, 22, 48.0),
            (11, 27, 28.0),
            (12, 8, 11.0),
            (13, 8, 15.0),
            (14, 0, 23.0),
            (14, 5, 44.0),
            (14, 13, 42.0),
            (15, 4, 69.0),
            (15, 28, 39.0),
            (16, 8, 60.0),
            (16, 20, 76.0),
            (16, 22, 86.0),
            (16, 27, 87.0),
            (17, 4, 24.0),
            (17, 13, 35.0),
            (17, 25, 21.0),
            (17, 27, 65.0),
            (18, 3, 84.0),
            (18, 10, 62.0),
            (18, 12, 70.0),
            (18, 14, 40.0),
            (18, 22, 7.0),
            (18, 27, 30.0),
            (19, 1, 20.0),
            (19, 25, 81.0),
            (19, 26, 34.0),
            (19, 29, 32.0),
            (20, 14, 33.0),
            (21, 4, 14.0),
            (21, 16, 31.0),
            (21, 19, 77.0),
            (21, 28, 52.0),
            (22, 16, 43.0),
            (22, 18, 75.0),
            (22, 20, 78.0),
            (22, 23, 74.0),
            (23, 2, 4.0),
            (23, 3, 27.0),
            (23, 4, 37.0),
            (23, 5, 56.0),
            (23, 21, 51.0),
            (23, 24, 10.0),
            (24, 20, 50.0),
            (26, 13, 38.0),
            (26, 28, 85.0),
            (27, 5, 45.0),
            (28, 8, 26.0),
            (28, 12, 9.0),
            (28, 15, 82.0),
            (28, 23, 19.0),
            (28, 27, 29.0),
            (29, 2, 5.0),
            (29, 5, 83.0),
        ],
        cut: &[
            (0, 8),
            (4, 6),
            (4, 17),
            (4, 22),
            (6, 10),
            (11, 3),
            (12, 8),
            (13, 8),
            (14, 0),
            (19, 1),
            (19, 26),
            (19, 29),
            (21, 16),
            (22, 16),
            (22, 18),
            (23, 2),
            (23, 3),
            (23, 21),
            (28, 8),
            (28, 15),
            (28, 23),
            (29, 2),
        ],
        feedback_weight: 772.0,
        total_weight: 4095.0,
    },
];

#[test]
fn test_parity_random_instances_match_reference() {
    for case in RANDOM_PARITY_CASES {
        let result = build(case.nodes, case.edges.to_vec()).eades_lin_smyth().unwrap();
        assert_eq!(result.feedback_edges(), case.cut, "cut mismatch for n={}", case.nodes);
        assert!(
            (result.feedback_weight() - case.feedback_weight).abs() < TOLERANCE,
            "feedback weight mismatch for n={}",
            case.nodes
        );
        assert!(
            (result.total_weight() - case.total_weight).abs() < TOLERANCE,
            "total weight mismatch for n={}",
            case.nodes
        );
    }
}

#[test]
fn test_rejects_non_positive_weight() {
    let matrix = build(2, vec![(0, 1, -1.0)]);

    assert!(matches!(matrix.eades_lin_smyth(), Err(EadesLinSmythError::NonPositiveWeight { .. })));
}

#[test]
fn test_rejects_non_finite_weight() {
    let matrix = build(2, vec![(0, 1, f64::NAN)]);

    assert!(matches!(matrix.eades_lin_smyth(), Err(EadesLinSmythError::NonFiniteWeight { .. })));
}

#[test]
fn test_larger_dag_is_exactly_acyclic() {
    // A diamond with an extra cross edge: still acyclic, so GR must drain it
    // entirely by sink-peeling and report zero feedback. This guards that
    // `is_acyclic` is exact (not merely heuristic) on a non-trivial DAG.
    let result = arrange(
        5,
        vec![(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (1, 2, 1.0), (3, 4, 1.0)],
    );

    assert!(result.is_acyclic());
    assert!(result.feedback_edges().is_empty());
    let positions = result.positions();
    assert!(positions[0] < positions[1]);
    assert!(positions[1] < positions[2]);
    assert!(positions[2] < positions[3]);
    assert!(positions[3] < positions[4]);
}
