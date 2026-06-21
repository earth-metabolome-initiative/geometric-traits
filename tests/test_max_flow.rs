//! Tests for the maximum s-t flow algorithms `Dinic` and `EdmondsKarp`.
//!
//! Both traits compute the same function (maximum flow value, a feasible flow,
//! and a minimum cut) on a directed capacity graph, so every case is run
//! through both and the two are required to agree. The unit cases pin
//! hand-computed values and the structural guarantees (per-arc feasibility,
//! conservation, and a saturated minimum cut equal to the flow). The fixture
//! cases cross-check the value against the NetworkX ground-truth corpus, and
//! the bipartite cases cross-check against the already-implemented
//! Hopcroft-Karp matcher.
#![cfg(feature = "std")]

#[path = "support/fixture_io.rs"]
mod fixture_io;

use std::collections::{BTreeMap, BTreeSet};

use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    prelude::*,
};
use serde::Deserialize;

type CapacityMatrix = ValuedCSR2D<usize, usize, usize, u64>;

const FIXTURE_NAME: &str = "max_flow_ground_truth.json.gz";

/// Builds a square capacity matrix from `(source, destination, capacity)`
/// triples (self-loops permitted, deduplicated, sorted for the CSR builder).
fn build(n: usize, mut edges: Vec<(usize, usize, u64)>) -> CapacityMatrix {
    edges.sort_unstable();
    edges.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    GenericEdgesBuilder::<_, CapacityMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

/// Asserts every structural guarantee of a single max-flow result.
fn validate(
    label: &str,
    n: usize,
    edges: &[(usize, usize, u64)],
    source: usize,
    sink: usize,
    result: &MaxFlowResult<u64>,
) {
    let value = result.max_flow();

    let mut capacity_of: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    for &(u, v, c) in edges {
        // Self-loops and zero-capacity arcs are dropped by the algorithms, so
        // they never appear in the flow or the minimum cut.
        if u != v && c != 0 {
            capacity_of.insert((u, v), c);
        }
    }

    let mut flow_of: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut net: Vec<i128> = vec![0; n];
    for &(u, v, f) in result.flows() {
        assert!(f > 0, "{label}: reported flow on ({u}, {v}) must be positive");
        let capacity = capacity_of
            .get(&(u, v))
            .copied()
            .unwrap_or_else(|| panic!("{label}: flow on non-existent arc ({u}, {v})"));
        assert!(f <= capacity, "{label}: flow {f} exceeds capacity {capacity} on ({u}, {v})");
        assert!(flow_of.insert((u, v), f).is_none(), "{label}: duplicate flow arc ({u}, {v})");
        net[u] += i128::from(f);
        net[v] -= i128::from(f);
    }

    for (node, balance) in net.iter().enumerate() {
        if node != source && node != sink {
            assert_eq!(*balance, 0, "{label}: flow not conserved at node {node}");
        }
    }
    assert_eq!(
        net[source],
        i128::from(value),
        "{label}: net flow out of source must equal max_flow"
    );
    assert_eq!(net[sink], -i128::from(value), "{label}: net flow into sink must equal max_flow");

    let side = result.source_side();
    assert_eq!(side.len(), n, "{label}: source_side must have one flag per node");
    assert!(side[source], "{label}: source must be on the source side");
    assert!(!side[sink], "{label}: sink must be off the source side after a max flow");
    let cut: BTreeSet<(usize, usize)> = result.min_cut().iter().copied().collect();
    assert_eq!(cut.len(), result.min_cut().len(), "{label}: min_cut must not repeat an arc");
    let mut cut_capacity: u64 = 0;
    for &(u, v) in result.min_cut() {
        assert!(
            side[u] && !side[v],
            "{label}: min-cut arc ({u}, {v}) does not cross the partition"
        );
        cut_capacity += capacity_of[&(u, v)];
    }
    assert_eq!(cut_capacity, value, "{label}: min-cut capacity must equal max_flow");
    for (&(u, v), &capacity) in &capacity_of {
        if side[u] && !side[v] {
            assert!(cut.contains(&(u, v)), "{label}: crossing arc ({u}, {v}) missing from min_cut");
            assert_eq!(
                flow_of.get(&(u, v)).copied().unwrap_or(0),
                capacity,
                "{label}: crossing arc ({u}, {v}) must be saturated"
            );
        }
    }
}

/// Runs a case through both algorithms, validates each, asserts they agree on
/// the value, and returns the common maximum-flow value.
fn max_flow(n: usize, edges: &[(usize, usize, u64)], source: usize, sink: usize) -> u64 {
    let matrix = build(n, edges.to_vec());
    let dinic = matrix.dinic(source, sink).unwrap();
    let edmonds_karp = matrix.edmonds_karp(source, sink).unwrap();
    validate("dinic", n, edges, source, sink, &dinic);
    validate("edmonds_karp", n, edges, source, sink, &edmonds_karp);
    assert_eq!(
        dinic.max_flow(),
        edmonds_karp.max_flow(),
        "Dinic and Edmonds-Karp disagree: {} vs {}",
        dinic.max_flow(),
        edmonds_karp.max_flow(),
    );
    dinic.max_flow()
}

#[test]
fn test_single_arc() {
    assert_eq!(max_flow(2, &[(0, 1, 5)], 0, 1), 5);
}

#[test]
fn test_path_is_limited_by_the_bottleneck() {
    // 0 -> 1 -> 2 -> 3 with capacities 3, 2, 5: the middle arc caps the flow.
    assert_eq!(max_flow(4, &[(0, 1, 3), (1, 2, 2), (2, 3, 5)], 0, 3), 2);
}

#[test]
fn test_diamond_sums_two_disjoint_paths() {
    // Two internal paths, each bottlenecked to 2 units, sum to 4.
    assert_eq!(max_flow(4, &[(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)], 0, 3), 4);
}

#[test]
fn test_antiparallel_arcs_stay_independent() {
    // The reverse arc 1 -> 0 must not cancel the forward arc 0 -> 1: the flow is
    // limited only by the 1 -> 2 arc.
    assert_eq!(max_flow(3, &[(0, 1, 3), (1, 0, 5), (1, 2, 2)], 0, 2), 2);
}

#[test]
fn test_clrs_textbook_network() {
    // The canonical Cormen et al. network whose maximum flow is 23. Greedy
    // augmentation without residual (reverse) arcs cannot reach this value, so
    // it exercises the cancellation that residual arcs provide.
    let edges = [
        (0, 1, 16),
        (0, 2, 13),
        (1, 2, 10),
        (2, 1, 4),
        (1, 3, 12),
        (3, 2, 9),
        (2, 4, 14),
        (4, 3, 7),
        (3, 5, 20),
        (4, 5, 4),
    ];
    assert_eq!(max_flow(6, &edges, 0, 5), 23);
}

#[test]
fn test_self_loops_are_ignored() {
    // Self-loops on the source, an interior node, and the sink must not change
    // the flow, which is bounded by the 1 -> 2 arc.
    let edges = [(0, 0, 9), (0, 1, 4), (1, 1, 7), (1, 2, 4), (2, 3, 4), (3, 3, 9)];
    assert_eq!(max_flow(4, &edges, 0, 3), 4);
}

#[test]
fn test_disconnected_source_and_sink_have_zero_flow() {
    let edges = [(0, 1, 7), (3, 4, 7)];
    assert_eq!(max_flow(5, &edges, 0, 4), 0);
    let matrix = build(5, edges.to_vec());
    let result = matrix.dinic(0, 4).unwrap();
    assert!(result.flows().is_empty());
    assert!(result.min_cut().is_empty());
}

#[test]
fn test_min_cut_is_reported() {
    // The min cut of the diamond is the pair of arcs leaving the source side,
    // and it is uniquely determined here, so both algorithms must report it.
    let matrix = build(4, vec![(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)]);
    let expected = BTreeSet::from([(0, 2), (1, 3)]);
    for (label, cut) in [
        ("dinic", matrix.dinic(0, 3).unwrap()),
        ("edmonds_karp", matrix.edmonds_karp(0, 3).unwrap()),
    ] {
        assert_eq!(cut.max_flow(), 4, "{label}: value");
        let actual: BTreeSet<(usize, usize)> = cut.min_cut().iter().copied().collect();
        assert_eq!(actual, expected, "{label}: min cut");
    }
}

#[test]
fn test_is_deterministic() {
    let edges = vec![(0, 1, 4), (0, 2, 4), (1, 3, 3), (2, 3, 3), (1, 2, 2)];
    let matrix = build(4, edges);
    assert_eq!(matrix.dinic(0, 3).unwrap().flows(), matrix.dinic(0, 3).unwrap().flows());
    assert_eq!(
        matrix.edmonds_karp(0, 3).unwrap().flows(),
        matrix.edmonds_karp(0, 3).unwrap().flows()
    );
}

#[test]
fn test_rejects_source_out_of_range() {
    let matrix = build(2, vec![(0, 1, 5)]);
    assert!(matches!(
        matrix.dinic(5, 1),
        Err(MaxFlowError::SourceOutOfRange { source_id: 5, order: 2 })
    ));
    assert!(matches!(
        matrix.edmonds_karp(5, 1),
        Err(MaxFlowError::SourceOutOfRange { source_id: 5, order: 2 })
    ));
}

#[test]
fn test_rejects_sink_out_of_range() {
    let matrix = build(2, vec![(0, 1, 5)]);
    assert!(matches!(matrix.dinic(0, 9), Err(MaxFlowError::SinkOutOfRange { sink: 9, order: 2 })));
    assert!(matches!(
        matrix.edmonds_karp(0, 9),
        Err(MaxFlowError::SinkOutOfRange { sink: 9, order: 2 })
    ));
}

#[test]
fn test_rejects_source_equal_to_sink() {
    let matrix = build(2, vec![(0, 1, 5)]);
    assert!(matches!(matrix.dinic(1, 1), Err(MaxFlowError::SourceEqualsSink { node: 1 })));
    assert!(matches!(matrix.edmonds_karp(1, 1), Err(MaxFlowError::SourceEqualsSink { node: 1 })));
}

#[test]
fn test_rejects_negative_capacity() {
    let matrix: ValuedCSR2D<usize, usize, usize, i64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, i64>>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 2))
            .edges(vec![(0usize, 1usize, -3i64)].into_iter())
            .build()
            .unwrap();
    assert!(matches!(matrix.dinic(0, 1), Err(MaxFlowError::NegativeCapacity { .. })));
    assert!(matches!(matrix.edmonds_karp(0, 1), Err(MaxFlowError::NegativeCapacity { .. })));
}

#[test]
fn test_rejects_non_finite_capacity() {
    let matrix: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 2))
            .edges(vec![(0usize, 1usize, f64::INFINITY)].into_iter())
            .build()
            .unwrap();
    assert!(matches!(matrix.dinic(0, 1), Err(MaxFlowError::NonFiniteCapacity { .. })));
    assert!(matches!(matrix.edmonds_karp(0, 1), Err(MaxFlowError::NonFiniteCapacity { .. })));
}

#[test]
fn test_float_capacities_are_supported() {
    // Both algorithms are generic over the capacity type.
    let matrix: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(4)
            .expected_shape((4, 4))
            .edges(vec![(0, 1, 1.5), (0, 2, 2.5), (1, 3, 1.0), (2, 3, 4.0)].into_iter())
            .build()
            .unwrap();
    // 0 -> 1 -> 3 carries 1.0, 0 -> 2 -> 3 carries 2.5, total 3.5.
    let result = matrix.dinic(0, 3).unwrap();
    assert!((result.max_flow() - 3.5).abs() < 1e-12);
    assert!(!result.flows().is_empty());
    assert!(!result.min_cut().is_empty());
    assert_eq!(result.source_side().len(), 4);
    assert!((matrix.edmonds_karp(0, 3).unwrap().max_flow() - 3.5).abs() < 1e-12);
}

#[test]
fn test_signed_integer_capacities() {
    // The algorithms are generic over the capacity type, so signed integer
    // capacities work end to end and produce a full result, not just a value.
    let matrix: ValuedCSR2D<usize, usize, usize, i64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, i64>>::default()
            .expected_number_of_edges(4)
            .expected_shape((4, 4))
            .edges(vec![(0, 1, 3i64), (0, 2, 2), (1, 3, 2), (2, 3, 3)].into_iter())
            .build()
            .unwrap();
    let result = matrix.dinic(0, 3).unwrap();
    assert_eq!(result.max_flow(), 4);
    assert_eq!(result.source_side().len(), 4);
    assert!(!result.flows().is_empty());
    assert!(!result.min_cut().is_empty());
    assert_eq!(matrix.edmonds_karp(0, 3).unwrap().max_flow(), 4);
}

#[test]
fn test_zero_capacity_arcs_are_ignored() {
    // An explicit zero-capacity arc 0 -> 1 carries nothing and is skipped, so
    // the only s-t path is 0 -> 2 -> 1 with capacity 5.
    let edges = [(0, 1, 0), (0, 2, 5), (2, 1, 5)];
    assert_eq!(max_flow(3, &edges, 0, 1), 5);
    let result = build(3, edges.to_vec()).dinic(0, 1).unwrap();
    assert!(
        result.flows().iter().all(|&(u, v, _)| (u, v) != (0, 1)),
        "the zero-capacity arc must carry no flow"
    );
}

#[test]
fn test_error_messages_are_formatted() {
    // Exercises the `Display` of every error variant.
    let matrix = build(2, vec![(0, 1, 5)]);
    assert!(matrix.dinic(5, 1).unwrap_err().to_string().contains("source node 5"));
    assert!(matrix.edmonds_karp(0, 9).unwrap_err().to_string().contains("sink node 9"));
    assert!(matrix.dinic(1, 1).unwrap_err().to_string().contains("must differ"));

    let negative: ValuedCSR2D<usize, usize, usize, i64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, i64>>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 2))
            .edges(vec![(0usize, 1usize, -3i64)].into_iter())
            .build()
            .unwrap();
    assert!(negative.dinic(0, 1).unwrap_err().to_string().contains("negative capacity"));

    let non_finite: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 2))
            .edges(vec![(0usize, 1usize, f64::NAN)].into_iter())
            .build()
            .unwrap();
    assert!(non_finite.edmonds_karp(0, 1).unwrap_err().to_string().contains("non-finite capacity"));
}

// ── Differential against the already-implemented Hopcroft-Karp matcher ──────

fn bipartite_matching_via_flow(left: usize, right: usize, edges: &[(usize, usize)]) -> (u64, u64) {
    let source = 0;
    let sink = left + right + 1;
    let n = left + right + 2;
    let mut flow_edges: Vec<(usize, usize, u64)> = Vec::new();
    for l in 0..left {
        flow_edges.push((source, 1 + l, 1));
    }
    for r in 0..right {
        flow_edges.push((1 + left + r, sink, 1));
    }
    for &(l, r) in edges {
        flow_edges.push((1 + l, 1 + left + r, 1));
    }
    let matrix = build(n, flow_edges);
    (
        matrix.dinic(source, sink).unwrap().max_flow(),
        matrix.edmonds_karp(source, sink).unwrap().max_flow(),
    )
}

fn hopcroft_karp_matching(left: usize, right: usize, edges: &[(usize, usize)]) -> u64 {
    let matrix: CSR2D<usize, usize, usize> = GenericEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((left, right))
        .edges(edges.iter().copied())
        .build()
        .unwrap();
    u64::try_from(matrix.hopcroft_karp().unwrap().len()).unwrap()
}

#[test]
fn test_matches_hopcroft_karp_perfect_matching() {
    let edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)];
    let expected = hopcroft_karp_matching(3, 3, &edges);
    assert_eq!(expected, 3);
    assert_eq!(bipartite_matching_via_flow(3, 3, &edges), (expected, expected));
}

#[test]
fn test_matches_hopcroft_karp_deficient_matching() {
    // Left 0 and 1 compete for right 0, left 2 and 3 compete for right 1, and
    // right 2 is isolated, so the maximum matching is only 2 (a Hall deficiency).
    let edges = [(0, 0), (1, 0), (2, 1), (3, 1)];
    let expected = hopcroft_karp_matching(4, 3, &edges);
    assert_eq!(expected, 2);
    assert_eq!(bipartite_matching_via_flow(4, 3, &edges), (expected, expected));
}

// ── NetworkX max-flow ground-truth corpus ───────────────────────────────────

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    networkx_version: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    node_count: usize,
    edges: Vec<(usize, usize, u64)>,
    source: usize,
    sink: usize,
    max_flow: u64,
    reference: Reference,
}

#[derive(Debug, Deserialize)]
struct Reference {
    dinitz_nanos: u64,
    edmonds_karp_nanos: u64,
}

fn fixture() -> Fixture {
    load_fixture_json(FIXTURE_NAME)
}

#[test]
fn test_fixture_metadata() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 2);
    assert!(!fixture.networkx_version.is_empty());
    assert!(fixture.cases.len() >= 10, "expected a non-trivial suite");
    for case in &fixture.cases {
        assert!(case.node_count >= 2, "case `{}` must have at least two nodes", case.id);
        assert!(case.source < case.node_count, "case `{}` source out of range", case.id);
        assert!(case.sink < case.node_count, "case `{}` sink out of range", case.id);
        assert!(case.reference.dinitz_nanos > 0, "case `{}` must record dinitz timing", case.id);
        assert!(
            case.reference.edmonds_karp_nanos > 0,
            "case `{}` must record edmonds_karp timing",
            case.id
        );
    }
}

#[test]
fn test_max_flow_matches_networkx() {
    let fixture = fixture();
    for case in &fixture.cases {
        let value = max_flow(case.node_count, &case.edges, case.source, case.sink);
        assert_eq!(
            value, case.max_flow,
            "case `{}`: ours={value}, networkx={}",
            case.id, case.max_flow
        );
    }
}
