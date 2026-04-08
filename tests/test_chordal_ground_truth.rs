//! Fixture-driven integration tests for chordal detection.
#![cfg(feature = "std")]

#[path = "support/chordal_fixture.rs"]
mod chordal_fixture;

use chordal_fixture::{build_undigraph, load_fixture_suite, normalize_edge};
use geometric_traits::traits::ChordalDetection;

const FIXTURE_NAME: &str = "chordal_ground_truth.json.gz";

fn assert_is_permutation(ordering: &[usize], node_count: usize, context: &str) {
    assert_eq!(
        ordering.len(),
        node_count,
        "ordering for `{context}` must include exactly one position per node"
    );
    let mut seen = vec![false; node_count];
    for &node in ordering {
        assert!(node < node_count, "ordering for `{context}` contains out-of-range node {node}");
        assert!(!seen[node], "ordering for `{context}` contains duplicate node {node}");
        seen[node] = true;
    }
}

#[test]
fn test_chordal_fixture_suite_header() {
    let suite = load_fixture_suite(FIXTURE_NAME);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.algorithm, "chordal_detection");
    assert_eq!(suite.graph_kind, "undirected_simple");
    assert_eq!(suite.generator, "networkx");
    assert_eq!(suite.networkx_version, "3.3");
    assert_eq!(suite.seed, 42);
    assert!(!suite.python_version.is_empty());
    assert!(suite.cases.len() >= 3000);
    assert!(suite.cases.iter().any(|case| case.is_chordal));
    assert!(suite.cases.iter().any(|case| !case.is_chordal));
}

#[test]
fn test_chordal_ground_truth_cases() {
    let suite = load_fixture_suite(FIXTURE_NAME);

    for case in suite.cases {
        let mut normalized_edges = case.edges.clone();
        for edge in &mut normalized_edges {
            *edge = normalize_edge(*edge);
        }
        normalized_edges.sort_unstable();
        normalized_edges.dedup();
        assert_eq!(
            case.edges, normalized_edges,
            "fixture case `{}` must store canonical simple edges",
            case.name
        );

        for edge in &case.edges {
            assert!(
                edge[0] < case.node_count && edge[1] < case.node_count,
                "fixture case `{}` contains an out-of-range edge {edge:?}",
                case.name
            );
            assert_ne!(
                edge[0], edge[1],
                "fixture case `{}` should not contain self-loops",
                case.name
            );
        }

        let graph = build_undigraph(&case);
        assert_eq!(
            graph.is_chordal(),
            case.is_chordal,
            "chordality mismatch for `{}` ({})",
            case.name,
            case.family
        );

        let mcs_ordering = graph.maximum_cardinality_search_ordering();
        assert_is_permutation(&mcs_ordering, case.node_count, &case.name);
        assert_eq!(
            graph.is_perfect_elimination_ordering(&mcs_ordering),
            case.is_chordal,
            "MCS ordering validity mismatch for `{}` ({})",
            case.name,
            case.family
        );

        let peo = graph.perfect_elimination_ordering();
        assert_eq!(
            peo.is_some(),
            case.is_chordal,
            "PEO existence mismatch for `{}` ({})",
            case.name,
            case.family
        );
        if let Some(ordering) = peo {
            assert_is_permutation(&ordering, case.node_count, &case.name);
            assert!(
                graph.is_perfect_elimination_ordering(&ordering),
                "reported PEO should verify for `{}` ({})",
                case.name,
                case.family
            );
        }
    }
}
