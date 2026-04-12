//! Validation of the local `K_{3,3}` homeomorph reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/k33_homeomorph_fixture.rs"]
mod k33_homeomorph_fixture;

use std::collections::BTreeSet;

use geometric_traits::traits::K33HomeomorphDetection;
use k33_homeomorph_fixture::load_fixture_suite;

const LOCAL_CORPUS_100K_PATH: &str = "k33_homeomorph_ground_truth_100k.json.gz";
const EXPECTED_100K_CASE_COUNT: usize = 100_000;

fn assert_local_corpus_contract(
    relative_path: &str,
    expected_case_count: usize,
) -> k33_homeomorph_fixture::K33HomeomorphFixtureSuite {
    let path = common::fixture_path(relative_path);
    assert!(path.exists(), "local corpus fixture missing at {}; generate it first", path.display());

    let suite = load_fixture_suite(relative_path);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.graph_kind, "undirected_simple_labeled");
    assert_eq!(suite.primary_oracle, "k33_homeomorph_boolean");
    assert_eq!(suite.cases.len(), expected_case_count);

    let observed_families: BTreeSet<&str> =
        suite.cases.iter().map(|case| case.family.as_str()).collect();
    for expected_family in [
        "erdos_renyi",
        "random_tree",
        "outerplanar_cycle_chords",
        "wheel",
        "clique",
        "k23_subdivision",
        "k33_subdivision",
        "k4_subdivision",
        "k5_subdivision",
    ] {
        assert!(
            observed_families.contains(expected_family),
            "local corpus must contain at least one {expected_family} case"
        );
    }

    assert_eq!(
        suite.family_sequence,
        [
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "clique",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ]
    );

    suite
}

#[test]
#[ignore = "expensive local oracle corpus; run manually when needed"]
fn test_local_k33_homeomorph_reference_corpus_100k() {
    let suite = assert_local_corpus_contract(LOCAL_CORPUS_100K_PATH, EXPECTED_100K_CASE_COUNT);

    let (positive_count, negative_count) =
        suite.cases.iter().enumerate().fold((0usize, 0usize), |counts, (index, case)| {
            if index % 1_000 == 0 {
                eprintln!("[k33-homeomorph-local-100k] progress {index}");
            }
            let graph = k33_homeomorph_fixture::build_undigraph(case);
            let has_k33_homeomorph = graph.has_k33_homeomorph().unwrap_or_else(|error| {
                panic!("K33 homeomorph detection failed on {}: {error}", case.name)
            });

            assert_eq!(
                has_k33_homeomorph, case.has_k33_homeomorph,
                "K33 homeomorph mismatched local reference case {} ({}) nodes={} edges={:?}",
                case.name, case.family, case.node_count, case.edges
            );

            (
                counts.0 + usize::from(has_k33_homeomorph),
                counts.1 + usize::from(!has_k33_homeomorph),
            )
        });

    assert!(positive_count > 0, "local corpus should contain positive cases");
    assert!(negative_count > 0, "local corpus should contain negative cases");
    eprintln!(
        "[k33-homeomorph-local-100k] validated {} cases, positive={}, negative={}",
        suite.cases.len(),
        positive_count,
        negative_count
    );
}
