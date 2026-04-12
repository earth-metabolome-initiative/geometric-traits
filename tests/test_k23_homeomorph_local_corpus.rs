//! Validation of the local `K_{2,3}` homeomorph reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/k23_homeomorph_fixture.rs"]
mod k23_homeomorph_fixture;

use std::collections::BTreeSet;

use geometric_traits::traits::K23HomeomorphDetection;
use k23_homeomorph_fixture::load_fixture_suite;

const LOCAL_CORPUS_100K_PATH: &str = "k23_homeomorph_ground_truth_100k.json.gz";
const EXPECTED_100K_CASE_COUNT: usize = 100_000;

fn assert_local_corpus_contract(
    relative_path: &str,
    expected_case_count: usize,
) -> k23_homeomorph_fixture::K23HomeomorphFixtureSuite {
    let path = common::fixture_path(relative_path);
    assert!(path.exists(), "local corpus fixture missing at {}; generate it first", path.display());

    let suite = load_fixture_suite(relative_path);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.graph_kind, "undirected_simple_labeled");
    assert_eq!(suite.primary_oracle, "k23_homeomorph_boolean");
    assert_eq!(suite.cases.len(), expected_case_count);

    let observed_families: BTreeSet<&str> =
        suite.cases.iter().map(|case| case.family.as_str()).collect();
    for expected_family in [
        "erdos_renyi",
        "random_tree",
        "outerplanar_cycle_chords",
        "wheel",
        "theta",
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
            "theta",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ]
    );

    suite
}

#[test]
fn test_local_k23_homeomorph_reference_corpus_100k() {
    let suite = assert_local_corpus_contract(LOCAL_CORPUS_100K_PATH, EXPECTED_100K_CASE_COUNT);

    let (positive_count, negative_count) =
        suite.cases.iter().fold((0usize, 0usize), |counts, case| {
            let graph = k23_homeomorph_fixture::build_undigraph(case);
            let has_k23_homeomorph = graph.has_k23_homeomorph().unwrap_or_else(|error| {
                panic!("K23 homeomorph detection failed on {}: {error}", case.name)
            });

            assert_eq!(
                has_k23_homeomorph, case.has_k23_homeomorph,
                "K23 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );

            (
                counts.0 + usize::from(has_k23_homeomorph),
                counts.1 + usize::from(!has_k23_homeomorph),
            )
        });

    assert!(positive_count > 0, "local corpus should contain positive cases");
    assert!(negative_count > 0, "local corpus should contain negative cases");
    eprintln!(
        "[k23-homeomorph-local-100k] validated {} cases, positive={}, negative={}",
        suite.cases.len(),
        positive_count,
        negative_count
    );
}
