//! Validation of the local planarity reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use std::collections::BTreeSet;

use geometric_traits::traits::{OuterplanarityDetection, PlanarityDetection};
use planarity_fixture::load_fixture_suite;

const LOCAL_CORPUS_100K_PATH: &str = "planarity_ground_truth_100k.json.gz";
const EXPECTED_100K_CASE_COUNT: usize = 100_000;

fn assert_local_corpus_contract(
    relative_path: &str,
    expected_case_count: usize,
) -> planarity_fixture::PlanarityFixtureSuite {
    let path = common::fixture_path(relative_path);
    assert!(path.exists(), "local corpus fixture missing at {}; generate it first", path.display());

    let suite = load_fixture_suite(relative_path);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.graph_kind, "undirected_simple_labeled");
    assert_eq!(suite.primary_oracle, "planarity_and_outerplanarity_booleans");
    assert_eq!(suite.cases.len(), expected_case_count);

    let observed_families: BTreeSet<&str> =
        suite.cases.iter().map(|case| case.family.as_str()).collect();
    for expected_family in [
        "erdos_renyi",
        "random_tree",
        "outerplanar_cycle_chords",
        "wheel",
        "k4_subdivision",
        "k23_subdivision",
        "k33_subdivision",
        "k5_subdivision",
    ] {
        assert!(
            observed_families.contains(expected_family),
            "local corpus must contain at least one {expected_family} case"
        );
    }

    suite
}

#[test]
fn test_local_planarity_reference_corpus_100k() {
    let suite = assert_local_corpus_contract(LOCAL_CORPUS_100K_PATH, EXPECTED_100K_CASE_COUNT);

    let (planar_count, outerplanar_count, nonplanar_count) =
        suite.cases.iter().fold((0usize, 0usize, 0usize), |counts, case| {
            let graph = planarity_fixture::build_undigraph(case);
            let is_planar = graph
                .is_planar()
                .unwrap_or_else(|error| panic!("planarity failed on {}: {error}", case.name));
            let is_outerplanar = graph
                .is_outerplanar()
                .unwrap_or_else(|error| panic!("outerplanarity failed on {}: {error}", case.name));

            assert_eq!(
                is_planar, case.is_planar,
                "planarity mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert_eq!(
                is_outerplanar, case.is_outerplanar,
                "outerplanarity mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert!(
                !is_outerplanar || is_planar,
                "outerplanar local reference case {} must also be planar",
                case.name
            );

            (
                counts.0 + usize::from(is_planar),
                counts.1 + usize::from(is_outerplanar),
                counts.2 + usize::from(!is_planar),
            )
        });

    assert!(planar_count > 0, "local corpus should contain planar cases");
    assert!(outerplanar_count > 0, "local corpus should contain outerplanar cases");
    assert!(nonplanar_count > 0, "local corpus should contain nonplanar cases");
    eprintln!(
        "[planarity-local-100k] validated {} cases, planar={}, outerplanar={}, nonplanar={}",
        suite.cases.len(),
        planar_count,
        outerplanar_count,
        nonplanar_count
    );
}
