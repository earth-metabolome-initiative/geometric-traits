//! Validation of the checked-in `K_{2,3}` homeomorph reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/k23_homeomorph_fixture.rs"]
mod k23_homeomorph_fixture;

use geometric_traits::traits::K23HomeomorphDetection;
use k23_homeomorph_fixture::load_fixture_suite;

const REFERENCE_CORPUS_100K_PATH: &str = "k23_homeomorph_ground_truth_100k.json.gz";
const EXPECTED_100K_CASE_COUNT: usize = 100_000;

#[test]
fn test_k23_homeomorph_reference_corpus_100k() {
    let suite = common::assert_reference_corpus_contract(
        REFERENCE_CORPUS_100K_PATH,
        EXPECTED_100K_CASE_COUNT,
        load_fixture_suite,
        |suite| suite.schema_version,
        |suite| suite.graph_kind.as_str(),
        |suite| suite.primary_oracle.as_str(),
        |suite| suite.cases.as_slice(),
        |case| case.family.as_str(),
        "undirected_simple_labeled",
        "k23_homeomorph_boolean",
        &[
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "theta",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ],
    );
    common::assert_reference_corpus_family_sequence(
        &suite.family_sequence,
        &[
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "theta",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ],
    );

    let (positive_count, negative_count) =
        suite.cases.iter().fold((0usize, 0usize), |counts, case| {
            let graph = k23_homeomorph_fixture::build_undigraph(case);
            let has_k23_homeomorph = graph.has_k23_homeomorph().unwrap_or_else(|error| {
                panic!("K23 homeomorph detection failed on {}: {error}", case.name)
            });

            assert_eq!(
                has_k23_homeomorph, case.has_k23_homeomorph,
                "K23 homeomorph mismatched reference case {} ({})",
                case.name, case.family
            );

            (
                counts.0 + usize::from(has_k23_homeomorph),
                counts.1 + usize::from(!has_k23_homeomorph),
            )
        });

    assert!(positive_count > 0, "reference corpus should contain positive cases");
    assert!(negative_count > 0, "reference corpus should contain negative cases");
    eprintln!(
        "[k23-homeomorph-reference-100k] validated {} cases, positive={}, negative={}",
        suite.cases.len(),
        positive_count,
        negative_count
    );
}
