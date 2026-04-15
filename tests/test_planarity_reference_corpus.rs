//! Validation of the checked-in planarity reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::traits::{OuterplanarityDetection, PlanarityDetection};
use planarity_fixture::load_fixture_suite;

const REFERENCE_CORPUS_100K_PATH: &str = "planarity_ground_truth_100k.json.gz";
const EXPECTED_100K_CASE_COUNT: usize = 100_000;

#[test]
fn test_planarity_reference_corpus_100k() {
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
        "planarity_and_outerplanarity_booleans",
        &[
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "k4_subdivision",
            "k23_subdivision",
            "k33_subdivision",
            "k5_subdivision",
        ],
    );

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
                "planarity mismatched reference case {} ({})",
                case.name, case.family
            );
            assert_eq!(
                is_outerplanar, case.is_outerplanar,
                "outerplanarity mismatched reference case {} ({})",
                case.name, case.family
            );
            assert!(
                !is_outerplanar || is_planar,
                "outerplanar reference case {} must also be planar",
                case.name
            );

            (
                counts.0 + usize::from(is_planar),
                counts.1 + usize::from(is_outerplanar),
                counts.2 + usize::from(!is_planar),
            )
        });

    assert!(planar_count > 0, "reference corpus should contain planar cases");
    assert!(outerplanar_count > 0, "reference corpus should contain outerplanar cases");
    assert!(nonplanar_count > 0, "reference corpus should contain nonplanar cases");
    eprintln!(
        "[planarity-reference-100k] validated {} cases, planar={}, outerplanar={}, nonplanar={}",
        suite.cases.len(),
        planar_count,
        outerplanar_count,
        nonplanar_count
    );
}
