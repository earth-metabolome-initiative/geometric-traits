//! Validation of the checked-in `K_4` homeomorph reference oracle in the
//! combined v4 corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/topological_validity_fixture.rs"]
mod topological_validity_fixture;

use std::sync::atomic::{AtomicUsize, Ordering};

use geometric_traits::traits::K4HomeomorphDetection;
use rayon::prelude::*;
use topological_validity_fixture::load_fixture_suite;

const REFERENCE_CORPUS_1M_PATH: &str = "topological_validity_ground_truth_1m_v4.json.gz";
const EXPECTED_1M_CASE_COUNT: usize = 1_000_000;

#[test]
#[ignore = "expensive checked-in reference corpus; run manually when needed"]
fn test_k4_homeomorph_reference_corpus_1m() {
    let suite = common::assert_reference_corpus_contract(
        REFERENCE_CORPUS_1M_PATH,
        EXPECTED_1M_CASE_COUNT,
        load_fixture_suite,
        |suite| suite.schema_version,
        |suite| suite.graph_kind.as_str(),
        |suite| suite.primary_oracle.as_str(),
        |suite| suite.cases.as_slice(),
        |case| case.family.as_str(),
        "undirected_simple_labeled",
        "planarity_outerplanarity_k23_k33_k4_booleans",
        &[
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "theta",
            "clique",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ],
    );
    assert_eq!(suite.count, EXPECTED_1M_CASE_COUNT);
    common::assert_reference_corpus_family_sequence(
        &suite.family_sequence,
        &[
            "erdos_renyi",
            "random_tree",
            "outerplanar_cycle_chords",
            "wheel",
            "theta",
            "clique",
            "k23_subdivision",
            "k33_subdivision",
            "k4_subdivision",
            "k5_subdivision",
        ],
    );

    eprintln!("[k4-homeomorph-reference-1m] progress 0");
    let completed = AtomicUsize::new(0);

    let (positive_count, negative_count) = suite
        .cases
        .par_iter()
        .map(|case| {
            let graph = topological_validity_fixture::build_undigraph(case);
            let has_k4_homeomorph = graph.has_k4_homeomorph().unwrap_or_else(|error| {
                panic!("K4 homeomorph detection failed on {}: {error}", case.name)
            });

            assert_eq!(
                has_k4_homeomorph, case.has_k4_homeomorph,
                "K4 homeomorph mismatched reference case {} ({}) nodes={} edges={:?}",
                case.name, case.family, case.node_count, case.edges
            );

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10_000 == 0 {
                eprintln!("[k4-homeomorph-reference-1m] progress {done}");
            }

            (usize::from(has_k4_homeomorph), usize::from(!has_k4_homeomorph))
        })
        .reduce(|| (0usize, 0usize), |left, right| (left.0 + right.0, left.1 + right.1));

    assert!(positive_count > 0, "reference corpus should contain positive cases");
    assert!(negative_count > 0, "reference corpus should contain negative cases");
    eprintln!(
        "[k4-homeomorph-reference-1m] validated {} cases, positive={}, negative={}",
        suite.cases.len(),
        positive_count,
        negative_count
    );
}
