//! Validation of the local `K_4` homeomorph reference oracle in the combined
//! v4 corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/topological_validity_fixture.rs"]
mod topological_validity_fixture;

use std::sync::atomic::{AtomicUsize, Ordering};

use geometric_traits::traits::K4HomeomorphDetection;
use rayon::prelude::*;
use topological_validity_fixture::load_fixture_suite;

const LOCAL_CORPUS_1M_PATH: &str = "topological_validity_ground_truth_1m_v4.json.gz";
const EXPECTED_1M_CASE_COUNT: usize = 1_000_000;

#[test]
#[ignore = "expensive local oracle corpus; run manually when needed"]
fn test_local_k4_homeomorph_reference_corpus_1m() {
    let path = common::fixture_path(LOCAL_CORPUS_1M_PATH);
    assert!(path.exists(), "local corpus fixture missing at {}; generate it first", path.display());

    let suite = load_fixture_suite(LOCAL_CORPUS_1M_PATH);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.graph_kind, "undirected_simple_labeled");
    assert_eq!(suite.primary_oracle, "planarity_outerplanarity_k23_k33_k4_booleans");
    assert_eq!(suite.count, EXPECTED_1M_CASE_COUNT);
    assert_eq!(suite.cases.len(), EXPECTED_1M_CASE_COUNT);

    eprintln!("[k4-homeomorph-local-1m] progress 0");
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
                "K4 homeomorph mismatched local reference case {} ({}) nodes={} edges={:?}",
                case.name, case.family, case.node_count, case.edges
            );

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10_000 == 0 {
                eprintln!("[k4-homeomorph-local-1m] progress {done}");
            }

            (usize::from(has_k4_homeomorph), usize::from(!has_k4_homeomorph))
        })
        .reduce(|| (0usize, 0usize), |left, right| (left.0 + right.0, left.1 + right.1));

    assert!(positive_count > 0, "local corpus should contain positive cases");
    assert!(negative_count > 0, "local corpus should contain negative cases");
    eprintln!(
        "[k4-homeomorph-local-1m] validated {} cases, positive={}, negative={}",
        suite.cases.len(),
        positive_count,
        negative_count
    );
}
