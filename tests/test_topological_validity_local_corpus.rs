//! Validation of the combined local topological-validity reference corpus.
#![cfg(feature = "std")]

mod common;

#[path = "support/topological_validity_fixture.rs"]
mod topological_validity_fixture;

use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use geometric_traits::traits::{
    K4HomeomorphDetection, K23HomeomorphDetection, K33HomeomorphDetection, OuterplanarityDetection,
    PlanarityDetection,
};
use rayon::prelude::*;
use topological_validity_fixture::load_fixture_suite;

const LOCAL_CORPUS_1M_PATH: &str = "topological_validity_ground_truth_1m_v4.json.gz";
const EXPECTED_1M_CASE_COUNT: usize = 1_000_000;

fn assert_local_corpus_contract(
    relative_path: &str,
    expected_case_count: usize,
) -> topological_validity_fixture::TopologicalValidityFixtureSuite {
    let path = common::fixture_path(relative_path);
    assert!(path.exists(), "local corpus fixture missing at {}; generate it first", path.display());

    let suite = load_fixture_suite(relative_path);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.graph_kind, "undirected_simple_labeled");
    assert_eq!(suite.primary_oracle, "planarity_outerplanarity_k23_k33_k4_booleans");
    assert_eq!(suite.count, expected_case_count);
    assert_eq!(suite.cases.len(), expected_case_count);

    let observed_families: BTreeSet<&str> =
        suite.cases.iter().map(|case| case.family.as_str()).collect();
    for expected_family in [
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
#[ignore = "expensive combined local oracle corpus; run manually when needed"]
#[allow(clippy::too_many_lines)]
fn test_local_topological_validity_reference_corpus_1m_without_k4() {
    let suite = assert_local_corpus_contract(LOCAL_CORPUS_1M_PATH, EXPECTED_1M_CASE_COUNT);
    let start_index = std::env::var("TOPOLOGY_START_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let end_index = std::env::var("TOPOLOGY_END_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(suite.cases.len());
    assert!(start_index <= end_index);
    assert!(end_index <= suite.cases.len());
    let cases = &suite.cases[start_index..end_index];

    eprintln!(
        "[topology-local-1m] validating range [{start_index}, {end_index}) count={}",
        cases.len()
    );
    eprintln!("[topology-local-1m] progress 0");
    let completed = AtomicUsize::new(0);
    let slow_threshold_ms =
        std::env::var("TOPOLOGY_TRACE_SLOW_MS").ok().and_then(|value| value.parse::<u128>().ok());

    let (planar_count, outerplanar_count, k23_count, k33_count) = cases
        .par_iter()
        .enumerate()
        .map(|(local_index, case)| {
            let global_index = start_index + local_index;
            let overall_start = Instant::now();
            let graph = topological_validity_fixture::build_undigraph(case);

            let planar_start = Instant::now();
            let is_planar = graph
                .is_planar()
                .unwrap_or_else(|error| panic!("planarity failed on {}: {error}", case.name));
            let planar_elapsed = planar_start.elapsed();

            let outerplanar_start = Instant::now();
            let is_outerplanar = graph
                .is_outerplanar()
                .unwrap_or_else(|error| panic!("outerplanarity failed on {}: {error}", case.name));
            let outerplanar_elapsed = outerplanar_start.elapsed();

            let k23_start = Instant::now();
            let has_k23_homeomorph = graph.has_k23_homeomorph().unwrap_or_else(|error| {
                panic!("K23 homeomorph detection failed on {}: {error}", case.name)
            });
            let k23_elapsed = k23_start.elapsed();

            let k33_start = Instant::now();
            let has_k33_homeomorph = graph.has_k33_homeomorph().unwrap_or_else(|error| {
                panic!("K33 homeomorph detection failed on {}: {error}", case.name)
            });
            let k33_elapsed = k33_start.elapsed();

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
            assert_eq!(
                has_k23_homeomorph, case.has_k23_homeomorph,
                "K23 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert_eq!(
                has_k33_homeomorph, case.has_k33_homeomorph,
                "K33 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert!(
                !is_outerplanar || is_planar,
                "outerplanar local reference case {} must also be planar",
                case.name
            );

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10_000 == 0 {
                eprintln!("[topology-local-1m] progress {done}");
            }

            let overall_elapsed = overall_start.elapsed();
            if let Some(threshold_ms) = slow_threshold_ms {
                if overall_elapsed.as_millis() >= threshold_ms {
                    eprintln!(
                        "[topology-local-1m-slow] index={} case={} family={} total={:?} planar={:?} outerplanar={:?} k23={:?} k33={:?}",
                        global_index,
                        case.name,
                        case.family,
                        overall_elapsed,
                        planar_elapsed,
                        outerplanar_elapsed,
                        k23_elapsed,
                        k33_elapsed
                    );
                }
            }

            (
                usize::from(is_planar),
                usize::from(is_outerplanar),
                usize::from(has_k23_homeomorph),
                usize::from(has_k33_homeomorph),
            )
        })
        .reduce(
            || (0usize, 0usize, 0usize, 0usize),
            |left, right| (left.0 + right.0, left.1 + right.1, left.2 + right.2, left.3 + right.3),
        );

    eprintln!(
        "[topology-local-1m] validated range [{start_index}, {end_index}) {} cases, planar={}, outerplanar={}, k23={}, k33={}",
        cases.len(),
        planar_count,
        outerplanar_count,
        k23_count,
        k33_count
    );
}

#[test]
#[ignore = "expensive combined local oracle corpus; run manually when needed"]
#[allow(clippy::too_many_lines)]
fn test_local_topological_validity_reference_corpus_1m_with_k4() {
    let suite = assert_local_corpus_contract(LOCAL_CORPUS_1M_PATH, EXPECTED_1M_CASE_COUNT);
    let start_index = std::env::var("TOPOLOGY_START_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let end_index = std::env::var("TOPOLOGY_END_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(suite.cases.len());
    assert!(start_index <= end_index);
    assert!(end_index <= suite.cases.len());
    let cases = &suite.cases[start_index..end_index];

    eprintln!(
        "[topology-local-1m+k4] validating range [{start_index}, {end_index}) count={}",
        cases.len()
    );
    eprintln!("[topology-local-1m+k4] progress 0");
    let completed = AtomicUsize::new(0);
    let slow_threshold_ms =
        std::env::var("TOPOLOGY_TRACE_SLOW_MS").ok().and_then(|value| value.parse::<u128>().ok());

    let (planar_count, outerplanar_count, k23_count, k33_count, k4_count) = cases
        .par_iter()
        .enumerate()
        .map(|(local_index, case)| {
            let global_index = start_index + local_index;
            let overall_start = Instant::now();
            let graph = topological_validity_fixture::build_undigraph(case);

            let planar_start = Instant::now();
            let is_planar = graph
                .is_planar()
                .unwrap_or_else(|error| panic!("planarity failed on {}: {error}", case.name));
            let planar_elapsed = planar_start.elapsed();

            let outerplanar_start = Instant::now();
            let is_outerplanar = graph
                .is_outerplanar()
                .unwrap_or_else(|error| panic!("outerplanarity failed on {}: {error}", case.name));
            let outerplanar_elapsed = outerplanar_start.elapsed();

            let k23_start = Instant::now();
            let has_k23_homeomorph = graph.has_k23_homeomorph().unwrap_or_else(|error| {
                panic!("K23 homeomorph detection failed on {}: {error}", case.name)
            });
            let k23_elapsed = k23_start.elapsed();

            let k33_start = Instant::now();
            let has_k33_homeomorph = graph.has_k33_homeomorph().unwrap_or_else(|error| {
                panic!("K33 homeomorph detection failed on {}: {error}", case.name)
            });
            let k33_elapsed = k33_start.elapsed();

            let k4_start = Instant::now();
            let has_k4_homeomorph = graph
                .has_k4_homeomorph()
                .unwrap_or_else(|error| panic!("K4 homeomorph detection failed on {}: {error}", case.name));
            let k4_elapsed = k4_start.elapsed();

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
            assert_eq!(
                has_k23_homeomorph, case.has_k23_homeomorph,
                "K23 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert_eq!(
                has_k33_homeomorph, case.has_k33_homeomorph,
                "K33 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert_eq!(
                has_k4_homeomorph, case.has_k4_homeomorph,
                "K4 homeomorph mismatched local reference case {} ({})",
                case.name, case.family
            );
            assert!(
                !is_outerplanar || is_planar,
                "outerplanar local reference case {} must also be planar",
                case.name
            );

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10_000 == 0 {
                eprintln!("[topology-local-1m+k4] progress {done}");
            }

            let overall_elapsed = overall_start.elapsed();
            if let Some(threshold_ms) = slow_threshold_ms {
                if overall_elapsed.as_millis() >= threshold_ms {
                    eprintln!(
                        "[topology-local-1m+k4-slow] index={} case={} family={} total={:?} planar={:?} outerplanar={:?} k23={:?} k33={:?} k4={:?}",
                        global_index,
                        case.name,
                        case.family,
                        overall_elapsed,
                        planar_elapsed,
                        outerplanar_elapsed,
                        k23_elapsed,
                        k33_elapsed,
                        k4_elapsed
                    );
                }
            }

            (
                usize::from(is_planar),
                usize::from(is_outerplanar),
                usize::from(has_k23_homeomorph),
                usize::from(has_k33_homeomorph),
                usize::from(has_k4_homeomorph),
            )
        })
        .reduce(
            || (0usize, 0usize, 0usize, 0usize, 0usize),
            |left, right| {
                (
                    left.0 + right.0,
                    left.1 + right.1,
                    left.2 + right.2,
                    left.3 + right.3,
                    left.4 + right.4,
                )
            },
        );

    eprintln!(
        "[topology-local-1m+k4] validated range [{start_index}, {end_index}) {} cases, planar={}, outerplanar={}, k23={}, k33={}, k4={}",
        cases.len(),
        planar_count,
        outerplanar_count,
        k23_count,
        k33_count,
        k4_count
    );
}
