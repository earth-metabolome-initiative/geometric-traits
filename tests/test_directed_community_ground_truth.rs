//! Cross-validation of the directed Louvain and directed Leiden detectors
//! against the reference C++ Directed Louvain implementation
//! (github.com/nicolasdugue/DirectedLouvain).
//!
//! The fixture `directed_community_ground_truth.json.gz` is a one-shot
//! recording of that reference (the generator is under
//! `tests/fixtures/generators/directed_community/`, not part of the build).
//! Each case stores a directed weighted edge list plus the reference final
//! partition and the directed modularity it achieves. The reference was run
//! deterministic (no shuffle, pre-numbered nodes) at resolution 1.
//!
//! Our metric must reproduce the reference modularity on the reference
//! partition exactly, and our greedy detectors must reach at least the
//! reference modularity within a small margin (they use a different move order,
//! so they are not expected to match bit for bit).
#![cfg(feature = "std")]

#[path = "support/fixture_io.rs"]
mod fixture_io;

use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{DirectedLeidenConfig, DirectedLouvainConfig},
};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const FIXTURE_NAME: &str = "directed_community_ground_truth.json.gz";

/// Our greedy detectors run a randomized move order, so they can land a hair
/// below the deterministic reference on a few cases. This margin bounds that
/// gap while still catching real regressions.
const QUALITY_MARGIN: f64 = 0.02;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    parameters: Parameters,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Parameters {
    resolution: f64,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    node_count: usize,
    edges: Vec<(usize, usize, f64)>,
    reference: Reference,
}

#[derive(Debug, Deserialize)]
struct Reference {
    partition: Vec<usize>,
    modularity: f64,
}

fn build_directed_graph(case: &Case) -> WeightedMatrix {
    let mut edges = case.edges.clone();
    edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((case.node_count, case.node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn approx_equal(left: f64, right: f64, relative_tolerance: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * relative_tolerance
}

fn fixture() -> Fixture {
    load_fixture_json(FIXTURE_NAME)
}

#[test]
fn test_fixture_metadata() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert!(approx_equal(fixture.parameters.resolution, 1.0, 1.0e-12));
    assert!(fixture.cases.len() >= 10, "expected a non-trivial suite");
    for case in &fixture.cases {
        assert_eq!(case.reference.partition.len(), case.node_count, "case `{}`", case.id);
        assert!(!case.edges.is_empty(), "case `{}` must have arcs", case.id);
    }
}

#[test]
fn test_our_metric_matches_reference_on_reference_partition() {
    let fixture = fixture();
    let resolution = fixture.parameters.resolution;

    for case in &fixture.cases {
        let graph = build_directed_graph(case);
        let ours = DirectedModularity::<usize>::directed_modularity(
            &graph,
            &case.reference.partition,
            resolution,
        )
        .unwrap();
        assert!(
            approx_equal(ours, case.reference.modularity, 1.0e-9),
            "case `{}`: ours={ours:.12}, reference={:.12}",
            case.id,
            case.reference.modularity,
        );
    }
}

#[test]
fn test_directed_louvain_reaches_reference_quality() {
    let fixture = fixture();
    let config =
        DirectedLouvainConfig { resolution: fixture.parameters.resolution, ..Default::default() };

    for case in &fixture.cases {
        let graph = build_directed_graph(case);
        let result = DirectedLouvain::<usize>::directed_louvain(&graph, &config).unwrap();

        // Self-consistency: the reported modularity equals the directed
        // modularity of the detected partition.
        let recomputed = DirectedModularity::<usize>::directed_modularity(
            &graph,
            result.final_partition(),
            config.resolution,
        )
        .unwrap();
        assert!(
            approx_equal(result.final_modularity(), recomputed, 1.0e-9),
            "case `{}`: reported {:.12} != recomputed {recomputed:.12}",
            case.id,
            result.final_modularity(),
        );

        assert!(
            result.final_modularity() >= case.reference.modularity - QUALITY_MARGIN,
            "case `{}`: directed Louvain {:.6} far below reference {:.6}",
            case.id,
            result.final_modularity(),
            case.reference.modularity,
        );
    }
}

#[test]
fn test_directed_leiden_reaches_reference_quality() {
    let fixture = fixture();
    let config =
        DirectedLeidenConfig { resolution: fixture.parameters.resolution, ..Default::default() };

    for case in &fixture.cases {
        let graph = build_directed_graph(case);
        let result = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap();

        let recomputed = DirectedModularity::<usize>::directed_modularity(
            &graph,
            result.final_partition(),
            config.resolution,
        )
        .unwrap();
        assert!(
            approx_equal(result.final_modularity(), recomputed, 1.0e-9),
            "case `{}`: reported {:.12} != recomputed {recomputed:.12}",
            case.id,
            result.final_modularity(),
        );

        assert!(
            result.final_modularity() >= case.reference.modularity - QUALITY_MARGIN,
            "case `{}`: directed Leiden {:.6} far below reference {:.6}",
            case.id,
            result.final_modularity(),
            case.reference.modularity,
        );
    }
}
