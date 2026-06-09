//! Regression tests of the directed (Leicht-Newman) modularity metric against
//! NetworkX `community.modularity` on directed graphs.
//!
//! The fixture `directed_modularity_ground_truth.json.gz` is a one-shot
//! recording from NetworkX (the generator is not part of the repository). Each
//! case stores a directed weighted edge list plus several partitions, each
//! annotated with the directed modularity reported by NetworkX. The directed
//! branch of NetworkX `community.modularity` is exactly the Leicht-Newman 2008
//! generalization.
#![cfg(feature = "std")]

#[path = "support/fixture_io.rs"]
mod fixture_io;

use fixture_io::load_fixture_json;
use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LeichtNewmanConfig};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const FIXTURE_NAME: &str = "directed_modularity_ground_truth.json.gz";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    parameters: Parameters,
    networkx_version: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Parameters {
    resolution: f64,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    node_count: usize,
    edges: Vec<(usize, usize, f64)>,
    partitions: Vec<NamedPartition>,
    reference: Option<Reference>,
}

#[derive(Debug, Deserialize)]
struct NamedPartition {
    name: String,
    partition: Vec<usize>,
    modularity: f64,
}

#[derive(Debug, Deserialize)]
struct Reference {
    partition: Vec<usize>,
    modularity: f64,
    nanos: u64,
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
    assert_eq!(fixture.parameters.seed, 42);
    assert!(!fixture.networkx_version.is_empty());
    assert!(fixture.cases.len() >= 10, "expected a non-trivial suite");
    for case in &fixture.cases {
        assert!(!case.edges.is_empty(), "case `{}` must have edges", case.id);
        for partition in &case.partitions {
            assert_eq!(
                partition.partition.len(),
                case.node_count,
                "case `{}` partition `{}` length",
                case.id,
                partition.name,
            );
        }
    }
}

#[test]
fn test_directed_modularity_matches_networkx_on_every_partition() {
    let fixture = fixture();
    let resolution = fixture.parameters.resolution;

    for case in &fixture.cases {
        let graph = build_directed_graph(case);

        for named in &case.partitions {
            let ours = DirectedModularity::<usize>::directed_modularity(
                &graph,
                &named.partition,
                resolution,
            )
            .unwrap();
            assert!(
                approx_equal(ours, named.modularity, 1.0e-9),
                "case `{}` partition `{}`: ours={ours:.15}, networkx={:.15}",
                case.id,
                named.name,
                named.modularity,
            );
        }

        if let Some(reference) = &case.reference {
            let ours = DirectedModularity::<usize>::directed_modularity(
                &graph,
                &reference.partition,
                resolution,
            )
            .unwrap();
            assert!(
                approx_equal(ours, reference.modularity, 1.0e-9),
                "case `{}` reference partition: ours={ours:.15}, networkx={:.15}",
                case.id,
                reference.modularity,
            );
            // The NetworkX reference timing is recorded for the benchmark.
            assert!(reference.nanos > 0, "case `{}` reference must record timing", case.id);
        }
    }
}

/// The detector must beat the trivial partitions, stay close to the NetworkX
/// greedy reference, and report a modularity consistent with its own partition.
#[test]
fn test_detector_quality_vs_networkx() {
    // The Leicht-Newman spectral method matches NetworkX greedy exactly on
    // structured graphs. On a structureless random digraph it trails slightly,
    // which is the expected behavior of spectral bisection. The margin bounds
    // that gap while still catching gross regressions.
    const GREEDY_MARGIN: f64 = 0.05;

    let fixture = fixture();
    let resolution = fixture.parameters.resolution;

    for case in &fixture.cases {
        let graph = build_directed_graph(case);
        let result =
            LeichtNewman::<usize>::leicht_newman(&graph, &LeichtNewmanConfig::default()).unwrap();
        let detector_modularity = result.modularity();

        // Self-consistency: the reported modularity equals the directed
        // modularity of the detected partition.
        let recomputed = DirectedModularity::<usize>::directed_modularity(
            &graph,
            result.partition(),
            resolution,
        )
        .unwrap();
        assert!(
            approx_equal(detector_modularity, recomputed, 1.0e-9),
            "case `{}`: reported {detector_modularity:.12} != recomputed {recomputed:.12}",
            case.id,
        );

        // A real maximizer never does worse than the trivial partitions.
        let trivial = case
            .partitions
            .iter()
            .filter(|partition| {
                partition.name == "singletons" || partition.name == "single_community"
            })
            .map(|partition| partition.modularity)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            detector_modularity >= trivial - 1.0e-9,
            "case `{}`: detector {detector_modularity:.6} below trivial floor {trivial:.6}",
            case.id,
        );

        // Stay within a small margin of the NetworkX greedy reference.
        if let Some(reference) = &case.reference {
            assert!(
                detector_modularity >= reference.modularity - GREEDY_MARGIN,
                "case `{}`: detector {detector_modularity:.6} far below greedy {:.6}",
                case.id,
                reference.modularity,
            );
        }
    }
}
