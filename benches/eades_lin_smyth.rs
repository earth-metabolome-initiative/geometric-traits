//! Criterion benchmarks for the Eades-Lin-Smyth (GR) feedback-arc-set
//! heuristic, compared against an independent reference implementation.
//!
//! The fixture `eades_lin_smyth_ground_truth.json.gz` stores, per case, the
//! reference implementation's minimum per-call wall-clock time in nanoseconds.
//! Each benchmark id carries that time so the Rust timings can be read directly
//! against the reference, and the Rust output (cut and weights) is validated
//! against the reference before timing.

#[path = "../tests/support/fixture_io.rs"]
mod fixture_io;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::{GenericBiMatrix2D, ValuedCSR2D},
    prelude::*,
};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;
type WeightedBiMatrix = GenericBiMatrix2D<WeightedMatrix, WeightedMatrix>;

const FIXTURE_NAME: &str = "eades_lin_smyth_ground_truth.json.gz";

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
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
    feedback_edges: Vec<(usize, usize)>,
    feedback_weight: f64,
    total_weight: f64,
    nanos: u64,
}

fn build_graph(case: &Case) -> WeightedBiMatrix {
    let mut edges = case.edges.clone();
    edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((case.node_count, case.node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericBiMatrix2D::new(matrix)
}

struct PreparedCase {
    graph: WeightedBiMatrix,
    edge_count: u64,
    label: String,
}

fn prepare() -> Vec<PreparedCase> {
    let fixture: Fixture = load_fixture_json(FIXTURE_NAME);
    let mut prepared = Vec::with_capacity(fixture.cases.len());

    for case in &fixture.cases {
        let graph = build_graph(case);
        let reference = &case.reference;

        // Validate the Rust output against the reference before timing. The
        // fixtures use distinct integer weights, so the GR cut is
        // tie-break-independent and the two must agree on the exact edge set.
        let result = graph.eades_lin_smyth().unwrap();
        let total_scale = reference.total_weight.abs().max(1.0);
        assert!(
            (result.total_weight() - reference.total_weight).abs() <= 1.0e-9 * total_scale,
            "case `{}`: total weight {} != reference {}",
            case.id,
            result.total_weight(),
            reference.total_weight,
        );
        let feedback_scale = reference.feedback_weight.abs().max(1.0);
        assert!(
            (result.feedback_weight() - reference.feedback_weight).abs() <= 1.0e-9 * feedback_scale,
            "case `{}`: feedback weight {} != reference {}",
            case.id,
            result.feedback_weight(),
            reference.feedback_weight,
        );
        assert_eq!(
            result.feedback_edges(),
            reference.feedback_edges.as_slice(),
            "case `{}`: feedback edge set differs from reference",
            case.id,
        );

        prepared.push(PreparedCase {
            graph,
            edge_count: u64::try_from(case.edges.len()).unwrap(),
            label: format!("{}_ref={}ns", case.id, reference.nanos),
        });
    }

    prepared
}

fn bench(criterion: &mut Criterion) {
    let cases = prepare();

    let mut group = criterion.benchmark_group("eades_lin_smyth");
    for case in &cases {
        group.throughput(Throughput::Elements(case.edge_count));
        group.bench_with_input(
            BenchmarkId::from_parameter(&case.label),
            &case.graph,
            |bencher, graph| {
                bencher.iter(|| {
                    let result = black_box(graph).eades_lin_smyth().unwrap();
                    black_box(result.feedback_weight());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
