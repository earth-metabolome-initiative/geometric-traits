//! Criterion benchmarks for the Leicht-Newman directed community detector and
//! the directed-modularity metric, compared against the NetworkX reference.
//!
//! The fixture `directed_modularity_ground_truth.json.gz` stores, per case, the
//! NetworkX `greedy_modularity_communities` wall-clock time in nanoseconds. The
//! benchmark group labels carry the summed NetworkX time so the Rust timings
//! can be read directly against the reference implementation.

#[path = "../tests/support/fixture_io.rs"]
mod fixture_io;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fixture_io::load_fixture_json;
use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LeichtNewmanConfig};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const FIXTURE_NAME: &str = "directed_modularity_ground_truth.json.gz";

#[derive(Debug, Deserialize)]
struct Fixture {
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
    reference: Option<Reference>,
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

struct PreparedCase {
    id: String,
    graph: WeightedMatrix,
    reference_partition: Option<Vec<usize>>,
}

fn prepare() -> (Vec<PreparedCase>, f64, u64, u64) {
    let fixture: Fixture = load_fixture_json(FIXTURE_NAME);
    let resolution = fixture.parameters.resolution;
    let mut prepared = Vec::with_capacity(fixture.cases.len());
    let mut networkx_total_nanos = 0_u64;
    let mut total_edges = 0_u64;

    for case in &fixture.cases {
        let graph = build_directed_graph(case);
        total_edges += u64::try_from(case.edges.len()).unwrap();

        let reference_partition = case.reference.as_ref().map(|reference| {
            networkx_total_nanos += reference.nanos;
            // Validate our metric against the stored NetworkX value before
            // timing.
            let ours = DirectedModularity::<usize>::directed_modularity(
                &graph,
                &reference.partition,
                resolution,
            )
            .unwrap();
            let scale = ours.abs().max(reference.modularity.abs()).max(1.0);
            assert!(
                (ours - reference.modularity).abs() <= scale * 1.0e-9,
                "case `{}`: directed modularity {ours} != NetworkX {}",
                case.id,
                reference.modularity,
            );
            reference.partition.clone()
        });

        prepared.push(PreparedCase { id: case.id.clone(), graph, reference_partition });
    }

    (prepared, resolution, networkx_total_nanos, total_edges)
}

fn bench(criterion: &mut Criterion) {
    let (cases, resolution, networkx_total_nanos, total_edges) = prepare();
    let config = LeichtNewmanConfig { resolution, ..LeichtNewmanConfig::default() };

    let label = format!("networkx_greedy_total={networkx_total_nanos}ns");

    let mut detector_group = criterion.benchmark_group("leicht_newman_detector");
    detector_group.throughput(Throughput::Elements(total_edges));
    detector_group.bench_function(BenchmarkId::new("rust_detect_total", &label), |bencher| {
        bencher.iter(|| {
            for case in &cases {
                let result = LeichtNewman::<usize>::leicht_newman(
                    black_box(&case.graph),
                    black_box(&config),
                )
                .unwrap();
                black_box(result.number_of_communities());
            }
        });
    });
    detector_group.finish();

    let mut metric_group = criterion.benchmark_group("directed_modularity_metric");
    metric_group.throughput(Throughput::Elements(total_edges));
    metric_group.bench_function(BenchmarkId::new("rust_modularity_total", &label), |bencher| {
        bencher.iter(|| {
            for case in &cases {
                if let Some(partition) = &case.reference_partition {
                    let value = DirectedModularity::<usize>::directed_modularity(
                        black_box(&case.graph),
                        black_box(partition),
                        resolution,
                    )
                    .unwrap();
                    black_box(value);
                }
            }
        });
    });
    metric_group.finish();

    // Touch every prepared id so the field is observably used.
    black_box(cases.iter().map(|case| case.id.len()).sum::<usize>());
}

criterion_group!(benches, bench);
criterion_main!(benches);
