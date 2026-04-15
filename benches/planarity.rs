//! Criterion benchmarks for planarity and outerplanarity detection.

#[path = "../tests/support/planarity_fixture.rs"]
mod planarity_fixture;

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, OuterplanarityDetection, PlanarityDetection, SquareMatrix,
        VocabularyBuilder,
        algorithms::randomized_graphs::{
            cycle_graph, erdos_renyi_gnp, grid_graph, path_graph, wheel_graph,
        },
    },
};
use planarity_fixture::{build_undigraph, semantic_cases};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_planar: bool,
    expected_outerplanar: bool,
}

#[derive(Clone)]
struct ScalingBenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
}

fn wrap_undi(graph: UndirectedAdjacency) -> UndiGraph<usize> {
    let order = graph.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, graph))
}

fn logical_edge_count(graph: &UndiGraph<usize>) -> usize {
    graph.sparse_coordinates().filter(|&(source, destination)| source <= destination).count()
}

fn prepare_fixture_cases() -> Vec<FixtureBenchCase> {
    semantic_cases()
        .into_iter()
        .map(|case| {
            FixtureBenchCase {
                logical_edge_count: case.edges.len(),
                graph: build_undigraph(&case),
                name: case.name,
                family: case.family,
                expected_planar: case.is_planar,
                expected_outerplanar: case.is_outerplanar,
            }
        })
        .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4_096))),
        ("cycle_4096", wrap_undi(cycle_graph(4_096))),
        ("wheel_2048", wrap_undi(wheel_graph(2_048))),
        ("grid_64x64", wrap_undi(grid_graph(64, 64))),
        (
            "erdos_renyi_2048_p0005",
            wrap_undi(erdos_renyi_gnp(0xA11C_E5E5_0005_2048, 2_048, 0.0005)),
        ),
        (
            "erdos_renyi_2048_p0020",
            wrap_undi(erdos_renyi_gnp(0xA11C_E5E5_0020_2048, 2_048, 0.0020)),
        ),
    ]
    .into_iter()
    .map(|(name, graph)| {
        ScalingBenchCase {
            name: name.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
        }
    })
    .collect()
}

fn assert_fixture_cases_match_oracle(cases: &[FixtureBenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.is_planar().unwrap(),
            case.expected_planar,
            "benchmark fixture case {} ({}) mismatched planarity oracle",
            case.name,
            case.family
        );
        assert_eq!(
            case.graph.is_outerplanar().unwrap(),
            case.expected_outerplanar,
            "benchmark fixture case {} ({}) mismatched outerplanarity oracle",
            case.name,
            case.family
        );
    }
}

fn bench_semantic_cases(c: &mut Criterion) {
    let cases = prepare_fixture_cases();
    assert_fixture_cases_match_oracle(&cases);

    let mut planarity_group = c.benchmark_group("planarity_semantic_total");
    planarity_group.sample_size(10);
    planarity_group.warm_up_time(Duration::from_millis(500));
    planarity_group.measurement_time(Duration::from_secs(2));
    planarity_group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));
    planarity_group.bench_function(
        BenchmarkId::new(
            "total_cases",
            format!(
                "cases={}_logical_edges={}",
                cases.len(),
                cases.iter().map(|case| case.logical_edge_count).sum::<usize>()
            ),
        ),
        |b| {
            b.iter(|| {
                let checksum = cases
                    .iter()
                    .map(|case| u64::from(case.graph.is_planar().unwrap()))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    planarity_group.finish();

    let mut outerplanarity_group = c.benchmark_group("outerplanarity_semantic_total");
    outerplanarity_group.sample_size(10);
    outerplanarity_group.warm_up_time(Duration::from_millis(500));
    outerplanarity_group.measurement_time(Duration::from_secs(2));
    outerplanarity_group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));
    outerplanarity_group.bench_function(
        BenchmarkId::new(
            "total_cases",
            format!(
                "cases={}_logical_edges={}",
                cases.len(),
                cases.iter().map(|case| case.logical_edge_count).sum::<usize>()
            ),
        ),
        |b| {
            b.iter(|| {
                let checksum = cases
                    .iter()
                    .map(|case| u64::from(case.graph.is_outerplanar().unwrap()))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    outerplanarity_group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();

    let mut planarity_group = c.benchmark_group("planarity_scaling");
    planarity_group.sample_size(10);
    planarity_group.warm_up_time(Duration::from_millis(500));
    planarity_group.measurement_time(Duration::from_secs(2));
    for case in &cases {
        planarity_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        planarity_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| black_box(case.graph.is_planar().unwrap()));
        });
    }
    planarity_group.finish();

    let mut outerplanarity_group = c.benchmark_group("outerplanarity_scaling");
    outerplanarity_group.sample_size(10);
    outerplanarity_group.warm_up_time(Duration::from_millis(500));
    outerplanarity_group.measurement_time(Duration::from_secs(2));
    for case in &cases {
        outerplanarity_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        outerplanarity_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| black_box(case.graph.is_outerplanar().unwrap()));
        });
    }
    outerplanarity_group.finish();
}

criterion_group!(benches, bench_semantic_cases, bench_scaling_cases);
criterion_main!(benches);
