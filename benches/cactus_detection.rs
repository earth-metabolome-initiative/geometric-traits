//! Criterion benchmarks for cactus graph detection.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        CactusDetection, MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            barbell_graph, complete_graph, cycle_graph, erdos_renyi_gnp, friendship_graph,
            grid_graph, path_graph,
        },
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct BenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_is_cactus: bool,
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

fn prepare_scaling_cases() -> Vec<BenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4096)), true),
        ("cycle_4096", wrap_undi(cycle_graph(4096)), true),
        ("friendship_2048", wrap_undi(friendship_graph(2048)), true),
        ("grid_64x64", wrap_undi(grid_graph(64, 64)), false),
        ("barbell_64_1024", wrap_undi(barbell_graph(64, 1024)), false),
        ("complete_256", wrap_undi(complete_graph(256)), false),
        (
            "erdos_renyi_2048_p0008",
            wrap_undi(erdos_renyi_gnp(0xCAC7_05D3_7EC7_10DE, 2048, 0.008)),
            false,
        ),
    ]
    .into_iter()
    .map(|(name, graph, expected_is_cactus)| {
        BenchCase {
            name: name.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_is_cactus,
        }
    })
    .collect()
}

fn assert_cases_match_expectations(cases: &[BenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.is_cactus(),
            case.expected_is_cactus,
            "benchmark case {} mismatched cactus expectation",
            case.name
        );
    }
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    assert_cases_match_expectations(&cases);

    let mut case_group = c.benchmark_group("cactus_detection_scaling");
    case_group.sample_size(10);
    case_group.warm_up_time(Duration::from_millis(500));
    case_group.measurement_time(Duration::from_secs(2));

    for case in &cases {
        case_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        case_group.bench_function(BenchmarkId::new("case", &case.name), {
            let graph = case.graph.clone();
            move |b| {
                b.iter(|| {
                    let is_cactus = graph.is_cactus();
                    black_box(is_cactus);
                });
            }
        });
    }
    case_group.finish();
}

criterion_group!(benches, bench_scaling_cases);
criterion_main!(benches);
