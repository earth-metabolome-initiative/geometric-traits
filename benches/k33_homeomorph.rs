//! Criterion benchmarks for `K_{3,3}` homeomorph detection.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        K33HomeomorphDetection, MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            complete_bipartite_graph, cycle_graph, path_graph, wheel_graph,
        },
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_has_k33_homeomorph: bool,
}

#[derive(Clone)]
struct ScalingBenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_has_k33_homeomorph: bool,
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

fn normalize_edge([left, right]: [usize; 2]) -> [usize; 2] {
    if left <= right { [left, right] } else { [right, left] }
}

fn build_undigraph(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    let mut normalized_edges: Vec<(usize, usize)> = edges
        .iter()
        .copied()
        .map(normalize_edge)
        .map(|[source, destination]| (source, destination))
        .collect();
    normalized_edges.sort_unstable();
    let matrix: UndirectedAdjacency = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn prepare_fixture_cases() -> Vec<FixtureBenchCase> {
    [
        ("path_four", "tree", build_undigraph(4, &[[0, 1], [1, 2], [2, 3]]), false),
        (
            "k23_complete_bipartite",
            "k23",
            build_undigraph(5, &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]),
            false,
        ),
        (
            "k33_complete_bipartite",
            "k33",
            build_undigraph(
                6,
                &[[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            ),
            true,
        ),
        (
            "k33_subdivision",
            "k33_subdivision",
            build_undigraph(
                15,
                &[
                    [0, 6],
                    [6, 3],
                    [0, 7],
                    [7, 4],
                    [0, 8],
                    [8, 5],
                    [1, 9],
                    [9, 3],
                    [1, 10],
                    [10, 4],
                    [1, 11],
                    [11, 5],
                    [2, 12],
                    [12, 3],
                    [2, 13],
                    [13, 4],
                    [2, 14],
                    [14, 5],
                ],
            ),
            true,
        ),
        (
            "k5_complete",
            "k5",
            build_undigraph(
                5,
                &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            ),
            false,
        ),
        (
            "k6_complete",
            "clique",
            build_undigraph(
                6,
                &[
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [0, 5],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [1, 5],
                    [2, 3],
                    [2, 4],
                    [2, 5],
                    [3, 4],
                    [3, 5],
                    [4, 5],
                ],
            ),
            true,
        ),
    ]
    .into_iter()
    .map(|(name, family, graph, expected_has_k33_homeomorph)| {
        FixtureBenchCase {
            name: name.to_string(),
            family: family.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_has_k33_homeomorph,
        }
    })
    .collect()
}

#[allow(clippy::too_many_lines)]
fn prepare_regression_cases() -> Vec<FixtureBenchCase> {
    [
        (
            "fuzzer_regression_20260411",
            "fuzzer",
            build_undigraph(
                15,
                &[
                    [0, 6],
                    [0, 9],
                    [0, 10],
                    [0, 12],
                    [0, 13],
                    [3, 9],
                    [3, 10],
                    [3, 12],
                    [3, 13],
                    [5, 9],
                    [7, 10],
                    [9, 10],
                    [9, 13],
                    [10, 11],
                    [10, 12],
                    [11, 13],
                    [12, 13],
                    [12, 14],
                    [13, 14],
                ],
            ),
            true,
        ),
        (
            "fuzzer_regression_20260411_b",
            "fuzzer",
            build_undigraph(
                15,
                &[
                    [0, 6],
                    [0, 9],
                    [0, 12],
                    [0, 13],
                    [3, 9],
                    [3, 10],
                    [3, 12],
                    [3, 13],
                    [5, 9],
                    [5, 13],
                    [6, 14],
                    [7, 10],
                    [9, 10],
                    [9, 13],
                    [10, 11],
                    [10, 14],
                    [11, 13],
                    [12, 14],
                ],
            ),
            true,
        ),
        (
            "fuzzer_regression_20260412",
            "fuzzer",
            build_undigraph(
                16,
                &[
                    [0, 1],
                    [0, 7],
                    [0, 11],
                    [0, 15],
                    [1, 5],
                    [1, 11],
                    [2, 4],
                    [3, 5],
                    [3, 10],
                    [3, 15],
                    [4, 5],
                    [4, 12],
                    [4, 13],
                    [5, 6],
                    [5, 7],
                    [5, 10],
                    [5, 11],
                    [5, 12],
                    [5, 15],
                    [7, 8],
                    [7, 11],
                    [7, 15],
                    [8, 9],
                    [9, 15],
                    [11, 12],
                    [11, 15],
                    [14, 15],
                ],
            ),
            true,
        ),
    ]
    .into_iter()
    .map(|(name, family, graph, expected_has_k33_homeomorph)| {
        FixtureBenchCase {
            name: name.to_string(),
            family: family.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_has_k33_homeomorph,
        }
    })
    .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4_096)), false),
        ("cycle_4096", wrap_undi(cycle_graph(4_096)), false),
        ("wheel_2048", wrap_undi(wheel_graph(2_048)), false),
        ("complete_bipartite_k3_2048", wrap_undi(complete_bipartite_graph(3, 2_048)), true),
        ("complete_bipartite_k5_1024", wrap_undi(complete_bipartite_graph(5, 1_024)), true),
    ]
    .into_iter()
    .map(|(name, graph, expected_has_k33_homeomorph)| {
        ScalingBenchCase {
            name: name.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_has_k33_homeomorph,
        }
    })
    .collect()
}

fn assert_cases_match_oracle(cases: &[FixtureBenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.has_k33_homeomorph().unwrap(),
            case.expected_has_k33_homeomorph,
            "benchmark fixture case {} ({}) mismatched K33 oracle",
            case.name,
            case.family
        );
    }
}

fn assert_scaling_cases_match_oracle(cases: &[ScalingBenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.has_k33_homeomorph().unwrap(),
            case.expected_has_k33_homeomorph,
            "benchmark scaling case {} mismatched K33 oracle",
            case.name
        );
    }
}

fn bench_semantic_cases(c: &mut Criterion) {
    let cases = prepare_fixture_cases();
    assert_cases_match_oracle(&cases);

    let mut group = c.benchmark_group("k33_homeomorph_semantic_total");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));
    group.bench_function(
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
                for case in &cases {
                    let detected = case.graph.has_k33_homeomorph().unwrap();
                    black_box(detected);
                }
            });
        },
    );
    group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    assert_scaling_cases_match_oracle(&cases);

    let mut group = c.benchmark_group("k33_homeomorph_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    for case in cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        group.bench_with_input(BenchmarkId::new("case", &case.name), &case, |b, bench_case| {
            b.iter(|| black_box(bench_case.graph.has_k33_homeomorph().unwrap()));
        });
    }
    group.finish();
}

fn bench_regression_cases(c: &mut Criterion) {
    let cases = prepare_regression_cases();
    assert_cases_match_oracle(&cases);

    let mut group = c.benchmark_group("k33_homeomorph_regressions");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for case in &cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| black_box(case.graph.has_k33_homeomorph().unwrap()));
        });
    }
    group.finish();
}

criterion_group!(k33_homeomorph, bench_semantic_cases, bench_scaling_cases, bench_regression_cases);
criterion_main!(k33_homeomorph);
