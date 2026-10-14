//! Criterion benchmarks for `K_{2,3}` homeomorph detection.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        K23HomeomorphDetection, MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{cycle_graph, path_graph, wheel_graph},
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_has_k23_homeomorph: bool,
}

#[derive(Clone)]
struct ScalingBenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_has_k23_homeomorph: bool,
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

fn complete_bipartite_graph(left_order: usize, right_order: usize) -> UndirectedAdjacency {
    let order = left_order + right_order;
    let edges =
        (0..left_order).flat_map(|left| (left_order..order).map(move |right| (left, right)));
    UndiEdgesBuilder::default()
        .expected_number_of_edges(left_order * right_order)
        .expected_shape(order)
        .edges(edges)
        .build()
        .unwrap()
}

fn theta_graph(order: usize) -> UndirectedAdjacency {
    assert!(order >= 5, "theta_graph requires at least five vertices");

    let mut internal_counts = [1usize, 1usize, 1usize];
    for extra_index in 0..(order - 5) {
        internal_counts[extra_index % 3] += 1;
    }

    let mut next_vertex = 2usize;
    let mut edges = Vec::with_capacity(order + 1);
    for internal_count in internal_counts {
        let mut previous = 0usize;
        for _ in 0..internal_count {
            let current = next_vertex;
            next_vertex += 1;
            let [left, right] = normalize_edge([previous, current]);
            edges.push((left, right));
            previous = current;
        }
        let [left, right] = normalize_edge([previous, 1usize]);
        edges.push((left, right));
    }
    edges.sort_unstable();

    UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(order)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn prepare_fixture_cases() -> Vec<FixtureBenchCase> {
    [
        ("path_four", "tree", build_undigraph(4, &[[0, 1], [1, 2], [2, 3]]), false),
        (
            "diamond_k4_minus_edge",
            "outerplanar_with_chord",
            build_undigraph(4, &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]),
            false,
        ),
        (
            "k4_complete",
            "outerplanarity_obstruction",
            build_undigraph(4, &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
            false,
        ),
        (
            "k23_complete_bipartite",
            "k23",
            build_undigraph(5, &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]),
            true,
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
            "k5_complete",
            "k5",
            build_undigraph(
                5,
                &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            ),
            true,
        ),
        (
            "theta_three_length_two_paths",
            "theta",
            build_undigraph(5, &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]]),
            true,
        ),
    ]
    .into_iter()
    .map(|(name, family, graph, expected_has_k23_homeomorph)| {
        FixtureBenchCase {
            name: name.to_string(),
            family: family.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_has_k23_homeomorph,
        }
    })
    .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4_096)), false),
        ("cycle_4096", wrap_undi(cycle_graph(4_096)), false),
        ("wheel_2048", wrap_undi(wheel_graph(2_048)), true),
        ("theta_4096", wrap_undi(theta_graph(4_096)), true),
        ("complete_bipartite_k2_2048", wrap_undi(complete_bipartite_graph(2, 2_048)), true),
    ]
    .into_iter()
    .map(|(name, graph, expected_has_k23_homeomorph)| {
        ScalingBenchCase {
            name: name.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_has_k23_homeomorph,
        }
    })
    .collect()
}

fn assert_cases_match_oracle(cases: &[FixtureBenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.has_k23_homeomorph().unwrap(),
            case.expected_has_k23_homeomorph,
            "benchmark fixture case {} ({}) mismatched K23 oracle",
            case.name,
            case.family
        );
    }
}

fn assert_scaling_cases_match_oracle(cases: &[ScalingBenchCase]) {
    for case in cases {
        assert_eq!(
            case.graph.has_k23_homeomorph().unwrap(),
            case.expected_has_k23_homeomorph,
            "benchmark scaling case {} mismatched K23 oracle",
            case.name
        );
    }
}

fn bench_semantic_cases(c: &mut Criterion) {
    let cases = prepare_fixture_cases();
    assert_cases_match_oracle(&cases);

    let mut group = c.benchmark_group("k23_homeomorph_semantic_total");
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
                let checksum = cases
                    .iter()
                    .map(|case| u64::from(case.graph.has_k23_homeomorph().unwrap()))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    assert_scaling_cases_match_oracle(&cases);

    let mut group = c.benchmark_group("k23_homeomorph_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for case in &cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| black_box(case.graph.has_k23_homeomorph().unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_semantic_cases, bench_scaling_cases);
criterion_main!(benches);
