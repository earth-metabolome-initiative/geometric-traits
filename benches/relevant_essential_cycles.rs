//! Criterion benchmarks for the exact relevant-cycles and essential-cycles
//! traits.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            complete_graph, cycle_graph, friendship_graph, grid_graph, hypercube_graph, path_graph,
        },
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct SemanticBenchCase {
    name: String,
    graph: UndiGraph<usize>,
    expected_relevant: Vec<Vec<usize>>,
    expected_essential: Vec<Vec<usize>>,
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

fn build_undigraph_from_edges(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    let mut normalized_edges = edges
        .iter()
        .copied()
        .map(|[left, right]| if left <= right { (left, right) } else { (right, left) })
        .collect::<Vec<_>>();
    normalized_edges.sort_unstable();
    normalized_edges.dedup();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn build_semantic_cases() -> Vec<SemanticBenchCase> {
    [
        (
            "tree_path_5",
            build_undigraph_from_edges(5, &[[0, 1], [1, 2], [2, 3], [3, 4]]),
            vec![],
            vec![],
        ),
        (
            "triangle",
            build_undigraph_from_edges(3, &[[0, 1], [1, 2], [0, 2]]),
            vec![vec![0, 1, 2]],
            vec![vec![0, 1, 2]],
        ),
        (
            "square_with_diagonal",
            build_undigraph_from_edges(4, &[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]),
            vec![vec![0, 1, 2], vec![0, 2, 3]],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        ),
        (
            "k4",
            build_undigraph_from_edges(4, &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
            vec![],
        ),
        (
            "articulation_triangles",
            build_undigraph_from_edges(5, &[[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [2, 4]]),
            vec![vec![0, 1, 2], vec![2, 3, 4]],
            vec![vec![0, 1, 2], vec![2, 3, 4]],
        ),
    ]
    .into_iter()
    .map(|(name, graph, expected_relevant, expected_essential)| {
        SemanticBenchCase {
            name: name.to_string(),
            graph,
            expected_relevant: normalize_cycles(expected_relevant),
            expected_essential: normalize_cycles(expected_essential),
        }
    })
    .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_1024", wrap_undi(path_graph(1024))),
        ("cycle_512", wrap_undi(cycle_graph(512))),
        ("friendship_128", wrap_undi(friendship_graph(128))),
        ("grid_4x4", wrap_undi(grid_graph(4, 4))),
        ("hypercube_4", wrap_undi(hypercube_graph(4))),
        ("complete_6", wrap_undi(complete_graph(6))),
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

fn normalize_cycles(mut cycles: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for cycle in &mut cycles {
        *cycle = normalize_cycle(cycle.clone());
    }
    cycles
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    cycles
}

fn normalize_cycle(mut cycle: Vec<usize>) -> Vec<usize> {
    if cycle.is_empty() {
        return cycle;
    }
    let start =
        cycle.iter().enumerate().min_by_key(|(_, node)| **node).map_or(0, |(index, _)| index);
    cycle.rotate_left(start);
    if cycle.len() > 2 && cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }
    cycle
}

fn relevant_checksum(result: &RelevantCyclesResult<usize>) -> u64 {
    result
        .relevant_cycles()
        .fold(0u64, |checksum, cycle| {
            cycle.iter().fold(checksum.wrapping_mul(1_099_511_628_211), |acc, &node| {
                acc.wrapping_mul(257).wrapping_add(node as u64 + 1)
            })
        })
        .wrapping_add(result.len() as u64)
}

fn essential_checksum(result: &EssentialCyclesResult<usize>) -> u64 {
    result
        .essential_cycles()
        .fold(0u64, |checksum, cycle| {
            cycle.iter().fold(checksum.wrapping_mul(1_099_511_628_211), |acc, &node| {
                acc.wrapping_mul(257).wrapping_add(node as u64 + 1)
            })
        })
        .wrapping_add(result.len() as u64)
}

fn assert_semantic_cases(cases: &[SemanticBenchCase]) {
    for case in cases {
        let relevant = case.graph.relevant_cycles().unwrap();
        let essential = case.graph.essential_cycles().unwrap();

        assert_eq!(
            normalize_cycles(relevant.relevant_cycles().cloned().collect::<Vec<_>>()),
            case.expected_relevant,
            "semantic benchmark case {} mismatched relevant cycles",
            case.name
        );
        assert_eq!(
            normalize_cycles(essential.essential_cycles().cloned().collect::<Vec<_>>()),
            case.expected_essential,
            "semantic benchmark case {} mismatched essential cycles",
            case.name
        );
    }
}

fn bench_semantic_relevant(c: &mut Criterion) {
    let cases = build_semantic_cases();
    assert_semantic_cases(&cases);

    let mut group = c.benchmark_group("relevant_cycles_semantic_total");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("case count should fit into u64"),
    ));
    group.bench_function(BenchmarkId::from_parameter("all"), |b| {
        b.iter(|| {
            let checksum = cases.iter().fold(0u64, |checksum, case| {
                let result = case.graph.relevant_cycles().unwrap();
                checksum.wrapping_mul(1_099_511_628_211).wrapping_add(relevant_checksum(&result))
            });
            black_box(checksum)
        });
    });
    group.finish();
}

fn bench_semantic_essential(c: &mut Criterion) {
    let cases = build_semantic_cases();
    assert_semantic_cases(&cases);

    let mut group = c.benchmark_group("essential_cycles_semantic_total");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("case count should fit into u64"),
    ));
    group.bench_function(BenchmarkId::from_parameter("all"), |b| {
        b.iter(|| {
            let checksum = cases.iter().fold(0u64, |checksum, case| {
                let result = case.graph.essential_cycles().unwrap();
                checksum.wrapping_mul(1_099_511_628_211).wrapping_add(essential_checksum(&result))
            });
            black_box(checksum)
        });
    });
    group.finish();
}

fn bench_scaling_relevant(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    let mut group = c.benchmark_group("relevant_cycles_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for case in cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        group.bench_with_input(BenchmarkId::from_parameter(&case.name), &case, |b, case| {
            b.iter(|| {
                let result = case.graph.relevant_cycles().unwrap();
                black_box(relevant_checksum(&result))
            });
        });
    }
    group.finish();
}

fn bench_scaling_essential(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    let mut group = c.benchmark_group("essential_cycles_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for case in cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        group.bench_with_input(BenchmarkId::from_parameter(&case.name), &case, |b, case| {
            b.iter(|| {
                let result = case.graph.essential_cycles().unwrap();
                black_box(essential_checksum(&result))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_semantic_relevant,
    bench_semantic_essential,
    bench_scaling_relevant,
    bench_scaling_essential
);
criterion_main!(benches);
