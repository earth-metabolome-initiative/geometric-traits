//! Criterion benchmarks for triangle and square counting kernels.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        SquareMatrix, VocabularyBuilder,
        algorithms::{
            MotifCountOrdering, SquareCountScorer, TriangleCountScorer,
            randomized_graphs::{complete_graph, erdos_renyi_gnp, grid_graph, path_graph},
        },
    },
};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct ScalingCase {
    name: String,
    graph: UndiGraph<usize>,
}

fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

fn triangle_counts_natural<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    TriangleCountScorer::new(MotifCountOrdering::Natural).score_nodes(graph)
}

fn triangle_counts_decreasing_degree<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    TriangleCountScorer::new(MotifCountOrdering::DecreasingDegree).score_nodes(graph)
}

fn triangle_counts_increasing_degree<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree).score_nodes(graph)
}

fn square_counts_natural<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    SquareCountScorer::new(MotifCountOrdering::Natural).score_nodes(graph)
}

fn square_counts_decreasing_degree<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    SquareCountScorer::new(MotifCountOrdering::DecreasingDegree).score_nodes(graph)
}

fn square_counts_increasing_degree<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    SquareCountScorer::new(MotifCountOrdering::IncreasingDegree).score_nodes(graph)
}

fn triangle_scaling_cases() -> Vec<ScalingCase> {
    vec![
        ScalingCase { name: "path_32".to_string(), graph: wrap_undi(path_graph(32)) },
        ScalingCase { name: "path_64".to_string(), graph: wrap_undi(path_graph(64)) },
        ScalingCase { name: "path_128".to_string(), graph: wrap_undi(path_graph(128)) },
        ScalingCase { name: "path_256".to_string(), graph: wrap_undi(path_graph(256)) },
        ScalingCase {
            name: "sparse_gnp_32_p005".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_0032, 32, 0.05)),
        },
        ScalingCase {
            name: "sparse_gnp_64_p005".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_0064, 64, 0.05)),
        },
        ScalingCase {
            name: "sparse_gnp_128_p005".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_0128, 128, 0.05)),
        },
        ScalingCase {
            name: "sparse_gnp_256_p005".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_0256, 256, 0.05)),
        },
        ScalingCase {
            name: "dense_gnp_32_p020".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_1032, 32, 0.20)),
        },
        ScalingCase {
            name: "dense_gnp_64_p020".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_1064, 64, 0.20)),
        },
        ScalingCase {
            name: "dense_gnp_128_p020".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_1128, 128, 0.20)),
        },
        ScalingCase {
            name: "dense_gnp_256_p020".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5452_495f_1256, 256, 0.20)),
        },
        ScalingCase { name: "complete_16".to_string(), graph: wrap_undi(complete_graph(16)) },
        ScalingCase { name: "complete_32".to_string(), graph: wrap_undi(complete_graph(32)) },
        ScalingCase { name: "complete_64".to_string(), graph: wrap_undi(complete_graph(64)) },
    ]
}

fn square_scaling_cases() -> Vec<ScalingCase> {
    vec![
        ScalingCase { name: "path_16".to_string(), graph: wrap_undi(path_graph(16)) },
        ScalingCase { name: "path_32".to_string(), graph: wrap_undi(path_graph(32)) },
        ScalingCase { name: "path_64".to_string(), graph: wrap_undi(path_graph(64)) },
        ScalingCase { name: "path_128".to_string(), graph: wrap_undi(path_graph(128)) },
        ScalingCase { name: "grid_4x4".to_string(), graph: wrap_undi(grid_graph(4, 4)) },
        ScalingCase { name: "grid_8x8".to_string(), graph: wrap_undi(grid_graph(8, 8)) },
        ScalingCase { name: "grid_12x12".to_string(), graph: wrap_undi(grid_graph(12, 12)) },
        ScalingCase { name: "grid_16x16".to_string(), graph: wrap_undi(grid_graph(16, 16)) },
        ScalingCase {
            name: "sparse_gnp_16_p010".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_0016, 16, 0.10)),
        },
        ScalingCase {
            name: "sparse_gnp_32_p010".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_0032, 32, 0.10)),
        },
        ScalingCase {
            name: "sparse_gnp_64_p010".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_0064, 64, 0.10)),
        },
        ScalingCase {
            name: "sparse_gnp_128_p010".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_0128, 128, 0.10)),
        },
        ScalingCase {
            name: "dense_gnp_16_p025".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_1016, 16, 0.25)),
        },
        ScalingCase {
            name: "dense_gnp_32_p025".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_1032, 32, 0.25)),
        },
        ScalingCase {
            name: "dense_gnp_64_p025".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_1064, 64, 0.25)),
        },
        ScalingCase {
            name: "dense_gnp_128_p025".to_string(),
            graph: wrap_undi(erdos_renyi_gnp(0x5351_525f_1128, 128, 0.25)),
        },
        ScalingCase { name: "complete_8".to_string(), graph: wrap_undi(complete_graph(8)) },
        ScalingCase { name: "complete_12".to_string(), graph: wrap_undi(complete_graph(12)) },
        ScalingCase { name: "complete_16".to_string(), graph: wrap_undi(complete_graph(16)) },
        ScalingCase { name: "complete_24".to_string(), graph: wrap_undi(complete_graph(24)) },
    ]
}

fn assert_triangle_variants_agree(cases: &[ScalingCase]) {
    for case in cases {
        let natural = triangle_counts_natural(&case.graph);
        assert_eq!(
            natural,
            triangle_counts_decreasing_degree(&case.graph),
            "triangle variants disagree on {}",
            case.name
        );
        assert_eq!(
            natural,
            triangle_counts_increasing_degree(&case.graph),
            "triangle variants disagree on {}",
            case.name
        );
    }
}

fn assert_square_variants_agree(cases: &[ScalingCase]) {
    for case in cases {
        let natural = square_counts_natural(&case.graph);
        assert_eq!(
            natural,
            square_counts_decreasing_degree(&case.graph),
            "square variants disagree on {}",
            case.name
        );
        assert_eq!(
            natural,
            square_counts_increasing_degree(&case.graph),
            "square variants disagree on {}",
            case.name
        );
    }
}

fn bench_triangle_count_variants(c: &mut Criterion) {
    let cases = triangle_scaling_cases();
    assert_triangle_variants_agree(&cases);

    let mut group = c.benchmark_group("triangle_count_variants");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(800));

    for case in &cases {
        group.throughput(Throughput::Elements(case.graph.number_of_nodes() as u64));
        group.bench_with_input(BenchmarkId::new("natural", &case.name), case, |b, case| {
            b.iter(|| black_box(triangle_counts_natural(black_box(&case.graph))));
        });
        group.bench_with_input(
            BenchmarkId::new("decreasing_degree", &case.name),
            case,
            |b, case| {
                b.iter(|| black_box(triangle_counts_decreasing_degree(black_box(&case.graph))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("increasing_degree", &case.name),
            case,
            |b, case| {
                b.iter(|| black_box(triangle_counts_increasing_degree(black_box(&case.graph))));
            },
        );
    }

    group.finish();
}

fn bench_square_count_variants(c: &mut Criterion) {
    let cases = square_scaling_cases();
    assert_square_variants_agree(&cases);

    let mut group = c.benchmark_group("square_count_variants");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(800));

    for case in &cases {
        group.throughput(Throughput::Elements(case.graph.number_of_nodes() as u64));
        group.bench_with_input(BenchmarkId::new("natural", &case.name), case, |b, case| {
            b.iter(|| black_box(square_counts_natural(black_box(&case.graph))));
        });
        group.bench_with_input(
            BenchmarkId::new("decreasing_degree", &case.name),
            case,
            |b, case| {
                b.iter(|| black_box(square_counts_decreasing_degree(black_box(&case.graph))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("increasing_degree", &case.name),
            case,
            |b, case| {
                b.iter(|| black_box(square_counts_increasing_degree(black_box(&case.graph))));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_triangle_count_variants, bench_square_count_variants);
criterion_main!(benches);
