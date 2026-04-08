//! Criterion benchmarks for the exact minimum-cycle-basis trait.

#[path = "../tests/support/minimum_cycle_basis_fixture.rs"]
mod minimum_cycle_basis_fixture;

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            cycle_graph, erdos_renyi_gnp, friendship_graph, grid_graph, path_graph,
        },
    },
};
use minimum_cycle_basis_fixture::{
    MinimumCycleBasisFixtureCase, build_undigraph, load_fixture_suite,
};

const FIXTURE_NAME: &str = "minimum_cycle_basis_networkx_1000.json.gz";

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_cycle_rank: usize,
    expected_total_weight: usize,
    expected_basis: Option<Vec<Vec<usize>>>,
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

fn build_semantic_cases() -> Vec<FixtureBenchCase> {
    [
        (
            "tree_path_5",
            "semantic",
            build_undigraph_from_edges(5, &[[0, 1], [1, 2], [2, 3], [3, 4]]),
            0,
            0,
            Some(vec![]),
        ),
        (
            "triangle",
            "semantic",
            build_undigraph_from_edges(3, &[[0, 1], [1, 2], [0, 2]]),
            1,
            3,
            Some(vec![vec![0, 1, 2]]),
        ),
        (
            "square_with_diagonal",
            "semantic",
            build_undigraph_from_edges(4, &[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]),
            2,
            6,
            Some(vec![vec![0, 1, 2], vec![0, 2, 3]]),
        ),
        (
            "articulation_triangles",
            "semantic",
            build_undigraph_from_edges(5, &[[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [2, 4]]),
            2,
            6,
            Some(vec![vec![0, 1, 2], vec![2, 3, 4]]),
        ),
        (
            "cubane_graph",
            "semantic",
            build_undigraph_from_edges(
                8,
                &[
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ],
            ),
            5,
            20,
            None,
        ),
    ]
    .into_iter()
    .map(|(name, family, graph, expected_cycle_rank, expected_total_weight, expected_basis)| {
        FixtureBenchCase {
            name: name.to_string(),
            family: family.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
            expected_cycle_rank,
            expected_total_weight,
            expected_basis,
        }
    })
    .collect()
}

fn build_undigraph_from_edges(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let case = MinimumCycleBasisFixtureCase {
        name: "inline".to_string(),
        family: "inline".to_string(),
        node_count,
        edges: edges.to_vec(),
        cycle_rank: 0,
        basis_size: 0,
        total_weight: 0,
        minimum_cycle_basis: Vec::new(),
        notes: String::new(),
    };
    build_undigraph(&case)
}

fn prepare_fixture_cases(relative_path: &str) -> Vec<FixtureBenchCase> {
    let suite = load_fixture_suite(relative_path);
    suite
        .cases
        .into_iter()
        .map(|case| {
            FixtureBenchCase {
                logical_edge_count: case.edges.len(),
                graph: build_undigraph(&case),
                expected_cycle_rank: case.cycle_rank,
                expected_total_weight: case.total_weight,
                expected_basis: Some(normalize_cycles(case.minimum_cycle_basis)),
                name: case.name,
                family: case.family,
            }
        })
        .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_1024", wrap_undi(path_graph(1024))),
        ("cycle_512", wrap_undi(cycle_graph(512))),
        ("grid_16x16", wrap_undi(grid_graph(16, 16))),
        ("friendship_128", wrap_undi(friendship_graph(128))),
        ("erdos_renyi_256_p005", wrap_undi(erdos_renyi_gnp(0xD3A1_5EED_0123_4567, 256, 0.005))),
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

fn basis_checksum(result: &MinimumCycleBasisResult<usize>) -> u64 {
    result
        .minimum_cycle_basis()
        .fold(0u64, |checksum, cycle| {
            cycle.iter().fold(checksum.wrapping_mul(1_099_511_628_211), |acc, &node| {
                acc.wrapping_mul(257).wrapping_add(node as u64 + 1)
            })
        })
        .wrapping_add(result.cycle_rank() as u64)
        .wrapping_add((result.total_weight() as u64) << 32)
}

fn assert_cases_match_oracle(cases: &[FixtureBenchCase]) {
    for case in cases {
        let result = case.graph.minimum_cycle_basis().unwrap();
        let basis = normalize_cycles(result.minimum_cycle_basis().cloned().collect::<Vec<_>>());

        assert_eq!(
            result.cycle_rank(),
            case.expected_cycle_rank,
            "benchmark fixture case {} ({}) mismatched cycle rank",
            case.name,
            case.family
        );
        assert_eq!(
            result.total_weight(),
            case.expected_total_weight,
            "benchmark fixture case {} ({}) mismatched total weight",
            case.name,
            case.family
        );
        if let Some(expected_basis) = &case.expected_basis {
            assert_eq!(
                &basis, expected_basis,
                "benchmark fixture case {} ({}) mismatched exact basis",
                case.name, case.family
            );
        }
    }
}

fn bench_semantic_cases(c: &mut Criterion) {
    let cases = build_semantic_cases();
    assert_cases_match_oracle(&cases);

    let mut total_group = c.benchmark_group("minimum_cycle_basis_semantic_total");
    total_group.sample_size(10);
    total_group.warm_up_time(Duration::from_millis(400));
    total_group.measurement_time(Duration::from_secs(2));
    total_group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));
    total_group.bench_function(
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
                    .map(|case| basis_checksum(&case.graph.minimum_cycle_basis().unwrap()))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();
}

fn bench_reference_fixture_total(c: &mut Criterion) {
    let cases = prepare_fixture_cases(FIXTURE_NAME);
    assert_cases_match_oracle(&cases);

    let mut total_group = c.benchmark_group("minimum_cycle_basis_networkx_1000_total");
    total_group.sample_size(10);
    total_group.warm_up_time(Duration::from_millis(400));
    total_group.measurement_time(Duration::from_secs(3));
    total_group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));
    total_group.bench_function(
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
                    .map(|case| basis_checksum(&case.graph.minimum_cycle_basis().unwrap()))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();

    let mut case_group = c.benchmark_group("minimum_cycle_basis_scaling");
    case_group.sample_size(10);
    case_group.warm_up_time(Duration::from_millis(400));
    case_group.measurement_time(Duration::from_secs(2));

    for case in &cases {
        case_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        case_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = basis_checksum(&case.graph.minimum_cycle_basis().unwrap());
                black_box(checksum);
            });
        });
    }
    case_group.finish();
}

criterion_group!(benches, bench_semantic_cases, bench_reference_fixture_total, bench_scaling_cases);
criterion_main!(benches);
