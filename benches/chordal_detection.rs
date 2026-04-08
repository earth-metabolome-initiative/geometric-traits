//! Criterion benchmarks for chordal graph detection.

#[path = "../tests/support/chordal_fixture.rs"]
mod chordal_fixture;

use std::{collections::BTreeMap, hint::black_box, time::Duration};

use chordal_fixture::{ChordalFixtureCase, build_undigraph, load_fixture_suite};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{complete_graph, cycle_graph, path_graph, star_graph},
    },
};

const FIXTURE_NAME: &str = "chordal_ground_truth.json.gz";

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    node_count: usize,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    is_chordal: bool,
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

fn split_graph(clique_size: usize, independent_size: usize) -> UndiGraph<usize> {
    let order = clique_size + independent_size;
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();

    let clique_edges =
        (0..clique_size).flat_map(|left| ((left + 1)..clique_size).map(move |right| (left, right)));
    let attachment_edges = (0..independent_size).flat_map(|offset| {
        (0..clique_size)
            .filter(move |&clique_node| (offset + clique_node) % 3 != 1)
            .map(move |clique_node| (clique_node, clique_size + offset))
    });
    let mut edges: Vec<(usize, usize)> = clique_edges.chain(attachment_edges).collect();
    edges.sort_unstable();

    let matrix: UndirectedAdjacency = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(order)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn complete_bipartite(left_partition: usize, right_partition: usize) -> UndiGraph<usize> {
    let order = left_partition + right_partition;
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    let edges = (0..left_partition)
        .flat_map(|left| (left_partition..order).map(move |right| (left, right)));
    let matrix: UndirectedAdjacency = UndiEdgesBuilder::default()
        .expected_number_of_edges(left_partition * right_partition)
        .expected_shape(order)
        .edges(edges)
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn prepare_fixture_cases(relative_path: &str) -> Vec<FixtureBenchCase> {
    let suite = load_fixture_suite(relative_path);
    prepare_cases(suite.cases)
}

fn prepare_cases(cases: Vec<ChordalFixtureCase>) -> Vec<FixtureBenchCase> {
    cases
        .into_iter()
        .map(|case| {
            let graph = build_undigraph(&case);
            let logical_edge_count = case.edges.len();
            FixtureBenchCase {
                name: case.name,
                family: case.family,
                node_count: case.node_count,
                logical_edge_count,
                graph,
                is_chordal: case.is_chordal,
            }
        })
        .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4096))),
        ("star_4096", wrap_undi(star_graph(4096))),
        ("cycle_4096", wrap_undi(cycle_graph(4096))),
        ("complete_512", wrap_undi(complete_graph(512))),
        ("split_384_640", split_graph(384, 640)),
        ("complete_bipartite_512_512", complete_bipartite(512, 512)),
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
            case.graph.is_chordal(),
            case.is_chordal,
            "benchmark fixture case {} ({}) mismatched oracle",
            case.name,
            case.family
        );
    }
}

fn total_nodes(cases: &[&FixtureBenchCase]) -> usize {
    cases.iter().map(|case| case.node_count).sum()
}

fn chordal_checksum(cases: &[&FixtureBenchCase]) -> usize {
    cases.iter().filter(|case| case.graph.is_chordal()).count()
}

fn bench_fixture_total(c: &mut Criterion) {
    let cases = prepare_fixture_cases(FIXTURE_NAME);
    assert_fixture_cases_match_oracle(&cases);

    let case_refs: Vec<&FixtureBenchCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("chordal_detection_fixture_total");
    total_group.sample_size(10);
    total_group.warm_up_time(Duration::from_millis(500));
    total_group.measurement_time(Duration::from_secs(3));
    total_group.throughput(Throughput::Elements(
        u64::try_from(case_refs.len()).expect("fixture case count should fit into u64"),
    ));
    total_group.bench_function(
        BenchmarkId::new(
            "total_cases",
            format!(
                "cases={}_nodes={}_logical_edges={}",
                case_refs.len(),
                total_nodes(&case_refs),
                case_refs.iter().map(|case| case.logical_edge_count).sum::<usize>()
            ),
        ),
        {
            let case_refs = case_refs.clone();
            move |b| {
                b.iter(|| {
                    let total = chordal_checksum(&case_refs);
                    black_box(total);
                });
            }
        },
    );
    total_group.finish();
}

fn bench_fixture_by_family(c: &mut Criterion) {
    let cases = prepare_fixture_cases(FIXTURE_NAME);
    assert_fixture_cases_match_oracle(&cases);

    let mut families: BTreeMap<String, Vec<&FixtureBenchCase>> = BTreeMap::new();
    for case in &cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group = c.benchmark_group("chordal_detection_fixture_by_family");
    family_group.sample_size(10);
    family_group.warm_up_time(Duration::from_millis(500));
    family_group.measurement_time(Duration::from_secs(2));

    for (family, family_cases) in families {
        family_group.throughput(Throughput::Elements(
            u64::try_from(family_cases.len()).expect("family case count should fit into u64"),
        ));
        family_group.bench_function(
            BenchmarkId::new(
                "family_cases",
                format!(
                    "{family}_cases={}_nodes={}_logical_edges={}",
                    family_cases.len(),
                    family_cases.iter().map(|case| case.node_count).sum::<usize>(),
                    family_cases.iter().map(|case| case.logical_edge_count).sum::<usize>()
                ),
            ),
            {
                let family_cases = family_cases.clone();
                move |b| {
                    b.iter(|| {
                        let total =
                            family_cases.iter().filter(|case| case.graph.is_chordal()).count();
                        black_box(total);
                    });
                }
            },
        );
    }
    family_group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();

    let mut case_group = c.benchmark_group("chordal_detection_scaling");
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
                    let is_chordal = graph.is_chordal();
                    black_box(is_chordal);
                });
            }
        });
    }
    case_group.finish();
}

criterion_group!(benches, bench_fixture_total, bench_fixture_by_family, bench_scaling_cases);
criterion_main!(benches);
