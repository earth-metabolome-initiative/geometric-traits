//! Criterion benchmarks for Hopcroft-Tarjan biconnected components.

#[path = "../tests/support/biconnected_fixture.rs"]
mod biconnected_fixture;

use std::hint::black_box;

use biconnected_fixture::{build_undigraph, load_fixture_suite, semantic_cases};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            barbell_graph, cycle_graph, erdos_renyi_gnp, friendship_graph, grid_graph, path_graph,
        },
    },
};

const EXHAUSTIVE_FIXTURE_NAME: &str = "biconnected_components_order5_exhaustive.json.gz";

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    family: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
    expected_edge_components: Vec<Vec<[usize; 2]>>,
    expected_vertex_components: Vec<Vec<usize>>,
    expected_articulation_points: Vec<usize>,
    expected_bridges: Vec<[usize; 2]>,
    expected_omitted_vertices: Vec<usize>,
    expected_cyclic_component_ids: Vec<usize>,
    expected_connected_component_count: usize,
    expected_is_biconnected: bool,
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

fn prepare_fixture_cases(relative_path: &str) -> Vec<FixtureBenchCase> {
    let suite = load_fixture_suite(relative_path);
    prepare_cases(suite.cases)
}

fn prepare_cases(cases: Vec<biconnected_fixture::BiconnectedFixtureCase>) -> Vec<FixtureBenchCase> {
    cases
        .into_iter()
        .map(|case| {
            let graph = build_undigraph(&case);
            let logical_edge_count = case.edges.len();
            let expected_connected_component_count = case.connected_components.len();
            FixtureBenchCase {
                name: case.name,
                family: case.family,
                logical_edge_count,
                graph,
                expected_edge_components: case.edge_biconnected_components,
                expected_vertex_components: case.vertex_biconnected_components,
                expected_articulation_points: case.articulation_points,
                expected_bridges: case.bridges,
                expected_omitted_vertices: case.vertices_without_biconnected_component,
                expected_cyclic_component_ids: case.cyclic_biconnected_component_indices,
                expected_connected_component_count,
                expected_is_biconnected: case.is_biconnected,
            }
        })
        .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_1024", wrap_undi(path_graph(1024))),
        ("cycle_1024", wrap_undi(cycle_graph(1024))),
        ("grid_32x32", wrap_undi(grid_graph(32, 32))),
        ("friendship_256", wrap_undi(friendship_graph(256))),
        ("barbell_64_256", wrap_undi(barbell_graph(64, 256))),
        ("erdos_renyi_1024_p001", wrap_undi(erdos_renyi_gnp(0xB1C0_5CC0_1DB1_C0DE, 1024, 0.01))),
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
        let decomposition = case.graph.biconnected_components().unwrap();
        let edge_components: Vec<Vec<[usize; 2]>> =
            decomposition.edge_biconnected_components().cloned().collect();
        let vertex_components: Vec<Vec<usize>> =
            decomposition.vertex_biconnected_components().cloned().collect();
        let articulation_points: Vec<usize> = decomposition.articulation_points().collect();
        let bridges: Vec<[usize; 2]> = decomposition.bridges().collect();
        let omitted_vertices: Vec<usize> =
            decomposition.vertices_without_biconnected_component().collect();
        let cyclic_component_ids: Vec<usize> =
            decomposition.cyclic_biconnected_component_ids().collect();

        assert_eq!(
            edge_components, case.expected_edge_components,
            "benchmark fixture case {} ({}) mismatched edge oracle",
            case.name, case.family
        );
        assert_eq!(
            vertex_components, case.expected_vertex_components,
            "benchmark fixture case {} ({}) mismatched vertex oracle",
            case.name, case.family
        );
        assert_eq!(
            articulation_points, case.expected_articulation_points,
            "benchmark fixture case {} ({}) mismatched articulation-point oracle",
            case.name, case.family
        );
        assert_eq!(
            bridges, case.expected_bridges,
            "benchmark fixture case {} ({}) mismatched bridge oracle",
            case.name, case.family
        );
        assert_eq!(
            omitted_vertices, case.expected_omitted_vertices,
            "benchmark fixture case {} ({}) mismatched omitted-vertex oracle",
            case.name, case.family
        );
        assert_eq!(
            cyclic_component_ids, case.expected_cyclic_component_ids,
            "benchmark fixture case {} ({}) mismatched cyclic-component oracle",
            case.name, case.family
        );
        assert_eq!(
            decomposition.number_of_connected_components(),
            case.expected_connected_component_count,
            "benchmark fixture case {} ({}) mismatched connected-component count",
            case.name,
            case.family
        );
        assert_eq!(
            decomposition.is_biconnected(),
            case.expected_is_biconnected,
            "benchmark fixture case {} ({}) mismatched graph-level flag",
            case.name,
            case.family
        );
    }
}

fn decomposition_checksum(result: &BiconnectedComponentsResult<usize>) -> u64 {
    let edge_checksum = result
        .edge_biconnected_components()
        .flat_map(|component| component.iter().copied())
        .fold(0u64, |checksum, [left, right]| {
            checksum
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(left as u64)
                .wrapping_add((right as u64) << 16)
        });
    let articulation_checksum = result
        .articulation_points()
        .fold(0u64, |checksum, vertex| checksum.wrapping_mul(257).wrapping_add(vertex as u64));
    let bridge_checksum = result.bridges().fold(0u64, |checksum, [left, right]| {
        checksum.wrapping_mul(131).wrapping_add(left as u64 + ((right as u64) << 8))
    });
    let cyclic_checksum =
        result.cyclic_biconnected_component_ids().fold(0u64, |checksum, component_id| {
            checksum.wrapping_mul(17).wrapping_add(component_id as u64)
        });

    edge_checksum
        .wrapping_add(articulation_checksum)
        .wrapping_add(bridge_checksum)
        .wrapping_add(cyclic_checksum)
        .wrapping_add(result.number_of_biconnected_components() as u64)
        .wrapping_add((result.number_of_connected_components() as u64) << 32)
        .wrapping_add(u64::from(result.is_biconnected()) << 63)
}

fn bench_semantic_cases(c: &mut Criterion) {
    let cases = prepare_cases(semantic_cases());
    assert_fixture_cases_match_oracle(&cases);

    let mut total_group = c.benchmark_group("biconnected_components_semantic_total");
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
                    .map(|case| {
                        decomposition_checksum(&case.graph.biconnected_components().unwrap())
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut case_group = c.benchmark_group("biconnected_components_semantic_by_case");
    for case in &cases {
        case_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        case_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum =
                    decomposition_checksum(&case.graph.biconnected_components().unwrap());
                black_box(checksum);
            });
        });
    }
    case_group.finish();
}

fn bench_exhaustive_fixture_total(c: &mut Criterion) {
    let cases = prepare_fixture_cases(EXHAUSTIVE_FIXTURE_NAME);
    assert_fixture_cases_match_oracle(&cases);

    let mut total_group = c.benchmark_group("biconnected_components_exhaustive_total");
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
                    .map(|case| {
                        decomposition_checksum(&case.graph.biconnected_components().unwrap())
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();

    let mut case_group = c.benchmark_group("biconnected_components_scaling");
    for case in &cases {
        case_group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        case_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum =
                    decomposition_checksum(&case.graph.biconnected_components().unwrap());
                black_box(checksum);
            });
        });
    }
    case_group.finish();
}

criterion_group!(
    benches,
    bench_semantic_cases,
    bench_exhaustive_fixture_total,
    bench_scaling_cases
);
criterion_main!(benches);
