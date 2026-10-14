//! Criterion benchmarks for exact undirected diameter computation.

use std::{collections::VecDeque, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        Diameter, DiameterError, MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{
            barabasi_albert, barbell_graph, cycle_graph, grid_graph, path_graph,
        },
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type UndirectedGraph = UndiGraph<usize>;

#[derive(Clone)]
struct BenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndirectedGraph,
}

fn wrap_undi(graph: UndirectedAdjacency) -> UndirectedGraph {
    let order = graph.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, graph))
}

fn logical_edge_count(graph: &UndirectedGraph) -> usize {
    graph.sparse_coordinates().filter(|&(source, destination)| source <= destination).count()
}

fn prepare_small_exact_cases() -> Vec<BenchCase> {
    [
        ("path_512", wrap_undi(path_graph(512))),
        ("cycle_512", wrap_undi(cycle_graph(512))),
        ("grid_20x20", wrap_undi(grid_graph(20, 20))),
        ("barbell_32_128", wrap_undi(barbell_graph(32, 128))),
        ("barabasi_albert_512_m4", wrap_undi(barabasi_albert(0xD1A0_AA51_0512_0004, 512, 4))),
    ]
    .into_iter()
    .map(|(name, graph)| {
        BenchCase { name: name.to_string(), logical_edge_count: logical_edge_count(&graph), graph }
    })
    .collect()
}

fn prepare_scaling_cases() -> Vec<BenchCase> {
    [
        ("path_4096", wrap_undi(path_graph(4_096))),
        ("cycle_4096", wrap_undi(cycle_graph(4_096))),
        ("grid_64x64", wrap_undi(grid_graph(64, 64))),
        ("barbell_128_1024", wrap_undi(barbell_graph(128, 1_024))),
        ("barabasi_albert_4096_m4", wrap_undi(barabasi_albert(0xD1A0_AA51_4096_0004, 4_096, 4))),
    ]
    .into_iter()
    .map(|(name, graph)| {
        BenchCase { name: name.to_string(), logical_edge_count: logical_edge_count(&graph), graph }
    })
    .collect()
}

fn normalize_crate_diameter(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    graph.diameter().map_err(|error| {
        match error {
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::DiameterError(error)) => {
                error
            }
            other => panic!("unexpected diameter error shape: {other}"),
        }
    })
}

fn bfs_profile(
    graph: &UndirectedGraph,
    source: usize,
    distances: &mut [usize],
    queue: &mut VecDeque<usize>,
) -> Result<(usize, usize), DiameterError> {
    let order = graph.number_of_nodes();
    distances.fill(usize::MAX);
    queue.clear();
    distances[source] = 0;
    queue.push_back(source);

    let mut visited = 1;
    let mut eccentricity = 0;
    let mut farthest = source;

    while let Some(node) = queue.pop_front() {
        let node_distance = distances[node];
        if node_distance > eccentricity || (node_distance == eccentricity && node > farthest) {
            eccentricity = node_distance;
            farthest = node;
        }

        for neighbor in graph.neighbors(node) {
            if distances[neighbor] != usize::MAX {
                continue;
            }

            distances[neighbor] = node_distance + 1;
            visited += 1;
            queue.push_back(neighbor);
        }
    }

    if visited != order {
        return Err(DiameterError::DisconnectedGraph);
    }

    Ok((eccentricity, farthest))
}

fn all_pairs_bfs_diameter(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    let order = graph.number_of_nodes();
    if order <= 1 {
        return Ok(0);
    }

    let mut distances = vec![usize::MAX; order];
    let mut queue = VecDeque::with_capacity(order);
    let mut diameter = 0;

    for source in graph.node_ids() {
        diameter = diameter.max(bfs_profile(graph, source, &mut distances, &mut queue)?.0);
    }

    Ok(diameter)
}

fn ifub_degree_start_diameter(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    let order = graph.number_of_nodes();
    if order <= 1 {
        return Ok(0);
    }

    let start = graph
        .node_ids()
        .max_by_key(|&node| (graph.degree(node), node))
        .expect("non-empty graph must contain at least one node");

    let mut distances = vec![usize::MAX; order];
    let mut queue = VecDeque::with_capacity(order);

    let start_eccentricity = bfs_profile(graph, start, &mut distances, &mut queue)?.0;
    let mut fringe_levels = vec![Vec::new(); start_eccentricity + 1];
    for node in graph.node_ids() {
        fringe_levels[distances[node]].push(node);
    }

    let mut lower_bound = start_eccentricity;
    let mut upper_bound = start_eccentricity.saturating_mul(2);
    let mut level = start_eccentricity;

    while upper_bound > lower_bound && level > 0 {
        let mut fringe_eccentricity = 0;
        for &node in &fringe_levels[level] {
            fringe_eccentricity =
                fringe_eccentricity.max(bfs_profile(graph, node, &mut distances, &mut queue)?.0);
        }

        lower_bound = lower_bound.max(fringe_eccentricity);

        let next_upper_bound = 2 * (level - 1);
        if lower_bound > next_upper_bound {
            return Ok(lower_bound);
        }

        upper_bound = next_upper_bound;
        level -= 1;
    }

    Ok(lower_bound)
}

fn middle_of_bfs_path(source: usize, target: usize, parents: &[Option<usize>]) -> usize {
    let mut path = Vec::new();
    let mut current = Some(target);

    while let Some(node) = current {
        path.push(node);
        if node == source {
            break;
        }
        current = parents[node];
    }

    debug_assert_eq!(path.last().copied(), Some(source));
    path.reverse();
    path[path.len() / 2]
}

fn ifub_4sweep_diameter(graph: &UndirectedGraph) -> Result<usize, DiameterError> {
    let order = graph.number_of_nodes();
    if order <= 1 {
        return Ok(0);
    }

    let start = graph
        .node_ids()
        .max_by_key(|&node| (graph.degree(node), node))
        .expect("non-empty graph must contain at least one node");

    let mut distances = vec![usize::MAX; order];
    let mut parents = vec![None; order];
    let mut queue = VecDeque::with_capacity(order);

    let (_, a1) = bfs_profile_with_parents(graph, start, &mut distances, &mut parents, &mut queue)?;
    let (ecc_a1, b1) =
        bfs_profile_with_parents(graph, a1, &mut distances, &mut parents, &mut queue)?;
    let r2 = middle_of_bfs_path(a1, b1, &parents);

    let (_, a2) = bfs_profile_with_parents(graph, r2, &mut distances, &mut parents, &mut queue)?;
    let (ecc_a2, b2) =
        bfs_profile_with_parents(graph, a2, &mut distances, &mut parents, &mut queue)?;
    let ifub_start = middle_of_bfs_path(a2, b2, &parents);

    let start_eccentricity =
        bfs_profile_with_parents(graph, ifub_start, &mut distances, &mut parents, &mut queue)?.0;

    let mut fringe_levels = vec![Vec::new(); start_eccentricity + 1];
    for node in graph.node_ids() {
        fringe_levels[distances[node]].push(node);
    }

    let mut lower_bound = ecc_a1.max(ecc_a2).max(start_eccentricity);
    let mut upper_bound = start_eccentricity.saturating_mul(2);
    let mut level = start_eccentricity;

    while upper_bound > lower_bound && level > 0 {
        let mut fringe_eccentricity = 0;
        for &node in &fringe_levels[level] {
            fringe_eccentricity = fringe_eccentricity.max(
                bfs_profile_with_parents(graph, node, &mut distances, &mut parents, &mut queue)?.0,
            );
        }

        lower_bound = lower_bound.max(fringe_eccentricity);

        let next_upper_bound = 2 * (level - 1);
        if lower_bound > next_upper_bound {
            return Ok(lower_bound);
        }

        upper_bound = next_upper_bound;
        level -= 1;
    }

    Ok(lower_bound)
}

fn bfs_profile_with_parents(
    graph: &UndirectedGraph,
    source: usize,
    distances: &mut [usize],
    parents: &mut [Option<usize>],
    queue: &mut VecDeque<usize>,
) -> Result<(usize, usize), DiameterError> {
    let order = graph.number_of_nodes();
    distances.fill(usize::MAX);
    parents.fill(None);
    queue.clear();
    distances[source] = 0;
    queue.push_back(source);

    let mut visited = 1;
    let mut eccentricity = 0;
    let mut farthest = source;

    while let Some(node) = queue.pop_front() {
        let node_distance = distances[node];
        if node_distance > eccentricity || (node_distance == eccentricity && node > farthest) {
            eccentricity = node_distance;
            farthest = node;
        }

        for neighbor in graph.neighbors(node) {
            if distances[neighbor] != usize::MAX {
                continue;
            }

            distances[neighbor] = node_distance + 1;
            parents[neighbor] = Some(node);
            visited += 1;
            queue.push_back(neighbor);
        }
    }

    if visited != order {
        return Err(DiameterError::DisconnectedGraph);
    }

    Ok((eccentricity, farthest))
}

fn assert_small_cases_match_exact_baselines(cases: &[BenchCase]) {
    for case in cases {
        let crate_diameter = normalize_crate_diameter(&case.graph).unwrap();
        let degree_start = ifub_degree_start_diameter(&case.graph).unwrap();
        let four_sweep = ifub_4sweep_diameter(&case.graph).unwrap();
        let all_pairs = all_pairs_bfs_diameter(&case.graph).unwrap();

        assert_eq!(
            crate_diameter, all_pairs,
            "small benchmark case {} mismatched all-pairs BFS oracle",
            case.name
        );
        assert_eq!(
            degree_start, all_pairs,
            "small benchmark case {} mismatched degree-start iFUB oracle",
            case.name
        );
        assert_eq!(
            four_sweep, all_pairs,
            "small benchmark case {} mismatched 4-sweep iFUB oracle",
            case.name
        );
    }
}

fn assert_scaling_cases_match_degree_start(cases: &[BenchCase]) {
    for case in cases {
        let crate_diameter = normalize_crate_diameter(&case.graph).unwrap();
        let degree_start = ifub_degree_start_diameter(&case.graph).unwrap();
        let four_sweep = ifub_4sweep_diameter(&case.graph).unwrap();

        assert_eq!(
            crate_diameter, degree_start,
            "scaling benchmark case {} mismatched degree-start iFUB result",
            case.name
        );
        assert_eq!(
            crate_diameter, four_sweep,
            "scaling benchmark case {} mismatched 4-sweep iFUB result",
            case.name
        );
    }
}

fn bench_small_exact_cases(c: &mut Criterion) {
    let cases = prepare_small_exact_cases();
    assert_small_cases_match_exact_baselines(&cases);

    let mut group = c.benchmark_group("diameter_small_exact");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for case in &cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));

        group.bench_function(BenchmarkId::new("all_pairs_bfs", &case.name), |b| {
            b.iter(|| black_box(all_pairs_bfs_diameter(&case.graph).unwrap()));
        });

        group.bench_function(BenchmarkId::new("ifub_degree_start", &case.name), |b| {
            b.iter(|| black_box(ifub_degree_start_diameter(&case.graph).unwrap()));
        });

        group.bench_function(BenchmarkId::new("ifub_4sweep", &case.name), |b| {
            b.iter(|| black_box(ifub_4sweep_diameter(&case.graph).unwrap()));
        });

        group.bench_function(BenchmarkId::new("crate_ifub_adaptive", &case.name), |b| {
            b.iter(|| black_box(normalize_crate_diameter(&case.graph).unwrap()));
        });
    }

    group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    assert_scaling_cases_match_degree_start(&cases);

    let mut group = c.benchmark_group("diameter_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for case in &cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));

        group.bench_function(BenchmarkId::new("ifub_degree_start", &case.name), |b| {
            b.iter(|| black_box(ifub_degree_start_diameter(&case.graph).unwrap()));
        });

        group.bench_function(BenchmarkId::new("ifub_4sweep", &case.name), |b| {
            b.iter(|| black_box(ifub_4sweep_diameter(&case.graph).unwrap()));
        });

        group.bench_function(BenchmarkId::new("crate_ifub_adaptive", &case.name), |b| {
            b.iter(|| black_box(normalize_crate_diameter(&case.graph).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_small_exact_cases, bench_scaling_cases);
criterion_main!(benches);
