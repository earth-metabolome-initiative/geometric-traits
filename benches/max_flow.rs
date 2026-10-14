//! Criterion benchmarks for the maximum s-t flow algorithms `Dinic` and
//! `EdmondsKarp`.
//!
//! Three comparisons are provided:
//!
//! * `max_flow_vs_networkx`: each fixture case is timed for both Rust
//!   algorithms, with the NetworkX `dinitz` and `edmonds_karp` per-call
//!   reference times carried in the benchmark ids. This is only a correctness
//!   floor: a compiled Rust solver must beat an interpreted one.
//! * `general_max_flow`: the meaningful same-language comparison. Dinic (`O(V^2
//!   E)`) versus Edmonds-Karp (`O(V E^2)`) on identical directed capacity
//!   graphs, with the two checked to agree on the value before timing.
//! * `bipartite_matching`: both general max-flow algorithms (on the
//!   unit-capacity reduction) against the specialized `HopcroftKarp` matcher on
//!   the same bipartite instances.

#[path = "../tests/support/fixture_io.rs"]
mod fixture_io;

use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::{CSR2D, ValuedCSR2D},
    prelude::*,
};
use serde::Deserialize;

type CapacityMatrix = ValuedCSR2D<usize, usize, usize, u64>;
type BipartiteMatrix = CSR2D<usize, usize, usize>;

const FIXTURE_NAME: &str = "max_flow_ground_truth.json.gz";

fn build_capacity_matrix(n: usize, mut edges: Vec<(usize, usize, u64)>) -> CapacityMatrix {
    edges.sort_unstable();
    edges.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    GenericEdgesBuilder::<_, CapacityMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn xorshift(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

// ── Both algorithms versus the NetworkX reference (a correctness floor) ──────

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    node_count: usize,
    edges: Vec<(usize, usize, u64)>,
    source: usize,
    sink: usize,
    max_flow: u64,
    reference: Reference,
}

#[derive(Debug, Deserialize)]
struct Reference {
    dinitz_nanos: u64,
    edmonds_karp_nanos: u64,
}

struct ReferenceCase {
    graph: CapacityMatrix,
    source: usize,
    sink: usize,
    edge_count: u64,
    dinic_label: String,
    edmonds_karp_label: String,
}

fn prepare_reference_cases() -> Vec<ReferenceCase> {
    let fixture: Fixture = load_fixture_json(FIXTURE_NAME);
    let mut prepared = Vec::new();

    for case in &fixture.cases {
        let graph = build_capacity_matrix(case.node_count, case.edges.clone());

        // Validate both algorithms against the reference before timing.
        let dinic_value = graph.dinic(case.source, case.sink).unwrap().max_flow();
        let edmonds_karp_value = graph.edmonds_karp(case.source, case.sink).unwrap().max_flow();
        assert_eq!(dinic_value, case.max_flow, "case `{}`: dinic value", case.id);
        assert_eq!(edmonds_karp_value, case.max_flow, "case `{}`: edmonds_karp value", case.id);

        prepared.push(ReferenceCase {
            graph,
            source: case.source,
            sink: case.sink,
            edge_count: u64::try_from(case.edges.len()).unwrap(),
            dinic_label: format!("{}_ref={}ns", case.id, case.reference.dinitz_nanos),
            edmonds_karp_label: format!("{}_ref={}ns", case.id, case.reference.edmonds_karp_nanos),
        });
    }

    prepared
}

fn bench_reference(criterion: &mut Criterion) {
    let cases = prepare_reference_cases();

    let mut group = criterion.benchmark_group("max_flow_vs_networkx");
    for case in &cases {
        group.throughput(Throughput::Elements(case.edge_count));
        group.bench_with_input(
            BenchmarkId::new("dinic", &case.dinic_label),
            case,
            |bencher, case| {
                bencher.iter(|| {
                    black_box(
                        black_box(&case.graph).dinic(case.source, case.sink).unwrap().max_flow(),
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("edmonds_karp", &case.edmonds_karp_label),
            case,
            |bencher, case| {
                bencher.iter(|| {
                    black_box(
                        black_box(&case.graph)
                            .edmonds_karp(case.source, case.sink)
                            .unwrap()
                            .max_flow(),
                    );
                });
            },
        );
    }
    group.finish();
}

// ── Dinic versus Edmonds-Karp on general directed capacity graphs ────────────

/// Deterministic directed capacity graph with roughly `target_edges` unique
/// arcs and capacities in `1..=max_capacity`.
fn random_digraph(
    n: usize,
    target_edges: usize,
    max_capacity: u64,
    seed: u64,
) -> Vec<(usize, usize, u64)> {
    let n_u64 = u64::try_from(n).unwrap();
    let mut state = seed | 1;
    let mut edges: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut attempts = 0;
    while edges.len() < target_edges && attempts < target_edges * 20 {
        attempts += 1;
        let u = usize::try_from(xorshift(&mut state) % n_u64).unwrap();
        let v = usize::try_from(xorshift(&mut state) % n_u64).unwrap();
        if u == v {
            continue;
        }
        let capacity = 1 + xorshift(&mut state) % max_capacity;
        edges.insert((u, v), capacity);
    }
    edges.into_iter().map(|((u, v), c)| (u, v, c)).collect()
}

struct GeneralCase {
    graph: CapacityMatrix,
    source: usize,
    sink: usize,
    edge_count: u64,
    label: String,
}

fn prepare_general_cases() -> Vec<GeneralCase> {
    // (nodes, target arcs, max capacity, seed, name)
    let configs = [
        (60usize, 400usize, 20u64, 11u64, "n60_dense"),
        (120, 1200, 20, 12, "n120_dense"),
        (200, 3000, 50, 13, "n200_dense"),
        (200, 3000, 1, 14, "n200_unit"),
    ];
    let mut prepared = Vec::new();

    for (n, target_edges, max_capacity, seed, name) in configs {
        let edges = random_digraph(n, target_edges, max_capacity, seed);
        let graph = build_capacity_matrix(n, edges.clone());
        let source = 0;
        let sink = n - 1;

        // The two algorithms must agree on the value before timing.
        let dinic_value = graph.dinic(source, sink).unwrap().max_flow();
        let edmonds_karp_value = graph.edmonds_karp(source, sink).unwrap().max_flow();
        assert_eq!(dinic_value, edmonds_karp_value, "Dinic and Edmonds-Karp disagree on `{name}`");

        prepared.push(GeneralCase {
            graph,
            source,
            sink,
            edge_count: u64::try_from(edges.len()).unwrap(),
            label: format!("{name}_f={dinic_value}"),
        });
    }

    prepared
}

fn bench_general(criterion: &mut Criterion) {
    let cases = prepare_general_cases();

    let mut group = criterion.benchmark_group("general_max_flow");
    for case in &cases {
        group.throughput(Throughput::Elements(case.edge_count));
        group.bench_with_input(BenchmarkId::new("dinic", &case.label), case, |bencher, case| {
            bencher.iter(|| {
                black_box(black_box(&case.graph).dinic(case.source, case.sink).unwrap().max_flow());
            });
        });
        group.bench_with_input(
            BenchmarkId::new("edmonds_karp", &case.label),
            case,
            |bencher, case| {
                bencher.iter(|| {
                    black_box(
                        black_box(&case.graph)
                            .edmonds_karp(case.source, case.sink)
                            .unwrap()
                            .max_flow(),
                    );
                });
            },
        );
    }
    group.finish();
}

// ── Both max-flow algorithms versus the specialized Hopcroft-Karp matcher ────

fn random_bipartite(
    left: usize,
    right: usize,
    target_edges: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let left_u64 = u64::try_from(left).unwrap();
    let right_u64 = u64::try_from(right).unwrap();
    let mut state = seed | 1;
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut attempts = 0;
    while edges.len() < target_edges && attempts < target_edges * 20 {
        attempts += 1;
        let l = usize::try_from(xorshift(&mut state) % left_u64).unwrap();
        let r = usize::try_from(xorshift(&mut state) % right_u64).unwrap();
        edges.insert((l, r));
    }
    edges.into_iter().collect()
}

struct BipartiteCase {
    hopcroft: BipartiteMatrix,
    reduction: CapacityMatrix,
    source: usize,
    sink: usize,
    edge_count: u64,
    label: String,
}

fn prepare_bipartite_cases() -> Vec<BipartiteCase> {
    let configs = [(40usize, 40usize, 200usize, 1u64), (100, 100, 800, 2), (200, 200, 2000, 3)];
    let mut prepared = Vec::new();

    for (left, right, target_edges, seed) in configs {
        let edges = random_bipartite(left, right, target_edges, seed);
        let hopcroft: BipartiteMatrix = GenericEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((left, right))
            .edges(edges.iter().copied())
            .build()
            .unwrap();

        let source = 0;
        let sink = left + right + 1;
        let n = left + right + 2;
        let mut flow_edges: Vec<(usize, usize, u64)> = Vec::new();
        for l in 0..left {
            flow_edges.push((source, 1 + l, 1));
        }
        for r in 0..right {
            flow_edges.push((1 + left + r, sink, 1));
        }
        for &(l, r) in &edges {
            flow_edges.push((1 + l, 1 + left + r, 1));
        }
        let reduction = build_capacity_matrix(n, flow_edges);

        // All three solvers must agree on the matching size before timing.
        let dinic_matching = reduction.dinic(source, sink).unwrap().max_flow();
        let edmonds_karp_matching = reduction.edmonds_karp(source, sink).unwrap().max_flow();
        let hopcroft_matching = u64::try_from(hopcroft.hopcroft_karp().unwrap().len()).unwrap();
        assert_eq!(dinic_matching, hopcroft_matching, "Dinic vs Hopcroft-Karp ({left}x{right})");
        assert_eq!(
            edmonds_karp_matching, hopcroft_matching,
            "Edmonds-Karp vs Hopcroft-Karp ({left}x{right})"
        );

        prepared.push(BipartiteCase {
            hopcroft,
            reduction,
            source,
            sink,
            edge_count: u64::try_from(edges.len()).unwrap(),
            label: format!("{left}x{right}_m={hopcroft_matching}"),
        });
    }

    prepared
}

fn bench_bipartite(criterion: &mut Criterion) {
    let cases = prepare_bipartite_cases();

    let mut group = criterion.benchmark_group("bipartite_matching");
    for case in &cases {
        group.throughput(Throughput::Elements(case.edge_count));
        group.bench_with_input(BenchmarkId::new("dinic", &case.label), case, |bencher, case| {
            bencher.iter(|| {
                black_box(
                    black_box(&case.reduction).dinic(case.source, case.sink).unwrap().max_flow(),
                );
            });
        });
        group.bench_with_input(
            BenchmarkId::new("edmonds_karp", &case.label),
            case,
            |bencher, case| {
                bencher.iter(|| {
                    black_box(
                        black_box(&case.reduction)
                            .edmonds_karp(case.source, case.sink)
                            .unwrap()
                            .max_flow(),
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("hopcroft_karp", &case.label),
            case,
            |bencher, case| {
                bencher.iter(|| {
                    black_box(black_box(&case.hopcroft).hopcroft_karp().unwrap().len());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_reference, bench_general, bench_bipartite);
criterion_main!(benches);
