//! Criterion benchmarks for the minimum-spanning-tree traits, timing `Kruskal`,
//! `Prim`, and `Boruvka` on the larger seeded `G(n, p)` cases from the shared
//! NetworkX corpus.

#[path = "../tests/support/fixture_io.rs"]
mod fixture_io;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::ValuedCSR2D,
    naive_structs::GenericEdgesBuilder,
    traits::{
        EdgesBuilder,
        algorithms::minimum_spanning_tree::{Boruvka, Kruskal, Prim},
    },
};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const FIXTURE_NAME: &str = "minimum_spanning_tree_networkx.json.gz";

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    node_count: usize,
    edges: Vec<(usize, usize, f64)>,
}

fn build_matrix(case: &Case) -> WeightedMatrix {
    let mut directed = Vec::with_capacity(case.edges.len() * 2);
    for &(source, destination, weight) in &case.edges {
        directed.push((source, destination, weight));
        if source != destination {
            directed.push((destination, source, weight));
        }
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((case.node_count, case.node_count))
        .edges(directed.into_iter())
        .build()
        .unwrap()
}

fn bench_minimum_spanning_tree(criterion: &mut Criterion) {
    let fixture: Fixture = load_fixture_json(FIXTURE_NAME);
    let mut group = criterion.benchmark_group("minimum_spanning_tree");

    for case in &fixture.cases {
        // Only the larger seeded graphs are worth timing.
        if case.node_count < 100 {
            continue;
        }
        let matrix = build_matrix(case);
        group.throughput(Throughput::Elements(case.edges.len() as u64));

        group.bench_with_input(BenchmarkId::new("kruskal", &case.name), &matrix, |bencher, m| {
            bencher.iter(|| black_box(m.minimum_spanning_tree_kruskal().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("prim", &case.name), &matrix, |bencher, m| {
            bencher.iter(|| black_box(m.minimum_spanning_tree_prim().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("boruvka", &case.name), &matrix, |bencher, m| {
            bencher.iter(|| black_box(m.minimum_spanning_tree_boruvka().unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_minimum_spanning_tree);
criterion_main!(benches);
