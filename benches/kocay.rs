#![allow(clippy::pedantic)]
//! Criterion benchmarks for the Kocay balanced-flow solver on deterministic
//! capacitated instances.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{impls::ValuedCSR2D, prelude::*};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};

type CapacityGraph = ValuedCSR2D<usize, usize, usize, usize>;

#[derive(Clone)]
struct KocayCase {
    name: String,
    capacities: CapacityGraph,
    budgets: Vec<usize>,
}

fn build_capacity_graph(n: usize, edges: &[(usize, usize, usize)]) -> CapacityGraph {
    let mut directed_edges = Vec::with_capacity(edges.len() * 2);
    for &(u, v, capacity) in edges {
        if u == v || capacity == 0 {
            continue;
        }
        directed_edges.push((u, v, capacity));
        directed_edges.push((v, u, capacity));
    }
    directed_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));

    let mut graph: CapacityGraph =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), directed_edges.len());
    for edge in directed_edges {
        MatrixMut::add(&mut graph, edge).expect("benchmark graph edge insertion should succeed");
    }
    graph
}

fn make_kocay_case(
    name: &str,
    seed: u64,
    n: usize,
    edge_count: usize,
    max_capacity: usize,
) -> KocayCase {
    assert!(n >= 2);
    assert!(edge_count <= n * (n - 1) / 2);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
    for u in 0..n {
        for v in (u + 1)..n {
            pairs.push((u, v));
        }
    }
    pairs.shuffle(&mut rng);

    let mut incident_capacity = vec![0usize; n];
    let mut edges = Vec::with_capacity(edge_count);
    for &(u, v) in pairs.iter().take(edge_count) {
        let capacity = rng.gen_range(1..=max_capacity);
        incident_capacity[u] += capacity;
        incident_capacity[v] += capacity;
        edges.push((u, v, capacity));
    }

    let mut budgets = Vec::with_capacity(n);
    for &total_incident_capacity in &incident_capacity {
        if total_incident_capacity == 0 {
            budgets.push(0);
            continue;
        }
        let lower = (total_incident_capacity / 3).max(1);
        let upper = ((2 * total_incident_capacity) / 3).max(lower);
        budgets.push(rng.gen_range(lower..=upper));
    }

    KocayCase { name: name.to_owned(), capacities: build_capacity_graph(n, &edges), budgets }
}

fn benchmark_cases() -> Vec<KocayCase> {
    vec![
        make_kocay_case("n128_m512_cap4", 0x0A11_C001, 128, 512, 4),
        make_kocay_case("n256_m1536_cap4", 0x0A11_C002, 256, 1_536, 4),
        make_kocay_case("n256_m4096_cap3", 0x0A11_C003, 256, 4_096, 3),
        make_kocay_case("n512_m4096_cap3", 0x0A11_C004, 512, 4_096, 3),
    ]
}

fn bench_kocay(c: &mut Criterion) {
    let cases = benchmark_cases();

    let mut group = c.benchmark_group("kocay");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("Rust", &case.name), case, |b, case| {
            b.iter(|| {
                let flow = case.capacities.kocay(black_box(&case.budgets));
                black_box(flow);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_kocay);
criterion_main!(benches);
