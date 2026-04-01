#![allow(clippy::pedantic)]

//! Criterion benchmark comparing the Rust Blossom V implementation against the
//! same deterministic weighted instances.

use std::{collections::BTreeSet, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{impls::ValuedCSR2D, prelude::*};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};

type Vcsr = ValuedCSR2D<usize, usize, usize, i32>;

#[derive(Clone)]
struct WeightedCase {
    name: String,
    graph: Vcsr,
}

fn build_valued_graph(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
    let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, w));
        sorted_edges.push((hi, lo, w));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted_edges.len());
    for (r, c, v) in sorted_edges {
        MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
    }
    vcsr
}

fn make_case(name: &str, seed: u64, n: usize, extra_edges: usize) -> WeightedCase {
    assert!(n >= 2 && n % 2 == 0);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);

    let mut edges_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut weighted_edges = Vec::new();

    for pair in perm.chunks_exact(2) {
        let u = pair[0].min(pair[1]);
        let v = pair[0].max(pair[1]);
        edges_set.insert((u, v));
        let w = rng.gen_range(-250..=-50);
        weighted_edges.push((u, v, w));
    }

    while weighted_edges.len() < n / 2 + extra_edges {
        let mut u = rng.gen_range(0..n);
        let mut v = rng.gen_range(0..n);
        if u == v {
            continue;
        }
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }
        if !edges_set.insert((u, v)) {
            continue;
        }
        let w = rng.gen_range(-100..=100);
        weighted_edges.push((u, v, w));
    }

    let graph = build_valued_graph(n, &weighted_edges);
    WeightedCase { name: name.to_string(), graph }
}

fn benchmark_cases() -> Vec<WeightedCase> {
    vec![
        make_case("n64_sparse", 0x00B1_0550_0001, 64, 96),
        make_case("n128_sparse", 0x00B1_0550_0002, 128, 320),
        make_case("n128_dense", 0x00B1_0550_0003, 128, 1400),
    ]
}

fn bench_blossom_v(c: &mut Criterion) {
    let cases = benchmark_cases();

    let mut group = c.benchmark_group("blossom_v");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("Rust", &case.name), case, |b, case| {
            b.iter(|| {
                let matching = case.graph.blossom_v().expect("benchmark case should solve");
                black_box(matching);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_blossom_v);
criterion_main!(benches);
