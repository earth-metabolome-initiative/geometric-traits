//! Criterion benchmark comparing maximum matching algorithms on random
//! undirected graphs of varying sizes and densities:
//!
//! - Blossom (O(V²E))
//! - Micali-Vazirani (O(√V·E))
//! - Blum (O(√V·(V+E)·α))
//! - External `blossom` crate (independent verification)

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::XorShift64,
};

/// Build the external `blossom` crate graph from our CSR.
fn to_blossom_graph(g: &SymmetricCSR2D<CSR2D<usize, usize, usize>>, n: usize) -> blossom::Graph {
    let adj: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|v| {
            let neighbors: Vec<usize> = g.sparse_row(v).collect();
            (v, neighbors)
        })
        .collect();
    adj.iter().collect()
}

/// Build a random undirected graph with `n` vertices and approximately
/// `n * avg_degree / 2` edges (no self-loops, no multi-edges).
fn random_graph(
    seed: u64,
    n: usize,
    avg_degree: usize,
) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let mut rng = XorShift64::from(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed });
    let target_edges = n * avg_degree / 2;
    let mut edges = Vec::with_capacity(target_edges);

    let n_u64 = n as u64;
    for _ in 0..target_edges * 2 {
        let raw_u = rng.next().unwrap() % n_u64;
        let raw_v = rng.next().unwrap() % n_u64;
        let u = usize::try_from(raw_u).expect("vertex index fits into usize");
        let v = usize::try_from(raw_v).expect("vertex index fits into usize");
        if u != v {
            edges.push((u.min(v), u.max(v)));
        }
    }
    edges.sort_unstable();
    edges.dedup();

    UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn bench_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching");

    for &(n, avg_deg) in &[(20usize, 4usize), (50, 6), (100, 6), (200, 6), (500, 6), (1000, 6)] {
        let g = random_graph(42 + n as u64, n, avg_deg);
        let label = format!("n={n}_d={avg_deg}");
        let ext = to_blossom_graph(&g, n);

        group.bench_with_input(BenchmarkId::new("Blossom", &label), &g, |b, g| {
            b.iter(|| black_box(g.blossom()));
        });

        group.bench_with_input(BenchmarkId::new("MicaliVazirani", &label), &g, |b, g| {
            b.iter(|| black_box(g.micali_vazirani()));
        });

        group.bench_with_input(BenchmarkId::new("Blum", &label), &g, |b, g| {
            b.iter(|| black_box(g.blum()));
        });

        group.bench_with_input(BenchmarkId::new("ExtBlossom", &label), &ext, |b, ext| {
            b.iter(|| black_box(ext.maximum_matching()));
        });
    }

    group.finish();
}

fn bench_matching_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_dense");

    for &n in &[20usize, 50, 100, 200] {
        let avg_deg = n / 2;
        let g = random_graph(99 + n as u64, n, avg_deg);
        let label = format!("n={n}_d={avg_deg}");
        let ext = to_blossom_graph(&g, n);

        group.bench_with_input(BenchmarkId::new("Blossom", &label), &g, |b, g| {
            b.iter(|| black_box(g.blossom()));
        });

        group.bench_with_input(BenchmarkId::new("MicaliVazirani", &label), &g, |b, g| {
            b.iter(|| black_box(g.micali_vazirani()));
        });

        group.bench_with_input(BenchmarkId::new("Blum", &label), &g, |b, g| {
            b.iter(|| black_box(g.blum()));
        });

        group.bench_with_input(BenchmarkId::new("ExtBlossom", &label), &ext, |b, ext| {
            b.iter(|| black_box(ext.maximum_matching()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matching, bench_matching_dense);
criterion_main!(benches);
