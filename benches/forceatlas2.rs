//! Criterion benchmarks for the ForceAtlas2 layout: exact pairwise
//! repulsion versus the Barnes-Hut approximation.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::ForceAtlas2Config};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

/// Ring of `n` nodes with deterministic long-range chords.
fn ring_with_chords(n: usize) -> WeightedMatrix {
    let mut edges: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, (i + 1) % n, 1.0)).collect();
    for i in (0..n).step_by(5) {
        let j = (i + 97) % n;
        edges.push((i.min(j), i.max(j), 1.0));
    }
    edges.sort_unstable_by_key(|edge| (edge.0, edge.1));
    edges.dedup_by_key(|&mut (u, v, _)| (u, v));

    let mut directed_edges = Vec::with_capacity(edges.len() * 2);
    for (source, destination, weight) in edges {
        directed_edges.push((source, destination, weight));
        directed_edges.push((destination, source, weight));
    }
    directed_edges.sort_unstable_by(|(s1, d1, _), (s2, d2, _)| (s1, d1).cmp(&(s2, d2)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed_edges.len())
        .expected_shape((n, n))
        .edges(directed_edges.into_iter())
        .build()
        .unwrap()
}

fn bench_forceatlas2(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("forceatlas2");
    group.sample_size(10);

    for &n in &[100_usize, 1000, 10000] {
        let graph = ring_with_chords(n);
        for (label, barnes_hut) in [("exact", false), ("barnes_hut", true)] {
            // Exact repulsion at 10000 nodes is quadratic and only there
            // to show the crossover, skip it to keep the suite fast.
            if n == 10000 && !barnes_hut {
                continue;
            }
            let config = ForceAtlas2Config { iterations: 5, barnes_hut, ..Default::default() };
            group.bench_with_input(BenchmarkId::new(label, n), &graph, |bencher, graph| {
                bencher.iter(|| black_box(graph.force_atlas2(black_box(&config)).unwrap()));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_forceatlas2);
criterion_main!(benches);
