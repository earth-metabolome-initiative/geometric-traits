//! Criterion benchmark comparing maximum matching algorithms on random
//! undirected graphs of varying sizes and densities:
//!
//! - Blossom (O(V²E))
//! - Blossom+KS1 / Blossom+KS12
//! - Micali-Vazirani (O(√V·E))
//! - Micali-Vazirani+KS1 / Micali-Vazirani+KS12
//! - Blum (implementation worst case O(V·(V+E)); paper phased bound
//!   O(√V·(V+E)))
//! - Blum+KS1 / Blum+KS12
//! - External `blossom` crate (independent verification)

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::{
        XorShift64, barabasi_albert, barbell_graph, cycle_graph, erdos_renyi_gnm, erdos_renyi_gnp,
        grid_graph, random_regular_graph, watts_strogatz,
    },
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

fn bench_exact_matchers(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    g: &SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    label: &str,
    ext: &blossom::Graph,
) {
    group.bench_with_input(BenchmarkId::new("Blossom", label), g, |b, g| {
        b.iter(|| black_box(g.blossom()));
    });

    group.bench_with_input(BenchmarkId::new("Blossom+KS1", label), g, |b, g| {
        b.iter(|| black_box(g.blossom_with_karp_sipser(KarpSipserRules::Degree1)));
    });

    group.bench_with_input(BenchmarkId::new("Blossom+KS12", label), g, |b, g| {
        b.iter(|| black_box(g.blossom_with_karp_sipser(KarpSipserRules::Degree1And2)));
    });

    group.bench_with_input(BenchmarkId::new("MicaliVazirani", label), g, |b, g| {
        b.iter(|| black_box(g.micali_vazirani()));
    });

    group.bench_with_input(BenchmarkId::new("MicaliVazirani+KS1", label), g, |b, g| {
        b.iter(|| black_box(g.micali_vazirani_with_karp_sipser(KarpSipserRules::Degree1)));
    });

    group.bench_with_input(BenchmarkId::new("MicaliVazirani+KS12", label), g, |b, g| {
        b.iter(|| black_box(g.micali_vazirani_with_karp_sipser(KarpSipserRules::Degree1And2)));
    });

    group.bench_with_input(BenchmarkId::new("Blum", label), g, |b, g| {
        b.iter(|| black_box(g.blum()));
    });

    group.bench_with_input(BenchmarkId::new("Blum+KS1", label), g, |b, g| {
        b.iter(|| black_box(g.blum_with_karp_sipser(KarpSipserRules::Degree1)));
    });

    group.bench_with_input(BenchmarkId::new("Blum+KS12", label), g, |b, g| {
        b.iter(|| black_box(g.blum_with_karp_sipser(KarpSipserRules::Degree1And2)));
    });

    group.bench_with_input(BenchmarkId::new("ExtBlossom", label), ext, |b, ext| {
        b.iter(|| black_box(ext.maximum_matching()));
    });
}

fn bench_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching");

    for &(n, avg_deg) in &[(20usize, 4usize), (50, 6), (100, 6), (200, 6), (500, 6), (1000, 6)] {
        let g = random_graph(42 + n as u64, n, avg_deg);
        let label = format!("n={n}_d={avg_deg}");
        let ext = to_blossom_graph(&g, n);
        bench_exact_matchers(&mut group, &g, &label, &ext);
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
        bench_exact_matchers(&mut group, &g, &label, &ext);
    }

    group.finish();
}

fn bench_matching_degree_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_degree_two");

    for &n in &[64usize, 128, 256, 512] {
        let cycle = cycle_graph(n);
        let cycle_label = format!("cycle_n={n}");
        let cycle_ext = to_blossom_graph(&cycle, n);
        bench_exact_matchers(&mut group, &cycle, &cycle_label, &cycle_ext);

        let ladder = grid_graph(2, n / 2);
        let ladder_label = format!("ladder_2x{}", n / 2);
        let ladder_ext = to_blossom_graph(&ladder, n);
        bench_exact_matchers(&mut group, &ladder, &ladder_label, &ladder_ext);
    }

    for &path_len in &[8usize, 16, 32, 64] {
        let barbell = barbell_graph(8, path_len);
        let label = format!("barbell_k=8_p={path_len}");
        let order = barbell.order();
        let ext = to_blossom_graph(&barbell, order);
        bench_exact_matchers(&mut group, &barbell, &label, &ext);
    }

    group.finish();
}

fn bench_matching_sparse_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_sparse_random");
    // This group is meant for quick local screening of whether KS is worth
    // carrying into the larger benchmark suite.
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for &n in &[256usize, 1024] {
        for &avg_degree in &[2usize, 4] {
            let order_f64 = f64::from(u32::try_from(n).expect("benchmark order fits in u32"));
            let degree_f64 =
                f64::from(u32::try_from(avg_degree).expect("benchmark degree fits in u32"));
            let prob_graph = erdos_renyi_gnp(
                0xA11C_E001_u64.wrapping_add((n as u64) << 8).wrapping_add(avg_degree as u64),
                n,
                degree_f64 / order_f64,
            );
            let prob_label = format!("gnp_n={n}_c={avg_degree}");
            let prob_ext_graph = to_blossom_graph(&prob_graph, n);
            bench_exact_matchers(&mut group, &prob_graph, &prob_label, &prob_ext_graph);

            let edge_count_graph = erdos_renyi_gnm(
                0xA11C_E002_u64.wrapping_add((n as u64) << 8).wrapping_add(avg_degree as u64),
                n,
                n * avg_degree / 2,
            );
            let edge_count_label = format!("gnm_n={n}_d={avg_degree}");
            let edge_count_ext_graph = to_blossom_graph(&edge_count_graph, n);
            bench_exact_matchers(
                &mut group,
                &edge_count_graph,
                &edge_count_label,
                &edge_count_ext_graph,
            );
        }

        for &k in &[3usize, 4] {
            let regular = random_regular_graph(
                0xA11C_E003_u64.wrapping_add((n as u64) << 8).wrapping_add(k as u64),
                n,
                k,
            );
            let regular_label = format!("regular_n={n}_k={k}");
            let regular_ext = to_blossom_graph(&regular, n);
            bench_exact_matchers(&mut group, &regular, &regular_label, &regular_ext);
        }
    }

    for &n in &[256usize, 1024] {
        let ba = barabasi_albert(0xA11C_E004_u64.wrapping_add(n as u64), n, 2);
        let ba_label = format!("barabasi_albert_n={n}_m=2");
        let ba_ext = to_blossom_graph(&ba, n);
        bench_exact_matchers(&mut group, &ba, &ba_label, &ba_ext);

        let ws = watts_strogatz(0xA11C_E005_u64.wrapping_add(n as u64), n, 4, 0.3);
        let ws_label = format!("watts_strogatz_n={n}_k=4_beta=0.3");
        let ws_ext = to_blossom_graph(&ws, n);
        bench_exact_matchers(&mut group, &ws, &ws_label, &ws_ext);
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matching,
    bench_matching_dense,
    bench_matching_degree_two,
    bench_matching_sparse_random
);
criterion_main!(benches);
