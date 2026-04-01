//! Criterion benchmark comparing maximum matching algorithms on random
//! undirected graphs of varying sizes and densities:
//!
//! - Blossom (O(V²E))
//! - Gabow 1976 (O(V^3))
//! - Micali-Vazirani (O(√V·E))

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

/// Gabow's worst-case family G_{6m} from the 1976 paper.
fn gabow_worst_case_graph(m: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let n = 6 * m;
    let clique_order = 4 * m;
    let mut edges = Vec::new();

    for u in 0..clique_order {
        for v in (u + 1)..clique_order {
            edges.push((u, v));
        }
    }

    for i in 0..(2 * m) {
        edges.push((2 * i, clique_order + i));
    }

    edges.sort_unstable();

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
) {
    group.bench_with_input(BenchmarkId::new("Blossom", label), g, |b, g| {
        b.iter(|| black_box(g.blossom()));
    });

    group.bench_with_input(BenchmarkId::new("Gabow1976", label), g, |b, g| {
        b.iter(|| black_box(g.gabow_1976()));
    });

    group.bench_with_input(BenchmarkId::new("MicaliVazirani", label), g, |b, g| {
        b.iter(|| black_box(g.micali_vazirani()));
    });
}

fn bench_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching");

    for &(n, avg_deg) in &[(20usize, 4usize), (50, 6), (100, 6), (200, 6), (500, 6), (1000, 6)] {
        let g = random_graph(42 + n as u64, n, avg_deg);
        let label = format!("n={n}_d={avg_deg}");
        bench_exact_matchers(&mut group, &g, &label);
    }

    group.finish();
}

fn bench_matching_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_dense");

    for &n in &[20usize, 50, 100, 200] {
        let avg_deg = n / 2;
        let g = random_graph(99 + n as u64, n, avg_deg);
        let label = format!("n={n}_d={avg_deg}");
        bench_exact_matchers(&mut group, &g, &label);
    }

    group.finish();
}

fn bench_matching_degree_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_degree_two");

    for &n in &[64usize, 128, 256, 512] {
        let cycle = cycle_graph(n);
        let cycle_label = format!("cycle_n={n}");
        bench_exact_matchers(&mut group, &cycle, &cycle_label);

        let ladder = grid_graph(2, n / 2);
        let ladder_label = format!("ladder_2x{}", n / 2);
        bench_exact_matchers(&mut group, &ladder, &ladder_label);
    }

    for &path_len in &[8usize, 16, 32, 64] {
        let barbell = barbell_graph(8, path_len);
        let label = format!("barbell_k=8_p={path_len}");
        bench_exact_matchers(&mut group, &barbell, &label);
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
            bench_exact_matchers(&mut group, &prob_graph, &prob_label);

            let edge_count_graph = erdos_renyi_gnm(
                0xA11C_E002_u64.wrapping_add((n as u64) << 8).wrapping_add(avg_degree as u64),
                n,
                n * avg_degree / 2,
            );
            let edge_count_label = format!("gnm_n={n}_d={avg_degree}");
            bench_exact_matchers(&mut group, &edge_count_graph, &edge_count_label);
        }

        for &k in &[3usize, 4] {
            let regular = random_regular_graph(
                0xA11C_E003_u64.wrapping_add((n as u64) << 8).wrapping_add(k as u64),
                n,
                k,
            )
            .expect("benchmark regular graph inputs should be valid");
            let regular_label = format!("regular_n={n}_k={k}");
            bench_exact_matchers(&mut group, &regular, &regular_label);
        }
    }

    for &n in &[256usize, 1024] {
        let ba = barabasi_albert(0xA11C_E004_u64.wrapping_add(n as u64), n, 2);
        let ba_label = format!("barabasi_albert_n={n}_m=2");
        bench_exact_matchers(&mut group, &ba, &ba_label);

        let ws = watts_strogatz(0xA11C_E005_u64.wrapping_add(n as u64), n, 4, 0.3);
        let ws_label = format!("watts_strogatz_n={n}_k=4_beta=0.3");
        bench_exact_matchers(&mut group, &ws, &ws_label);
    }

    group.finish();
}

fn bench_gabow_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("gabow_worst_case");

    for &m in &[4usize, 8, 12] {
        let g = gabow_worst_case_graph(m);
        let label = format!("m={m}_n={}", g.order());
        bench_exact_matchers(&mut group, &g, &label);
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matching,
    bench_matching_dense,
    bench_matching_degree_two,
    bench_matching_sparse_random,
    bench_gabow_worst_case
);
criterion_main!(benches);
