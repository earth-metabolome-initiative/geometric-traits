//! Criterion benchmarks for the bit-set-backed algorithms: `BitSquareMatrix`
//! primitives, the generic and partitioned maximum clique searches, the
//! modular product, and cycle detection.
//!
//! Every input is generated from a fixed seed, so runs on different revisions
//! time the same work.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{BitSquareMatrix, CSR2D, SquareCSR2D},
    prelude::{
        CycleDetection, GenericGraph, MaximumClique, PartitionedMaximumClique, RandomizedDAG,
    },
    traits::{
        ModularProduct, SizedSparseMatrix, SparseMatrix2D, SquareMatrix,
        randomized_graphs::erdos_renyi_gnp,
    },
};

type Dag = GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>>;

fn gnp(seed: u64, order: usize, probability: f64) -> BitSquareMatrix {
    let graph = erdos_renyi_gnp(seed, order, probability);
    let edges: Vec<(usize, usize)> =
        (0..order).flat_map(|row| graph.sparse_row(row).map(move |column| (row, column))).collect();
    BitSquareMatrix::from_edges(order, edges)
}

fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_square_matrix");
    for &(order, probability) in &[(64usize, 0.3f64), (256, 0.1), (1024, 0.02)] {
        let matrix = gnp(11, order, probability);
        let edges: Vec<(usize, usize)> = (0..order)
            .flat_map(|row| matrix.sparse_row(row).map(move |column| (row, column)))
            .collect();
        let label = format!("n{order}_p{probability}");

        group.bench_with_input(BenchmarkId::new("from_edges", &label), &edges, |b, edges| {
            b.iter(|| BitSquareMatrix::from_edges(order, black_box(edges).iter().copied()));
        });
        group.bench_with_input(BenchmarkId::new("has_entry_all", &label), &matrix, |b, m| {
            b.iter(|| {
                let mut hits = 0usize;
                for row in 0..order {
                    for column in 0..order {
                        hits += usize::from(m.has_entry(black_box(row), black_box(column)));
                    }
                }
                hits
            });
        });
        group.bench_with_input(BenchmarkId::new("sparse_row_all", &label), &matrix, |b, m| {
            b.iter(|| {
                (0..order).map(|row| m.sparse_row(black_box(row)).sum::<usize>()).sum::<usize>()
            });
        });
        group.bench_with_input(BenchmarkId::new("sparse_row_rev_all", &label), &matrix, |b, m| {
            b.iter(|| {
                (0..order)
                    .map(|row| m.sparse_row(black_box(row)).rev().sum::<usize>())
                    .sum::<usize>()
            });
        });
        group.bench_with_input(
            BenchmarkId::new("neighbor_intersection_all", &label),
            &matrix,
            |b, m| {
                b.iter(|| {
                    let mut total = 0usize;
                    for i in 0..order {
                        for j in i + 1..order {
                            total += m.neighbor_intersection_count(black_box(i), black_box(j));
                        }
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

fn bench_maximum_clique(c: &mut Criterion) {
    let mut group = c.benchmark_group("maximum_clique");
    for &(order, probability) in &[(120usize, 0.5f64), (200, 0.3), (60, 0.8)] {
        let matrix = gnp(7, order, probability);
        let label = format!("n{order}_p{probability}");
        group.bench_with_input(BenchmarkId::new("one", &label), &matrix, |b, m| {
            b.iter(|| black_box(m).maximum_clique());
        });
        group.bench_with_input(BenchmarkId::new("all", &label), &matrix, |b, m| {
            b.iter(|| black_box(m).all_maximum_cliques());
        });
    }
    group.finish();
}

fn bench_partitioned_clique(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioned_clique");
    for &(left, right, probability) in &[(12usize, 14usize, 0.35f64), (16, 16, 0.3)] {
        let first = gnp(3, left, probability);
        let second = gnp(5, right, probability);
        let product = first.modular_product_filtered(&second, |_, _| true);
        let (matrix, pairs) = product.into_parts();
        let label = format!("{left}x{right}_p{probability}");

        group.bench_with_input(
            BenchmarkId::new("modular_product", &label),
            &(&first, &second),
            |b, (first, second)| {
                b.iter(|| black_box(*first).modular_product_filtered(*second, |_, _| true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("one", &label),
            &(&matrix, &pairs),
            |b, (m, pairs)| {
                b.iter(|| black_box(*m).maximum_clique_with_pairs(pairs));
            },
        );
        assert_eq!(matrix.order(), left * right);
        assert!(matrix.number_of_defined_values() > 0);
    }
    group.finish();
}

fn bench_cycle_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cycle_detection");
    for &nodes in &[100usize, 1000, 5000] {
        let dag: Dag = RandomizedDAG::randomized_dag(29, nodes);
        assert!(!dag.has_cycle());
        group.bench_with_input(BenchmarkId::new("acyclic", nodes), &dag, |b, dag| {
            b.iter(|| black_box(dag).has_cycle());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_primitives,
    bench_maximum_clique,
    bench_partitioned_clique,
    bench_cycle_detection
);
criterion_main!(benches);
