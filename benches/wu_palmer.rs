//! Criterion benchmark to evaluate the performance of the 'wu-palmer' function.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use geometric_traits::impls::{CSR2D, SquareCSR2D};
use geometric_traits::traits::ScalarSimilarity;
use geometric_traits::{
    prelude::{GenericGraph, RandomizedDAG},
    traits::{MonopartiteGraph, WuPalmer, randomized_graphs::XorShift64},
};

/// Benchmark for the `wu-palmer` function
fn bench_wu_palmer(c: &mut Criterion) {
    c.bench_function("wu_palmer_10", |b| {
        const NUMBER_OF_DAGS: usize = 10;
        let mut dags = Vec::with_capacity(NUMBER_OF_DAGS);
        let mut xorshift = XorShift64::from(24537839457);
        for _ in 0..NUMBER_OF_DAGS {
            let seed = xorshift.next().unwrap();
            let dag: GenericGraph<usize, SquareCSR2D<CSR2D<usize, usize, usize>>> =
                RandomizedDAG::randomized_dag(seed, 10);
            dags.push(dag);
        }
        b.iter(|| {
            let mut total_similarity = 0.0;
            for dag in &dags {
                let wu_palmer = dag.wu_palmer().unwrap();

                for src in black_box(dag.node_ids()) {
                    for dst in black_box(dag.node_ids()) {
                        total_similarity += wu_palmer.similarity(black_box(&src), black_box(&dst));
                    }
                }
            }
            black_box(total_similarity)
        });
    });
}

criterion_group!(benches, bench_wu_palmer);
criterion_main!(benches);
