//! Criterion benchmark to evaluate the performance of the 'lin' function.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::{GenericGraph, RandomizedDAG},
    traits::{Lin, MonopartiteGraph, ScalarSimilarity, randomized_graphs::XorShift64},
};

/// Benchmark for the `lin` function
fn bench_lin(c: &mut Criterion) {
    c.bench_function("lin_10", |b| {
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
                let occurrences: Vec<usize> = vec![1; dag.number_of_nodes()];
                let lin = dag.lin(&occurrences).unwrap();
                for src in black_box(dag.node_ids()) {
                    for dst in black_box(dag.node_ids()) {
                        total_similarity += lin.similarity(black_box(&src), black_box(&dst));
                    }
                }
            }
            black_box(total_similarity)
        });
    });
}

criterion_group!(benches, bench_lin);
criterion_main!(benches);
