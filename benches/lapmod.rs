//! Criterion benchmark comparing LAPMOD and SparseLAPJV on sparse matrices
//! of varying sizes and densities.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{LAPMOD, MatrixMut, SparseLAPJV, SparseMatrixMut, SparseValuedMatrix},
    traits::algorithms::randomized_graphs::XorShift64,
};
use num_traits::ToPrimitive;

fn target_edge_count(n: usize, density: f64) -> usize {
    let Some(total_cells) = n.checked_mul(n).and_then(|value| value.to_f64()) else {
        return usize::MAX;
    };
    (total_cells * density).floor().to_usize().unwrap_or(usize::MAX)
}

fn random_index(rng: &mut XorShift64, n: usize) -> usize {
    let n_u64 = u64::try_from(n).expect("usize values always fit into u64");
    let raw = rng.next().expect("XorShift64 produces infinite values") % n_u64;
    usize::try_from(raw).expect("raw index is modulo n and always fits usize")
}

fn random_cost(rng: &mut XorShift64) -> f64 {
    let raw = rng.next().expect("XorShift64 produces infinite values") % 999 + 1;
    let cents = u32::try_from(raw).expect("bounded to the range 1..=999");
    f64::from(cents) / 100.0
}

/// Generate a random n×n sparse valued matrix with the given density.
///
/// Guarantees at least one edge per row so that a perfect matching exists.
fn sparse_valued_matrix(
    seed: u64,
    n: usize,
    density: f64,
) -> ValuedCSR2D<usize, usize, usize, f64> {
    let mut rng = XorShift64::from(seed);
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((n, n), target_edge_count(n, density));

    // Ensure every row has at least one edge (required for feasibility).
    for row in 0..n {
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng); // 0.01 .. 9.99
        let _ = csr.add((row, col, cost));
    }

    // Add random additional edges according to the target density.
    let target_edges = target_edge_count(n, density);
    for _ in 0..target_edges {
        let row = random_index(&mut rng, n);
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng);
        let _ = csr.add((row, col, cost)); // silently ignore duplicate keys
    }

    csr
}

/// Benchmark LAPMOD vs SparseLAPJV for small/medium sizes and two densities.
fn bench_lapmod_vs_sparse_lapjv(c: &mut Criterion) {
    let mut group = c.benchmark_group("lapmod_vs_sparse_lapjv");

    for &n in &[20usize, 50, 100] {
        for &density in &[0.05f64, 0.20] {
            let csr = sparse_valued_matrix(
                42 + u64::try_from(n).expect("usize values always fit into u64"),
                n,
                density,
            );
            let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;
            let padding = max_cost * 0.9; // padding < max_cost

            let label = format!("n={n}_d={density:.2}");

            group.bench_with_input(BenchmarkId::new("LAPMOD", &label), &csr, |b, m| {
                b.iter(|| black_box(m.lapmod(black_box(max_cost)).ok()));
            });

            group.bench_with_input(BenchmarkId::new("SparseLAPJV", &label), &csr, |b, m| {
                b.iter(|| black_box(m.sparse_lapjv(black_box(padding), black_box(max_cost)).ok()));
            });
        }
    }

    group.finish();
}

/// Large-sparse benchmark where LAPMOD's O(|E|) memory advantage is clear.
fn bench_lapmod_large_sparse(c: &mut Criterion) {
    let n = 200;
    let density = 0.05;
    let csr = sparse_valued_matrix(12345, n, density);
    let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;
    let padding = max_cost * 0.9;

    let mut group = c.benchmark_group("lapmod_large_sparse");

    group.bench_function("LAPMOD_200x5pct", |b| {
        b.iter(|| black_box(csr.lapmod(black_box(max_cost)).ok()));
    });

    group.bench_function("SparseLAPJV_200x5pct", |b| {
        b.iter(|| black_box(csr.sparse_lapjv(black_box(padding), black_box(max_cost)).ok()));
    });

    group.finish();
}

criterion_group!(benches, bench_lapmod_vs_sparse_lapjv, bench_lapmod_large_sparse);
criterion_main!(benches);
