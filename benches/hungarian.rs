//! Criterion benchmark comparing Hungarian, LAPJV, and LAPMOD on dense
//! (padded) matrices of varying sizes.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{MatrixMut, SparseHungarian, SparseLAPJV, SparseMatrixMut, SparseValuedMatrix},
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

fn sparse_valued_matrix(
    seed: u64,
    n: usize,
    density: f64,
) -> ValuedCSR2D<usize, usize, usize, f64> {
    let mut rng = XorShift64::from(seed);
    let mut csr: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((n, n), target_edge_count(n, density));

    for row in 0..n {
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng);
        let _ = csr.add((row, col, cost));
    }

    let target_edges = target_edge_count(n, density);
    for _ in 0..target_edges {
        let row = random_index(&mut rng, n);
        let col = random_index(&mut rng, n);
        let cost = random_cost(&mut rng);
        let _ = csr.add((row, col, cost));
    }

    csr
}

fn bench_hungarian_vs_lapjv(c: &mut Criterion) {
    let mut group = c.benchmark_group("hungarian_vs_lapjv");

    // Exercise the overflow fallback once outside the measured benchmark loops.
    let _ = black_box(target_edge_count(usize::MAX, 1.0));

    for &n in &[20usize, 50, 100] {
        let density = 0.20;
        let csr = sparse_valued_matrix(
            42 + u64::try_from(n).expect("usize values always fit into u64"),
            n,
            density,
        );
        let max_cost = csr.max_sparse_value().unwrap_or(100.0) * 2.0 + 1.0;
        let padding = max_cost * 0.9;

        let label = format!("n={n}");

        group.bench_with_input(BenchmarkId::new("Hungarian", &label), &csr, |b, m| {
            b.iter(|| black_box(m.sparse_hungarian(black_box(padding), black_box(max_cost)).ok()));
        });

        group.bench_with_input(BenchmarkId::new("SparseLAPJV", &label), &csr, |b, m| {
            b.iter(|| black_box(m.sparse_lapjv(black_box(padding), black_box(max_cost)).ok()));
        });
    }

    group.finish();
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_target_edge_count_overflow_falls_back_to_usize_max() {
        assert_eq!(super::target_edge_count(usize::MAX, 1.0), usize::MAX);
    }
}

criterion_group!(benches, bench_hungarian_vs_lapjv);
criterion_main!(benches);
