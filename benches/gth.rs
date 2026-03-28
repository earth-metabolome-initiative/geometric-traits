//! Criterion benchmark for the dense GTH stationary-distribution algorithm.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{impls::VecMatrix2D, prelude::*};

const SEED: u64 = 0x5EED_5EED_D15E_A5E5;
const MASK: u64 = u64::MAX;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn deterministic_weight(seed: u64, row: usize, column: usize) -> f64 {
    let row_u64 = u64::try_from(row).expect("usize always fits into u64");
    let column_u64 = u64::try_from(column).expect("usize always fits into u64");
    let mixed = seed
        ^ row_u64.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ column_u64.wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ MASK;
    let raw = splitmix64(mixed);
    let mut value = 0.05 + f64::from((raw % 10_000) as u32) / 10_000.0;
    if row == column {
        value += 0.5;
    }
    value
}

fn deterministic_row_stochastic_matrix(seed: u64, n: usize) -> VecMatrix2D<f64> {
    let mut data = Vec::with_capacity(n * n);
    for row in 0..n {
        let row_start = data.len();
        let mut row_sum = 0.0;
        for column in 0..n {
            let value = deterministic_weight(seed, row, column);
            row_sum += value;
            data.push(value);
        }
        for value in &mut data[row_start..row_start + n] {
            *value /= row_sum;
        }
    }
    VecMatrix2D::new(n, n, data)
}

fn bench_gth_dense_stationary_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("gth_dense_stationary_distribution");
    let config = GthConfig::default();

    for &n in &[10usize, 100, 1000] {
        let matrix = deterministic_row_stochastic_matrix(
            SEED ^ u64::try_from(n).expect("usize always fits into u64"),
            n,
        );

        group.bench_with_input(BenchmarkId::new("GTH", n), &matrix, |b, matrix| {
            b.iter(|| {
                let matrix = black_box(matrix);
                let config = black_box(&config);
                let result = matrix.gth(config).ok();
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gth_dense_stationary_distribution);
criterion_main!(benches);
