//! Criterion benchmark comparing the old valued symmetric construction path
//! against the new one-step builder.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::UndiEdgesBuilder,
    prelude::*,
    traits::{EdgesBuilder, MatrixMut, SparseMatrix},
};

type UpperEntry = (usize, usize, u32);
type Topology = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type ValuedSymmetric = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u32>>;

fn synthetic_upper_entries(order: usize, fanout: usize) -> Vec<UpperEntry> {
    let mut entries = Vec::new();

    for row in 0..order {
        for step in 1..=fanout {
            let column = row + step;
            if column < order {
                entries.push((row, column, payload(row, column)));
            }
        }

        let chord = row + fanout + 2;
        if row % 3 == 0 && chord < order {
            entries.push((row, chord, payload(row, chord)));
        }
    }

    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    entries.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    entries
}

fn payload(row: usize, column: usize) -> u32 {
    let mixed = row.wrapping_mul(1_315_423_911usize) ^ column.wrapping_mul(2_654_435_761usize);
    u32::try_from(mixed & 0xffff).expect("16-bit masked payload always fits in u32")
}

fn payload_lookup(entries: &[UpperEntry], row: usize, column: usize) -> u32 {
    let (upper_row, upper_column) = if row <= column { (row, column) } else { (column, row) };
    let index = entries
        .binary_search_by(|&(candidate_row, candidate_column, _)| {
            candidate_row.cmp(&upper_row).then(candidate_column.cmp(&upper_column))
        })
        .expect("symmetric topology must only contain known upper-triangular edges");
    entries[index].2
}

fn build_slow(order: usize, entries: &[UpperEntry]) -> ValuedSymmetric {
    let topology: Topology = UndiEdgesBuilder::default()
        .expected_shape(order)
        .expected_number_of_edges(entries.len())
        .edges(entries.iter().map(|&(row, column, _)| (row, column)))
        .build()
        .unwrap();

    let diagonal_values = topology.number_of_defined_diagonal_values();
    let mut valued = ValuedCSR2D::with_sparse_shaped_capacity(
        (order, order),
        topology.number_of_defined_values(),
    );
    for (row, column) in SparseMatrix::sparse_coordinates(&topology) {
        let value = payload_lookup(entries, row, column);
        MatrixMut::add(&mut valued, (row, column, value)).unwrap();
    }

    SymmetricCSR2D::from_parts(SquareCSR2D::from_parts(valued, diagonal_values))
}

fn build_fast(order: usize, entries: &[UpperEntry]) -> ValuedSymmetric {
    SymmetricCSR2D::from_sorted_upper_triangular_entries(order, entries.iter().copied()).unwrap()
}

fn bench_valued_symmetric_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("valued_symmetric_builder");
    group.measurement_time(Duration::from_secs(2));

    for &(order, fanout) in &[(24usize, 2usize), (48, 3), (96, 3), (192, 4)] {
        let entries = synthetic_upper_entries(order, fanout);
        let edge_count = entries.len();
        let case = format!("n{order}_e{edge_count}_f{fanout}");
        group.throughput(Throughput::Elements(edge_count as u64));

        group.bench_with_input(BenchmarkId::new("slow", &case), &entries, |b, entries| {
            b.iter(|| black_box(build_slow(order, black_box(entries))));
        });

        group.bench_with_input(BenchmarkId::new("fast", &case), &entries, |b, entries| {
            b.iter(|| black_box(build_fast(order, black_box(entries))));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_valued_symmetric_builder);
criterion_main!(benches);
