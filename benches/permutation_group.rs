//! Criterion benchmarks for `PermutationGroup`: building the stabilizer chain
//! and finding the smallest image of a set, on direct products of
//! transpositions, symmetric groups, the wreath product of a transposition
//! with a symmetric group, and the parity group of a chain of stereocentres.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::prelude::PermutationGroup;

/// The permutation of `0..degree` mapping `cycle[i]` to `cycle[i + 1]`.
fn cycle(degree: usize, cycle: &[usize]) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..degree).collect();
    for (index, &point) in cycle.iter().enumerate() {
        permutation[point] = cycle[(index + 1) % cycle.len()];
    }
    permutation
}

/// `(0 1)(2 3)…` with one generator per pair, acting on one point per pair.
fn transpositions(pairs: usize) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    let degree = 2 * pairs;
    let generators = (0..pairs).map(|pair| cycle(degree, &[2 * pair, 2 * pair + 1])).collect();
    (degree, generators, (0..pairs).map(|pair| 2 * pair + 1).collect())
}

/// The symmetric group, acting on every point not divisible by three.
fn symmetric(degree: usize) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    let all: Vec<usize> = (0..degree).collect();
    let generators = vec![cycle(degree, &[0, 1]), cycle(degree, &all)];
    (degree, generators, (0..degree).filter(|point| point % 3 != 0).collect())
}

/// Flips within pairs and arbitrary permutations of the pairs, acting on one
/// point per pair.
fn wreath(pairs: usize) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    let degree = 2 * pairs;
    let mut swap: Vec<usize> = (0..degree).collect();
    swap.swap(0, 2);
    swap.swap(1, 3);
    let rotate = (0..degree).map(|point| (point + 2) % degree).collect();
    let generators = vec![cycle(degree, &[0, 1]), swap, rotate];
    let set = (0..pairs).map(|pair| 2 * pair + usize::from(pair % 3 == 0)).collect();
    (degree, generators, set)
}

/// Automorphisms of disjoint triangles, acting on the final point.
fn triangles(copies: usize) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    let degree = 3 * copies;
    let mut generators = Vec::new();
    for copy in 0..copies {
        let base = 3 * copy;
        generators.push(cycle(degree, &[base, base + 1, base + 2]));
        generators.push(cycle(degree, &[base + 1, base + 2]));
    }
    for copy in 0..copies.saturating_sub(1) {
        let mut swap: Vec<usize> = (0..degree).collect();
        for offset in 0..3 {
            swap.swap(3 * copy + offset, 3 * copy + 3 + offset);
        }
        generators.push(swap);
    }
    (degree, generators, vec![degree - 1])
}

/// Independent parity flips of `centres` stereocentres plus the end-to-end
/// reversal, acting on one parity point per centre.
fn stereo_chain(centres: usize) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    let degree = 2 * centres;
    let mut generators: Vec<Vec<usize>> =
        (0..centres).map(|centre| cycle(degree, &[2 * centre, 2 * centre + 1])).collect();
    generators.push((0..degree).map(|point| 2 * (centres - 1 - point / 2) + point % 2).collect());
    let set = (0..centres).map(|centre| 2 * centre + usize::from(centre % 2 == 0)).collect();
    (degree, generators, set)
}

fn bench_permutation_group(c: &mut Criterion) {
    let cases = [
        ("transpositions_32", transpositions(32)),
        ("transpositions_64", transpositions(64)),
        ("symmetric_40", symmetric(40)),
        ("wreath_24", wreath(24)),
        ("stereo_chain_48", stereo_chain(48)),
        ("triangles_40", triangles(40)),
        ("triangles_80", triangles(80)),
        ("triangles_200", triangles(200)),
    ];
    let mut group = c.benchmark_group("permutation_group");
    for (label, (degree, generators, set)) in &cases {
        group.bench_with_input(BenchmarkId::new("new", label), generators, |b, generators| {
            b.iter(|| PermutationGroup::new(*degree, black_box(generators)).unwrap());
        });
        let permutation_group = PermutationGroup::new(*degree, generators).unwrap();
        group.bench_with_input(
            BenchmarkId::new("smallest_image", label),
            &(&permutation_group, set),
            |b, (permutation_group, set)| {
                b.iter(|| black_box(*permutation_group).smallest_image(black_box(set)).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_permutation_group);
criterion_main!(benches);
