#![allow(clippy::pedantic)]
//! Criterion benchmarks for minimum-cost maximum balanced flow, including a
//! `kocay` baseline on the same deterministic instances.

use std::{collections::HashSet, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{impls::ValuedCSR2D, prelude::*};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};

type CapacityGraph = ValuedCSR2D<usize, usize, usize, usize>;
type CostGraph = ValuedCSR2D<usize, usize, usize, i64>;

#[derive(Clone)]
struct WeightedCase {
    name: String,
    capacities: CapacityGraph,
    costs: CostGraph,
    budgets: Vec<usize>,
}

fn build_capacity_graph(n: usize, edges: &[(usize, usize, usize, i64)]) -> CapacityGraph {
    let mut directed_edges = Vec::with_capacity(edges.len() * 2);
    for &(u, v, capacity, _cost) in edges {
        if u == v || capacity == 0 {
            continue;
        }
        directed_edges.push((u, v, capacity));
        directed_edges.push((v, u, capacity));
    }
    directed_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));

    let mut graph: CapacityGraph =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), directed_edges.len());
    for edge in directed_edges {
        MatrixMut::add(&mut graph, edge).expect("benchmark graph edge insertion should succeed");
    }
    graph
}

fn build_cost_graph(n: usize, edges: &[(usize, usize, usize, i64)]) -> CostGraph {
    let mut directed_edges = Vec::with_capacity(edges.len() * 2);
    for &(u, v, capacity, cost) in edges {
        if u == v || capacity == 0 {
            continue;
        }
        directed_edges.push((u, v, cost));
        directed_edges.push((v, u, cost));
    }
    directed_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));

    let mut graph: CostGraph =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), directed_edges.len());
    for edge in directed_edges {
        MatrixMut::add(&mut graph, edge).expect("benchmark graph edge insertion should succeed");
    }
    graph
}

fn make_weighted_case(
    name: &str,
    seed: u64,
    n: usize,
    edge_count: usize,
    max_capacity: usize,
    max_cost: i64,
) -> WeightedCase {
    assert!(n >= 2);
    assert!(edge_count <= n * (n - 1) / 2);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
    for u in 0..n {
        for v in (u + 1)..n {
            pairs.push((u, v));
        }
    }
    pairs.shuffle(&mut rng);

    let mut incident_capacity = vec![0usize; n];
    let mut edges = Vec::with_capacity(edge_count);
    for &(u, v) in pairs.iter().take(edge_count) {
        let capacity = rng.gen_range(1..=max_capacity);
        let cost = rng.gen_range(0..=max_cost);
        incident_capacity[u] += capacity;
        incident_capacity[v] += capacity;
        edges.push((u, v, capacity, cost));
    }

    let mut budgets = Vec::with_capacity(n);
    for &total_incident_capacity in &incident_capacity {
        if total_incident_capacity == 0 {
            budgets.push(0);
            continue;
        }
        let lower = (total_incident_capacity / 3).max(1);
        let upper = ((2 * total_incident_capacity) / 3).max(lower);
        budgets.push(rng.gen_range(lower..=upper));
    }

    WeightedCase {
        name: name.to_owned(),
        capacities: build_capacity_graph(n, &edges),
        costs: build_cost_graph(n, &edges),
        budgets,
    }
}

fn make_bipartite_case(
    name: &str,
    seed: u64,
    left_size: usize,
    right_size: usize,
    edge_count: usize,
    max_capacity: usize,
    max_cost: i64,
) -> WeightedCase {
    assert!(left_size > 0);
    assert!(right_size > 0);
    assert!(edge_count >= left_size + right_size - 1);
    assert!(edge_count <= left_size * right_size);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut incident_capacity = vec![0usize; left_size + right_size];
    let mut chosen_pairs = HashSet::new();
    let mut edges = Vec::with_capacity(edge_count);

    for right in 0..right_size {
        let left = if right == 0 { 0 } else { rng.gen_range(0..left_size) };
        if chosen_pairs.insert((left, right)) {
            let capacity = rng.gen_range(1..=max_capacity);
            let cost = rng.gen_range(0..=max_cost);
            let right_vertex = left_size + right;
            incident_capacity[left] += capacity;
            incident_capacity[right_vertex] += capacity;
            edges.push((left, right_vertex, capacity, cost));
        }
    }
    for left in 1..left_size {
        let right = rng.gen_range(0..right_size);
        if chosen_pairs.insert((left, right)) {
            let capacity = rng.gen_range(1..=max_capacity);
            let cost = rng.gen_range(0..=max_cost);
            let right_vertex = left_size + right;
            incident_capacity[left] += capacity;
            incident_capacity[right_vertex] += capacity;
            edges.push((left, right_vertex, capacity, cost));
        }
    }

    let mut remaining_pairs = Vec::with_capacity(left_size * right_size);
    for left in 0..left_size {
        for right in 0..right_size {
            if !chosen_pairs.contains(&(left, right)) {
                remaining_pairs.push((left, right));
            }
        }
    }
    remaining_pairs.shuffle(&mut rng);

    for (left, right) in remaining_pairs.into_iter().take(edge_count.saturating_sub(edges.len())) {
        let capacity = rng.gen_range(1..=max_capacity);
        let cost = rng.gen_range(0..=max_cost);
        let right_vertex = left_size + right;
        incident_capacity[left] += capacity;
        incident_capacity[right_vertex] += capacity;
        edges.push((left, right_vertex, capacity, cost));
    }

    let mut budgets = Vec::with_capacity(incident_capacity.len());
    for &total_incident_capacity in &incident_capacity {
        if total_incident_capacity == 0 {
            budgets.push(0);
            continue;
        }
        let lower = (total_incident_capacity / 3).max(1);
        let upper = ((2 * total_incident_capacity) / 3).max(lower);
        budgets.push(rng.gen_range(lower..=upper));
    }

    WeightedCase {
        name: name.to_owned(),
        capacities: build_capacity_graph(left_size + right_size, &edges),
        costs: build_cost_graph(left_size + right_size, &edges),
        budgets,
    }
}

fn make_tree_case(
    name: &str,
    seed: u64,
    n: usize,
    max_capacity: usize,
    max_cost: i64,
) -> WeightedCase {
    assert!(n >= 2);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut incident_capacity = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    for vertex in 1..n {
        let parent = rng.gen_range(0..vertex);
        let capacity = rng.gen_range(1..=max_capacity);
        let cost = rng.gen_range(0..=max_cost);
        incident_capacity[parent] += capacity;
        incident_capacity[vertex] += capacity;
        edges.push((parent, vertex, capacity, cost));
    }

    let mut budgets = Vec::with_capacity(n);
    for &total_incident_capacity in &incident_capacity {
        if total_incident_capacity == 0 {
            budgets.push(0);
            continue;
        }
        let lower = (total_incident_capacity / 3).max(1);
        let upper = ((2 * total_incident_capacity) / 3).max(lower);
        budgets.push(rng.gen_range(lower..=upper));
    }

    WeightedCase {
        name: name.to_owned(),
        capacities: build_capacity_graph(n, &edges),
        costs: build_cost_graph(n, &edges),
        budgets,
    }
}

fn benchmark_cases() -> Vec<WeightedCase> {
    vec![
        make_tree_case("tree_n256_cap4", 0xBADA_55F0, 256, 4, 24),
        make_bipartite_case("bipartite_n128_m512_cap4", 0xBADA_55B0, 64, 64, 512, 4, 24),
        make_weighted_case("n48_m144_cap4", 0xBADA_5501, 48, 144, 4, 20),
        make_weighted_case("n64_m256_cap4", 0xBADA_5502, 64, 256, 4, 24),
        make_weighted_case("n96_m384_cap4", 0xBADA_5503, 96, 384, 4, 30),
        make_disconnected_union_case(
            "union_2xn48_m144_cap4",
            &make_weighted_case("left", 0xBADA_5511, 48, 144, 4, 20),
            &make_weighted_case("right", 0xBADA_5512, 48, 144, 4, 20),
        ),
    ]
}

fn make_disconnected_union_case(
    name: &str,
    left: &WeightedCase,
    right: &WeightedCase,
) -> WeightedCase {
    let left_n = left.capacities.number_of_rows();
    let right_n = right.capacities.number_of_rows();
    let total_n = left_n + right_n;

    let mut edges = Vec::new();
    for row in left.capacities.row_indices() {
        let u = row;
        for (column, capacity) in
            left.capacities.sparse_row(row).zip(left.capacities.sparse_row_values(row))
        {
            let v = column;
            if v <= u {
                continue;
            }
            let cost = left.costs.sparse_value_at(u, v).expect("left cost must exist");
            edges.push((u, v, capacity, cost));
        }
    }
    for row in right.capacities.row_indices() {
        let u = row;
        for (column, capacity) in
            right.capacities.sparse_row(row).zip(right.capacities.sparse_row_values(row))
        {
            let v = column;
            if v <= u {
                continue;
            }
            let cost = right.costs.sparse_value_at(u, v).expect("right cost must exist");
            edges.push((left_n + u, left_n + v, capacity, cost));
        }
    }

    let mut budgets = left.budgets.clone();
    budgets.extend_from_slice(&right.budgets);

    WeightedCase {
        name: name.to_owned(),
        capacities: build_capacity_graph(total_n, &edges),
        costs: build_cost_graph(total_n, &edges),
        budgets,
    }
}

fn bench_weighted_solver(c: &mut Criterion) {
    let cases = benchmark_cases();

    let mut group = c.benchmark_group("minimum_cost_balanced_flow");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("KocayBaseline", &case.name), case, |b, case| {
            b.iter(|| {
                let flow = case.capacities.kocay(black_box(&case.budgets));
                black_box(flow);
            });
        });

        group.bench_with_input(BenchmarkId::new("Weighted", &case.name), case, |b, case| {
            b.iter(|| {
                let flow = case
                    .capacities
                    .minimum_cost_balanced_flow(black_box(&case.budgets), black_box(&case.costs));
                black_box(flow);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_weighted_solver);
criterion_main!(benches);
