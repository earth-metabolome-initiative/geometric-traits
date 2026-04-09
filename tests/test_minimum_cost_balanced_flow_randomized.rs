//! Randomized differential tests for minimum-cost maximum balanced flow.
#![cfg(feature = "std")]

#[path = "support/minimum_cost_balanced_flow_oracle.rs"]
mod minimum_cost_balanced_flow_oracle;

use minimum_cost_balanced_flow_oracle::{
    WeightedEdge, assert_solver_matches_oracle, solve_weighted_flow, validate_flow,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn shuffle<T>(values: &mut [T], rng: &mut SmallRng) {
    if values.len() < 2 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.gen_range(0..=i);
        values.swap(i, j);
    }
}

fn sample_instance(
    rng: &mut SmallRng,
    max_vertices: usize,
    max_edges: usize,
    max_capacity: usize,
    max_budget: usize,
    max_cost: i64,
) -> (usize, Vec<WeightedEdge>, Vec<usize>) {
    let n = rng.gen_range(2..=max_vertices);
    let mut pairs = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            pairs.push((u, v));
        }
    }
    shuffle(&mut pairs, rng);

    let edge_count = rng.gen_range(0..=pairs.len().min(max_edges));
    let mut edges = Vec::with_capacity(edge_count);
    for &(u, v) in pairs.iter().take(edge_count) {
        let capacity = rng.gen_range(1..=max_capacity);
        let cost = rng.gen_range(0..=max_cost);
        edges.push((u, v, capacity, cost));
    }

    let budgets = (0..n).map(|_| rng.gen_range(0..=max_budget)).collect();
    (n, edges, budgets)
}

fn relabel_instance(
    n: usize,
    edges: &[WeightedEdge],
    budgets: &[usize],
    permutation: &[usize],
) -> (Vec<WeightedEdge>, Vec<usize>) {
    let mut relabeled_edges = Vec::with_capacity(edges.len());
    for &(u, v, capacity, cost) in edges {
        let mut left = permutation[u];
        let mut right = permutation[v];
        if left > right {
            core::mem::swap(&mut left, &mut right);
        }
        relabeled_edges.push((left, right, capacity, cost));
    }
    relabeled_edges.sort_unstable();

    let mut relabeled_budgets = vec![0usize; n];
    for old_vertex in 0..n {
        relabeled_budgets[permutation[old_vertex]] = budgets[old_vertex];
    }
    (relabeled_edges, relabeled_budgets)
}

#[test]
fn test_random_small_instances_match_oracle() {
    let mut rng = SmallRng::seed_from_u64(0x5eed_cafe);

    for _case in 0..250 {
        let (n, edges, budgets) = sample_instance(&mut rng, 6, 7, 2, 3, 9);
        assert_solver_matches_oracle(n, &edges, &budgets);
    }
}

#[test]
fn test_random_higher_capacity_instances_match_oracle() {
    let mut rng = SmallRng::seed_from_u64(0xface_b00c);

    for _case in 0..150 {
        let (n, edges, budgets) = sample_instance(&mut rng, 5, 6, 3, 4, 12);
        assert_solver_matches_oracle(n, &edges, &budgets);
    }
}

#[test]
fn test_random_relabeling_preserves_objective() {
    let mut rng = SmallRng::seed_from_u64(0x0dec_afba_d5e5);

    for _case in 0..120 {
        let (n, edges, budgets) = sample_instance(&mut rng, 6, 7, 2, 3, 9);
        let flow = solve_weighted_flow(n, &edges, &budgets);
        let objective = validate_flow(n, &edges, &budgets, &flow);

        let mut permutation: Vec<usize> = (0..n).collect();
        shuffle(&mut permutation, &mut rng);
        let (relabeled_edges, relabeled_budgets) =
            relabel_instance(n, &edges, &budgets, &permutation);
        let relabeled_flow = solve_weighted_flow(n, &relabeled_edges, &relabeled_budgets);
        let relabeled_objective =
            validate_flow(n, &relabeled_edges, &relabeled_budgets, &relabeled_flow);

        assert_eq!(
            objective, relabeled_objective,
            "objective should be invariant under relabeling"
        );
    }
}
