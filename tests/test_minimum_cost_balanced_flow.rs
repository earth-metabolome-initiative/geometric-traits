//! Deterministic tests for minimum-cost maximum balanced flow.
#![cfg(feature = "std")]

#[path = "support/minimum_cost_balanced_flow_oracle.rs"]
mod minimum_cost_balanced_flow_oracle;

use geometric_traits::prelude::*;
use minimum_cost_balanced_flow_oracle::*;

#[test]
fn test_triangle_picks_cheapest_edge() {
    let edges = [(0, 1, 1, 9), (0, 2, 1, 1), (1, 2, 1, 4)];
    let budgets = [1, 1, 1];

    let flow = solve_weighted_flow(3, &edges, &budgets);
    assert_eq!(flow, vec![(0, 2, 1)]);
}

#[test]
fn test_k4_picks_cheapest_perfect_matching() {
    let edges =
        [(0, 1, 1, 1), (0, 2, 1, 8), (0, 3, 1, 7), (1, 2, 1, 6), (1, 3, 1, 8), (2, 3, 1, 1)];
    let budgets = [1, 1, 1, 1];

    let flow = solve_weighted_flow(4, &edges, &budgets);
    assert_eq!(flow, vec![(0, 1, 1), (2, 3, 1)]);
}

#[test]
fn test_costs_shape_high_capacity_distribution() {
    let edges = [(0, 1, 2, 1), (1, 2, 2, 10)];
    let budgets = [2, 3, 2];

    let flow = solve_weighted_flow(3, &edges, &budgets);
    assert_eq!(flow, vec![(0, 1, 2), (1, 2, 1)]);
}

#[test]
fn test_triangle_costs_match_bruteforce_for_small_weights() {
    let budgets = [1, 1, 1];

    for cost01 in 0..=3 {
        for cost02 in 0..=3 {
            for cost12 in 0..=3 {
                let edges = [(0, 1, 1, cost01), (0, 2, 1, cost02), (1, 2, 1, cost12)];
                assert_solver_matches_oracle(3, &edges, &budgets);
            }
        }
    }
}

#[test]
fn test_path_costs_match_bruteforce_for_small_weights() {
    let budgets = [2, 3, 2];

    for cost01 in 0..=4 {
        for cost12 in 0..=4 {
            let edges = [(0, 1, 2, cost01), (1, 2, 2, cost12)];
            assert_solver_matches_oracle(3, &edges, &budgets);
        }
    }
}

#[test]
fn test_non_bipartite_case_matches_bruteforce() {
    let edges =
        [(0, 1, 1, 6), (0, 2, 1, 1), (1, 2, 1, 5), (2, 3, 1, 2), (3, 4, 1, 2), (2, 4, 1, 9)];
    let budgets = [1, 1, 2, 1, 1];

    assert_solver_matches_oracle(5, &edges, &budgets);
}

#[test]
fn test_zero_maximum_flow_returns_empty() {
    let edges = [(0, 1, 2, 5)];
    let budgets = [0, 0];

    let flow = solve_weighted_flow(2, &edges, &budgets);
    assert!(flow.is_empty());
}

#[test]
fn test_capacity_edges_require_costs() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let costs: CostGraph = SparseMatrixMut::with_sparse_shape((2, 2));

    let result =
        std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1, 1], &costs));
    assert!(result.is_err());
}

#[test]
fn test_validate_flow_cost_helper() {
    let edges = [(0, 1, 2, 3), (1, 2, 1, 5)];
    let flow = [(0, 1, 2), (1, 2, 1)];
    let (total, cost) = validate_flow(3, &edges, &[2, 3, 1], &flow);
    assert_eq!(total, 3);
    assert_eq!(cost, 11);
    assert_eq!(edge_cost(&edges, 0, 1), 3);
}

#[test]
fn test_solver_uses_all_capacity_when_budgets_allow() {
    let edges = [(0, 1, 2, 5), (1, 2, 1, 7)];
    let budgets = [2, 3, 1];
    let flow = solve_weighted_flow(3, &edges, &budgets);

    assert_eq!(flow, vec![(0, 1, 2), (1, 2, 1)]);
}

#[test]
fn test_solver_handles_uniform_costs() {
    let edges = [(0, 1, 1, 9), (0, 2, 1, 9), (1, 2, 1, 9)];
    let budgets = [1, 1, 1];
    let flow = solve_weighted_flow(3, &edges, &budgets);

    assert_eq!(flow.len(), 1);
    let (total, cost) = validate_flow(3, &edges, &budgets, &flow);
    assert_eq!(total, 1);
    assert_eq!(cost, 9);
}

#[test]
fn test_solver_handles_connected_components() {
    let edges = [(0, 1, 2, 1), (1, 2, 2, 10), (3, 4, 1, 6), (4, 5, 1, 2), (3, 5, 1, 9)];
    let budgets = [2, 3, 2, 1, 1, 1];
    let flow = solve_weighted_flow(6, &edges, &budgets);

    assert_solver_matches_oracle(6, &edges, &budgets);
    assert_eq!(flow, vec![(0, 1, 2), (1, 2, 1), (4, 5, 1)]);
}
