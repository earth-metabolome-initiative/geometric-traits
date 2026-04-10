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
fn test_bipartite_case_picks_cheapest_maximum_matching() {
    let edges = [(0, 2, 1, 1), (0, 3, 1, 8), (1, 2, 1, 7), (1, 3, 1, 2)];
    let budgets = [1, 1, 1, 1];

    let flow = solve_weighted_flow(4, &edges, &budgets);
    assert_eq!(flow, vec![(0, 2, 1), (1, 3, 1)]);
}

#[test]
fn test_non_tree_bipartite_case_matches_bruteforce() {
    let edges =
        [(0, 3, 2, 3), (0, 4, 1, -1), (1, 3, 1, 2), (1, 5, 2, 4), (2, 4, 2, 1), (2, 5, 1, 0)];
    let budgets = [2, 2, 2, 2, 2, 2];

    assert_solver_matches_oracle(6, &edges, &budgets);
}

#[test]
fn test_tree_case_matches_bruteforce() {
    let edges = [(0, 1, 2, 4), (1, 2, 1, 1), (1, 3, 2, 6), (3, 4, 1, 2), (3, 5, 2, 3)];
    let budgets = [2, 3, 1, 3, 1, 2];

    assert_solver_matches_oracle(6, &edges, &budgets);
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
fn test_one_sided_cost_entry_is_accepted() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let mut costs: CostGraph = SparseMatrixMut::with_sparse_shaped_capacity((2, 2), 1);
    MatrixMut::add(&mut costs, (0usize, 1usize, 5i64)).unwrap();

    let flow = capacities.minimum_cost_balanced_flow(&[1, 1], &costs);
    assert_eq!(flow, vec![(0, 1, 1)]);
}

#[test]
fn test_asymmetric_costs_panic() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let mut costs: CostGraph = SparseMatrixMut::with_sparse_shaped_capacity((2, 2), 2);
    MatrixMut::add(&mut costs, (0usize, 1usize, 4i64)).unwrap();
    MatrixMut::add(&mut costs, (1usize, 0usize, 7i64)).unwrap();

    let result =
        std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1, 1], &costs));
    assert!(result.is_err());
}

#[test]
fn test_non_square_capacity_matrix_panics() {
    let mut capacities: CapacityGraph = SparseMatrixMut::with_sparse_shaped_capacity((2, 3), 2);
    MatrixMut::add(&mut capacities, (0usize, 1usize, 1usize)).unwrap();
    MatrixMut::add(&mut capacities, (1usize, 0usize, 1usize)).unwrap();
    let costs = build_cost_graph(2, &[(0, 1, 1, 2)]);

    let result =
        std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1, 1], &costs));
    assert!(result.is_err());
}

#[test]
fn test_non_square_cost_matrix_panics() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let mut costs: CostGraph = SparseMatrixMut::with_sparse_shaped_capacity((2, 3), 1);
    MatrixMut::add(&mut costs, (0usize, 1usize, 5i64)).unwrap();

    let result =
        std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1, 1], &costs));
    assert!(result.is_err());
}

#[test]
fn test_matrix_order_mismatch_panics() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let costs = build_cost_graph(3, &[(0, 1, 1, 5)]);

    let result =
        std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1, 1], &costs));
    assert!(result.is_err());
}

#[test]
fn test_budget_length_mismatch_panics() {
    let capacities = build_capacity_graph(2, &[(0, 1, 1, 0)]);
    let costs = build_cost_graph(2, &[(0, 1, 1, 5)]);

    let result = std::panic::catch_unwind(|| capacities.minimum_cost_balanced_flow(&[1], &costs));
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
fn test_non_tree_solver_uses_all_capacity_when_budgets_allow() {
    let edges = [(0, 1, 1, 4), (0, 2, 1, 9), (1, 2, 1, 1)];
    let budgets = [2, 2, 2];
    let flow = solve_weighted_flow(3, &edges, &budgets);

    assert_eq!(flow, vec![(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
}

#[test]
fn test_explicit_zero_capacity_entries_are_ignored() {
    let mut capacities: CapacityGraph = SparseMatrixMut::with_sparse_shaped_capacity((3, 3), 4);
    MatrixMut::add(&mut capacities, (0usize, 1usize, 0usize)).unwrap();
    MatrixMut::add(&mut capacities, (1usize, 0usize, 0usize)).unwrap();
    MatrixMut::add(&mut capacities, (1usize, 2usize, 1usize)).unwrap();
    MatrixMut::add(&mut capacities, (2usize, 1usize, 1usize)).unwrap();

    let costs = build_cost_graph(3, &[(1, 2, 1, 5)]);
    let flow = capacities.minimum_cost_balanced_flow(&[0, 1, 1], &costs);

    assert_eq!(flow, vec![(1, 2, 1)]);
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

#[test]
fn test_non_bipartite_slack_case_matches_bruteforce() {
    let edges = [(0, 1, 2, 6), (0, 2, 2, 1), (1, 2, 2, 4)];
    let budgets = [3, 3, 3];

    assert_solver_matches_oracle(3, &edges, &budgets);
}
