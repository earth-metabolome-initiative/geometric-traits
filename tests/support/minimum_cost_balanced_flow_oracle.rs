//! Shared helpers and small-instance oracle for minimum-cost maximum balanced
//! flow tests.
#![allow(dead_code)]

use geometric_traits::{impls::ValuedCSR2D, prelude::*};

pub type CapacityGraph = ValuedCSR2D<usize, usize, usize, usize>;
pub type CostGraph = ValuedCSR2D<usize, usize, usize, i64>;
pub type WeightedEdge = (usize, usize, usize, i64);
pub type WeightedFlow = (usize, usize, usize);

pub fn build_capacity_graph(n: usize, edges: &[WeightedEdge]) -> CapacityGraph {
    let mut directed_edges = Vec::new();
    for &(u, v, capacity, _) in edges {
        if u == v || capacity == 0 {
            continue;
        }
        directed_edges.push((u, v, capacity));
        directed_edges.push((v, u, capacity));
    }
    directed_edges.sort_unstable();
    directed_edges.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);

    let mut graph: CapacityGraph =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), directed_edges.len());
    for edge in directed_edges {
        MatrixMut::add(&mut graph, edge).unwrap();
    }
    graph
}

pub fn build_cost_graph(n: usize, edges: &[WeightedEdge]) -> CostGraph {
    let mut directed_edges = Vec::new();
    for &(u, v, capacity, cost) in edges {
        if u == v || capacity == 0 {
            continue;
        }
        directed_edges.push((u, v, cost));
        directed_edges.push((v, u, cost));
    }
    directed_edges.sort_unstable();
    directed_edges.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);

    let mut graph: CostGraph =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), directed_edges.len());
    for edge in directed_edges {
        MatrixMut::add(&mut graph, edge).unwrap();
    }
    graph
}

#[allow(dead_code)]
pub fn edge_cost(edges: &[WeightedEdge], u: usize, v: usize) -> i64 {
    edges
        .iter()
        .find(|&&(left, right, _, _)| (left == u && right == v) || (left == v && right == u))
        .map(|&(_, _, _, cost)| cost)
        .unwrap()
}

pub fn validate_flow(
    n: usize,
    edges: &[WeightedEdge],
    budgets: &[usize],
    flow: &[WeightedFlow],
) -> (usize, i64) {
    let mut used_budget = vec![0usize; n];
    let mut total_flow = 0usize;
    let mut total_cost = 0i64;

    for &(u, v, assigned) in flow {
        assert!(u < v, "flow triples must satisfy u < v, got ({u}, {v}, {assigned})");
        let &(_, _, capacity, cost) = edges
            .iter()
            .find(|&&(left, right, _, _)| (left == u && right == v) || (left == v && right == u))
            .expect("flow must use an original edge");
        assert!(assigned <= capacity, "flow {assigned} exceeds capacity {capacity} on ({u}, {v})");
        used_budget[u] += assigned;
        used_budget[v] += assigned;
        total_flow += assigned;
        total_cost += cost * i64::try_from(assigned).unwrap();
    }

    for (vertex, &used) in used_budget.iter().enumerate() {
        assert!(
            used <= budgets[vertex],
            "vertex {vertex} exceeds budget: {used} > {}",
            budgets[vertex]
        );
    }

    (total_flow, total_cost)
}

pub fn brute_force_best(edges: &[WeightedEdge], budgets: &[usize]) -> (usize, i64, Vec<usize>) {
    fn search(
        edge_index: usize,
        edges: &[WeightedEdge],
        remaining_budgets: &mut [usize],
        current_assignment: &mut Vec<usize>,
        current_flow: usize,
        current_cost: i64,
        best: &mut Option<(usize, i64, Vec<usize>)>,
    ) {
        if edge_index == edges.len() {
            let candidate = (current_flow, current_cost, current_assignment.clone());
            match best {
                Some((best_flow, _, _)) if *best_flow > current_flow => {}
                Some((best_flow, best_cost, _))
                    if *best_flow == current_flow && *best_cost <= current_cost => {}
                _ => *best = Some(candidate),
            }
            return;
        }

        let (u, v, capacity, cost) = edges[edge_index];
        let max_assignable = capacity.min(remaining_budgets[u]).min(remaining_budgets[v]);

        for assigned in 0..=max_assignable {
            remaining_budgets[u] -= assigned;
            remaining_budgets[v] -= assigned;
            current_assignment.push(assigned);
            search(
                edge_index + 1,
                edges,
                remaining_budgets,
                current_assignment,
                current_flow + assigned,
                current_cost + cost * i64::try_from(assigned).unwrap(),
                best,
            );
            current_assignment.pop();
            remaining_budgets[u] += assigned;
            remaining_budgets[v] += assigned;
        }
    }

    let mut remaining_budgets = budgets.to_vec();
    let mut current_assignment = Vec::with_capacity(edges.len());
    let mut best = None;
    search(0, edges, &mut remaining_budgets, &mut current_assignment, 0, 0, &mut best);
    best.expect("the empty assignment is always feasible")
}

pub fn solve_weighted_flow(
    n: usize,
    edges: &[WeightedEdge],
    budgets: &[usize],
) -> Vec<WeightedFlow> {
    let capacities = build_capacity_graph(n, edges);
    let costs = build_cost_graph(n, edges);
    let mut flow = capacities.minimum_cost_balanced_flow(budgets, &costs);
    flow.sort_unstable();
    flow
}

pub fn assert_solver_matches_oracle(n: usize, edges: &[WeightedEdge], budgets: &[usize]) {
    let flow = solve_weighted_flow(n, edges, budgets);
    let (solver_total, solver_cost) = validate_flow(n, edges, budgets, &flow);
    let capacities = build_capacity_graph(n, edges);
    let kocay_total: usize = capacities.kocay(budgets).iter().map(|&(_, _, value)| value).sum();
    let (best_total, best_cost, _) = brute_force_best(edges, budgets);

    assert_eq!(solver_total, kocay_total, "weighted solver total flow should match Kocay");
    assert_eq!(
        (solver_total, solver_cost),
        (best_total, best_cost),
        "solver objective does not match brute-force optimum"
    );
}
