//! Shared helpers and small-instance oracle for minimum-cost maximum balanced
//! flow tests.
#![allow(dead_code)]

use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use geometric_traits::{impls::ValuedCSR2D, prelude::*};
use serde::{Deserialize, Serialize};

pub type CapacityGraph = ValuedCSR2D<usize, usize, usize, usize>;
pub type CostGraph = ValuedCSR2D<usize, usize, usize, i64>;
pub type WeightedEdge = (usize, usize, usize, i64);
pub type WeightedFlow = (usize, usize, usize);

#[derive(Debug, Deserialize)]
pub struct HighsOracleResult {
    pub total_flow: usize,
    pub total_cost: i64,
    pub assignment: Vec<usize>,
}

#[derive(Serialize)]
struct HighsOracleRequest<'a> {
    n: usize,
    budgets: &'a [usize],
    edges: Vec<[i64; 4]>,
}

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

pub fn highs_oracle_available() -> bool {
    Command::new("python")
        .args(["-c", "import highspy"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn highs_oracle_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("support")
        .join("highs_weighted_flow_oracle.py")
}

pub fn solve_with_highs(
    n: usize,
    edges: &[WeightedEdge],
    budgets: &[usize],
) -> Result<HighsOracleResult, String> {
    let request = HighsOracleRequest {
        n,
        budgets,
        edges: edges
            .iter()
            .map(|&(u, v, capacity, cost)| {
                [
                    i64::try_from(u).unwrap(),
                    i64::try_from(v).unwrap(),
                    i64::try_from(capacity).unwrap(),
                    cost,
                ]
            })
            .collect(),
    };
    let request_json = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
    let output = Command::new("python")
        .arg(highs_oracle_script())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;

            child.stdin.as_mut().expect("stdin should be available").write_all(&request_json)?;
            child.wait_with_output()
        })
        .map_err(|error| error.to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("HiGHS oracle failed: {}", stderr.trim()));
    }

    serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "failed to parse HiGHS oracle output: {error}; stdout was: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

pub fn assert_solver_matches_highs(n: usize, edges: &[WeightedEdge], budgets: &[usize]) {
    if !highs_oracle_available() {
        eprintln!("skipping HiGHS oracle check because python highspy is unavailable");
        return;
    }

    let flow = solve_weighted_flow(n, edges, budgets);
    let (solver_total, solver_cost) = validate_flow(n, edges, budgets, &flow);
    let oracle =
        solve_with_highs(n, edges, budgets).expect("HiGHS oracle should solve the instance");

    assert_eq!(
        (solver_total, solver_cost),
        (oracle.total_flow, oracle.total_cost),
        "solver objective does not match HiGHS oracle"
    );

    if !oracle.assignment.is_empty() {
        let oracle_flow: Vec<WeightedFlow> = edges
            .iter()
            .zip(oracle.assignment.iter().copied())
            .filter_map(|(&(u, v, _capacity, _cost), assigned)| {
                (assigned > 0).then_some((u.min(v), u.max(v), assigned))
            })
            .collect();
        let oracle_objective = validate_flow(n, edges, budgets, &oracle_flow);
        assert_eq!(
            oracle_objective,
            (oracle.total_flow, oracle.total_cost),
            "HiGHS oracle returned an inconsistent assignment"
        );
    }
}
