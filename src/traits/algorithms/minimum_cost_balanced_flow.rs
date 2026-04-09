//! Minimum-cost maximum balanced flow in general graphs.
//!
//! This solver combines two existing building blocks from the crate:
//! - [`crate::traits::Kocay`] for the maximum feasible total flow.
//! - [`crate::traits::BlossomV`] for a weighted perfect-matching reduction.
//!
//! Given integer edge capacities, vertex budgets, and per-edge costs, it first
//! computes the maximum total flow value `k`. It then reduces the exact
//! cardinality-`k` capacitated `b`-matching problem to a minimum-cost perfect
//! matching instance by:
//! - expanding each unit of edge capacity into a unit edge copy,
//! - adding a slack vertex to absorb unused budget,
//! - enforcing exact degrees through a standard exact `b`-matching gadget, and
//! - solving the resulting weighted perfect matching with Blossom V.
//!
//! The returned edge flows are integral, maximize total flow, and among those
//! attain minimum total cost.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::{
    impls::ValuedCSR2D,
    traits::{
        BlossomV, Kocay, MatrixMut, Number, PositiveInteger, SparseMatrixMut, SparseValuedMatrix2D,
    },
};

type ReducedCostGraph = ValuedCSR2D<usize, usize, usize, i64>;

#[derive(Clone, Copy)]
enum ExpandedEdgeKind {
    Original(usize),
    Slack,
}

#[derive(Clone, Copy)]
struct ExpandedEdge {
    u: usize,
    v: usize,
    cost: i64,
    kind: ExpandedEdgeKind,
}

#[derive(Clone, Copy)]
struct OriginalEdge {
    u: usize,
    v: usize,
    capacity: usize,
    cost: i64,
}

fn find_cross_edge(cross_edges: &[(usize, usize, usize)], u: usize, v: usize) -> Option<usize> {
    let pair = if u < v { (u, v) } else { (v, u) };
    cross_edges
        .binary_search_by_key(&(pair.0, pair.1), |&(left, right, _)| (left, right))
        .ok()
        .map(|position| cross_edges[position].2)
}

fn convert_index<I: PositiveInteger>(index: usize) -> I {
    I::try_from_usize(index).ok().expect("index must fit into the target integer type")
}

fn symmetric_cost_at<C>(edge_costs: &C, i: usize, j: usize) -> i64
where
    C: SparseValuedMatrix2D,
    C::Value: Number + AsPrimitive<i64>,
    C::RowIndex: PositiveInteger,
    C::ColumnIndex: PositiveInteger,
{
    let i_row = convert_index::<C::RowIndex>(i);
    let j_row = convert_index::<C::RowIndex>(j);
    let i_col = convert_index::<C::ColumnIndex>(i);
    let j_col = convert_index::<C::ColumnIndex>(j);

    match (edge_costs.sparse_value_at(i_row, j_col), edge_costs.sparse_value_at(j_row, i_col)) {
        (Some(forward), Some(backward)) => {
            let forward_cost = forward.as_();
            let backward_cost = backward.as_();
            assert!(
                forward_cost == backward_cost,
                "cost matrix is not symmetric on edge ({i}, {j}): {forward_cost} != {backward_cost}"
            );
            forward_cost
        }
        (Some(cost), None) | (None, Some(cost)) => cost.as_(),
        (None, None) => panic!("cost matrix is missing a value for capacity edge ({i}, {j})"),
    }
}

fn connected_components_from_capacity_matrix<M>(capacities: &M) -> (usize, Vec<usize>)
where
    M: SparseValuedMatrix2D,
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    let n_rows: usize = capacities.number_of_rows().as_();
    let mut component_of_vertex = vec![usize::MAX; n_rows];
    let mut stack = Vec::new();
    let mut component_count = 0usize;

    for start in 0..n_rows {
        if component_of_vertex[start] != usize::MAX {
            continue;
        }

        component_of_vertex[start] = component_count;
        stack.push(start);

        while let Some(vertex) = stack.pop() {
            let row = convert_index::<M::RowIndex>(vertex);
            for column in capacities.sparse_row(row) {
                let neighbour = column.as_();
                if component_of_vertex[neighbour] != usize::MAX {
                    continue;
                }
                component_of_vertex[neighbour] = component_count;
                stack.push(neighbour);
            }
        }

        component_count += 1;
    }

    (component_count, component_of_vertex)
}

fn minimum_cost_balanced_flow_impl<M, C>(
    capacities: &M,
    vertex_budgets: &[M::Value],
    edge_costs: &C,
) -> Vec<(M::RowIndex, M::ColumnIndex, M::Value)>
where
    M: SparseValuedMatrix2D,
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
    C: SparseValuedMatrix2D<RowIndex = M::RowIndex, ColumnIndex = M::ColumnIndex>,
    C::Value: Number + AsPrimitive<i64>,
{
    let n_rows: usize = capacities.number_of_rows().as_();
    let n_cols: usize = capacities.number_of_columns().as_();
    let cost_rows: usize = edge_costs.number_of_rows().as_();
    let cost_cols: usize = edge_costs.number_of_columns().as_();

    assert!(
        n_rows == n_cols,
        "MinimumCostBalancedFlow requires a square capacity matrix, got {n_rows} x {n_cols}"
    );
    assert!(
        cost_rows == cost_cols,
        "MinimumCostBalancedFlow requires a square cost matrix, got {cost_rows} x {cost_cols}"
    );
    assert!(n_rows == cost_rows, "capacity matrix order {n_rows} != cost matrix order {cost_rows}");
    assert!(
        vertex_budgets.len() == n_rows,
        "vertex_budgets length {} != matrix order {n_rows}",
        vertex_budgets.len()
    );

    let (component_count, component_of_vertex) =
        connected_components_from_capacity_matrix(capacities);

    if component_count > 1 {
        let mut component_nodes = vec![Vec::new(); component_count];
        for vertex in 0..n_rows {
            component_nodes[component_of_vertex[vertex]].push(vertex);
        }

        let mut global_flow = Vec::new();
        for nodes in component_nodes {
            if nodes.is_empty() {
                continue;
            }

            let component_size = nodes.len();
            let mut local_index_of = vec![usize::MAX; n_rows];
            for (local, &global) in nodes.iter().enumerate() {
                local_index_of[global] = local;
            }

            let mut local_budgets = Vec::with_capacity(component_size);
            for &global in &nodes {
                local_budgets.push(vertex_budgets[global]);
            }

            let mut capacity_edges = Vec::new();
            let mut cost_edges = Vec::new();
            for &global_u in &nodes {
                let local_u = local_index_of[global_u];
                let row = convert_index::<M::RowIndex>(global_u);
                for (column, capacity) in
                    capacities.sparse_row(row).zip(capacities.sparse_row_values(row))
                {
                    let global_v = column.as_();
                    if global_v <= global_u
                        || component_of_vertex[global_v] != component_of_vertex[global_u]
                    {
                        continue;
                    }
                    let local_v = local_index_of[global_v];
                    let cost = symmetric_cost_at(edge_costs, global_u, global_v);
                    capacity_edges.push((local_u, local_v, capacity));
                    cost_edges.push((local_u, local_v, cost));
                }
            }

            let mut local_capacities: ValuedCSR2D<usize, usize, usize, M::Value> =
                SparseMatrixMut::with_sparse_shaped_capacity(
                    (component_size, component_size),
                    capacity_edges.len() * 2,
                );
            let mut directed_capacity_edges = Vec::with_capacity(capacity_edges.len() * 2);
            for (u, v, capacity) in capacity_edges {
                directed_capacity_edges.push((u, v, capacity));
                directed_capacity_edges.push((v, u, capacity));
            }
            directed_capacity_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));
            for (u, v, capacity) in directed_capacity_edges {
                MatrixMut::add(&mut local_capacities, (u, v, capacity))
                    .expect("component capacity insertion should succeed");
            }

            let mut local_costs: ReducedCostGraph = SparseMatrixMut::with_sparse_shaped_capacity(
                (component_size, component_size),
                cost_edges.len() * 2,
            );
            let mut directed_cost_edges = Vec::with_capacity(cost_edges.len() * 2);
            for (u, v, cost) in cost_edges {
                directed_cost_edges.push((u, v, cost));
                directed_cost_edges.push((v, u, cost));
            }
            directed_cost_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));
            for (u, v, cost) in directed_cost_edges {
                MatrixMut::add(&mut local_costs, (u, v, cost))
                    .expect("component cost insertion should succeed");
            }

            let local_flow =
                minimum_cost_balanced_flow_impl(&local_capacities, &local_budgets, &local_costs);
            for (local_u, local_v, flow) in local_flow {
                let global_u = nodes[local_u];
                let global_v = nodes[local_v];
                global_flow.push((
                    convert_index::<M::RowIndex>(global_u),
                    convert_index::<M::ColumnIndex>(global_v),
                    flow,
                ));
            }
        }

        global_flow.sort_unstable();
        return global_flow;
    }

    let maximum_flow = capacities.kocay(vertex_budgets);
    let maximum_total_flow: usize = maximum_flow.iter().map(|&(_, _, flow)| flow.as_()).sum();
    if maximum_total_flow == 0 {
        return Vec::new();
    }

    let mut original_edges = Vec::new();
    let mut uniform_edge_cost = None;
    let mut all_costs_equal = true;
    for row in capacities.row_indices() {
        let i = row.as_();
        for (column, capacity) in capacities.sparse_row(row).zip(capacities.sparse_row_values(row))
        {
            let j = column.as_();
            if j <= i {
                continue;
            }
            let capacity_units = capacity.as_();
            if capacity_units == 0 {
                continue;
            }
            let edge_cost = symmetric_cost_at(edge_costs, i, j);
            if let Some(cost) = uniform_edge_cost {
                all_costs_equal &= cost == edge_cost;
            } else {
                uniform_edge_cost = Some(edge_cost);
            }
            original_edges.push(OriginalEdge {
                u: i,
                v: j,
                capacity: capacity_units,
                cost: edge_cost,
            });
        }
    }
    let expanded_original_edge_units: usize = original_edges.iter().map(|edge| edge.capacity).sum();

    if maximum_total_flow == expanded_original_edge_units {
        let flow = original_edges
            .into_iter()
            .map(|edge| {
                (
                    convert_index::<M::RowIndex>(edge.u),
                    convert_index::<M::ColumnIndex>(edge.v),
                    convert_index::<M::Value>(edge.capacity),
                )
            })
            .collect();
        return flow;
    }

    if all_costs_equal {
        return maximum_flow;
    }

    let total_budget: usize = vertex_budgets.iter().map(|budget| (*budget).as_()).sum();
    let slack_degree = total_budget - 2 * maximum_total_flow;
    let used_slack_vertex = slack_degree > 0;
    let slack_vertex = n_rows;
    let augmented_order = n_rows + usize::from(used_slack_vertex);

    let mut expanded_edges: Vec<ExpandedEdge> = Vec::new();
    for (original_index, edge) in original_edges.iter().enumerate() {
        for _ in 0..edge.capacity {
            expanded_edges.push(ExpandedEdge {
                u: edge.u,
                v: edge.v,
                cost: edge.cost,
                kind: ExpandedEdgeKind::Original(original_index),
            });
        }
    }

    if used_slack_vertex {
        for (vertex, budget) in vertex_budgets.iter().enumerate() {
            let slack_copies = budget.as_().min(slack_degree);
            for _ in 0..slack_copies {
                expanded_edges.push(ExpandedEdge {
                    u: vertex,
                    v: slack_vertex,
                    cost: 0,
                    kind: ExpandedEdgeKind::Slack,
                });
            }
        }
    }

    let mut incident_edges: Vec<Vec<usize>> = vec![Vec::new(); augmented_order];
    for (edge_index, edge) in expanded_edges.iter().enumerate() {
        incident_edges[edge.u].push(edge_index);
        incident_edges[edge.v].push(edge_index);
    }

    let mut exact_degree = vec![0usize; augmented_order];
    for vertex in 0..n_rows {
        exact_degree[vertex] = vertex_budgets[vertex].as_();
        assert!(
            incident_edges[vertex].len() >= exact_degree[vertex],
            "vertex {vertex} has budget {} larger than its augmented degree {}",
            exact_degree[vertex],
            incident_edges[vertex].len()
        );
    }
    if used_slack_vertex {
        exact_degree[slack_vertex] = slack_degree;
    }

    let mut endpoint_nodes = vec![[usize::MAX; 2]; expanded_edges.len()];
    let mut a_nodes_by_vertex: Vec<Vec<usize>> = vec![Vec::new(); augmented_order];
    let mut next_node = 0usize;

    for vertex in 0..augmented_order {
        let mut a_nodes = Vec::with_capacity(incident_edges[vertex].len());
        for &edge_index in &incident_edges[vertex] {
            let node = next_node;
            next_node += 1;
            a_nodes.push(node);
            if expanded_edges[edge_index].u == vertex {
                endpoint_nodes[edge_index][0] = node;
            } else {
                endpoint_nodes[edge_index][1] = node;
            }
        }
        a_nodes_by_vertex[vertex] = a_nodes;
    }
    let mut b_nodes_by_vertex: Vec<Vec<usize>> = vec![Vec::new(); augmented_order];
    for vertex in 0..augmented_order {
        let internal_nodes = incident_edges[vertex].len() - exact_degree[vertex];
        let mut b_nodes = Vec::with_capacity(internal_nodes);
        for _ in 0..internal_nodes {
            b_nodes.push(next_node);
            next_node += 1;
        }
        b_nodes_by_vertex[vertex] = b_nodes;
    }
    let mut reduced_edges: Vec<(usize, usize, i64)> = Vec::with_capacity(expanded_edges.len());

    for vertex in 0..augmented_order {
        for &a_node in &a_nodes_by_vertex[vertex] {
            for &b_node in &b_nodes_by_vertex[vertex] {
                reduced_edges.push((a_node, b_node, 0));
            }
        }
    }

    let mut cross_edges: Vec<(usize, usize, usize)> = Vec::with_capacity(expanded_edges.len());
    for (edge_index, edge) in expanded_edges.iter().enumerate() {
        let [left, right] = endpoint_nodes[edge_index];
        assert!(
            left != usize::MAX && right != usize::MAX,
            "every expanded edge endpoint must receive a gadget node"
        );
        reduced_edges.push((left, right, edge.cost));
        let pair = if left < right { (left, right) } else { (right, left) };
        cross_edges.push((pair.0, pair.1, edge_index));
    }
    cross_edges.sort_unstable_by_key(|&(left, right, _)| (left, right));

    let mut directed_reduced_edges = Vec::with_capacity(reduced_edges.len() * 2);
    for (left, right, cost) in reduced_edges {
        directed_reduced_edges.push((left, right, cost));
        directed_reduced_edges.push((right, left, cost));
    }
    directed_reduced_edges.sort_unstable_by_key(|&(left, right, _)| (left, right));

    let mut reduced_graph: ReducedCostGraph = SparseMatrixMut::with_sparse_shaped_capacity(
        (next_node, next_node),
        directed_reduced_edges.len(),
    );
    for edge in directed_reduced_edges {
        MatrixMut::add(&mut reduced_graph, edge)
            .expect("reduced graph edge insertion must succeed");
    }

    let perfect_matching = reduced_graph
        .blossom_v()
        .expect("the exact b-matching reduction should always admit a perfect matching");

    let mut original_flow = vec![0usize; original_edges.len()];
    for (left, right) in perfect_matching {
        if let Some(edge_index) = find_cross_edge(&cross_edges, left, right) {
            if let ExpandedEdgeKind::Original(original_index) = expanded_edges[edge_index].kind {
                original_flow[original_index] += 1;
            }
        }
    }

    let recovered_total_flow: usize = original_flow.iter().sum();
    assert!(
        recovered_total_flow == maximum_total_flow,
        "recovered flow {recovered_total_flow} != maximum flow {maximum_total_flow}"
    );

    let mut flow = Vec::with_capacity(original_edges.len());
    for (edge_index, assigned_flow) in original_flow.into_iter().enumerate() {
        if assigned_flow == 0 {
            continue;
        }
        let edge = &original_edges[edge_index];
        flow.push((
            convert_index::<M::RowIndex>(edge.u),
            convert_index::<M::ColumnIndex>(edge.v),
            convert_index::<M::Value>(assigned_flow),
        ));
    }
    flow
}

/// Minimum-cost maximum balanced flow in general graphs.
///
/// The input capacity matrix represents an undirected graph where each entry
/// `(i, j)` with value `cap` means there is an edge between vertices `i` and
/// `j` with capacity `cap`. The `edge_costs` matrix must use the same support
/// and provide the per-unit cost for every capacity edge.
///
/// Each vertex has an integer budget limiting the sum of incident flow. The
/// returned triples `(row, column, flow)` satisfy:
/// - `row < column`,
/// - `0 < flow <= capacity(row, column)`,
/// - the budget constraint at every vertex,
/// - maximum total flow, and
/// - minimum total cost among all maximum-flow assignments.
pub trait MinimumCostBalancedFlow: SparseValuedMatrix2D + Sized
where
    Self::Value: PositiveInteger,
    Self::RowIndex: PositiveInteger,
    Self::ColumnIndex: PositiveInteger,
{
    /// Computes a minimum-cost maximum balanced flow.
    ///
    /// # Arguments
    ///
    /// * `vertex_budgets` - Budget (maximum total flow) per vertex.
    /// * `edge_costs` - Symmetric cost matrix with the same support as the
    ///   capacity matrix on all positive-capacity edges.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - either matrix is not square,
    /// - the two matrix orders differ,
    /// - `vertex_budgets.len()` does not equal the matrix order,
    /// - or a positive-capacity edge is missing a cost.
    fn minimum_cost_balanced_flow<C>(
        &self,
        vertex_budgets: &[Self::Value],
        edge_costs: &C,
    ) -> Vec<(Self::RowIndex, Self::ColumnIndex, Self::Value)>
    where
        C: SparseValuedMatrix2D<RowIndex = Self::RowIndex, ColumnIndex = Self::ColumnIndex>,
        C::Value: Number + AsPrimitive<i64>,
    {
        minimum_cost_balanced_flow_impl(self, vertex_budgets, edge_costs)
    }
}

impl<M> MinimumCostBalancedFlow for M
where
    M: SparseValuedMatrix2D,
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
}
