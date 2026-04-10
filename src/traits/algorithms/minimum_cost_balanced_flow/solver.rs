use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{
    bipartite::bipartite_minimum_cost_balanced_flow,
    shared::{
        ExpandedEdge, ExpandedEdgeKind, OriginalEdge, ReducedCostGraph,
        connected_components_from_capacity_matrix, convert_index, find_cross_edge,
        symmetric_cost_at,
    },
    tree::tree_minimum_cost_balanced_flow,
};
use crate::{
    impls::ValuedCSR2D,
    traits::{
        BlossomV, Kocay, MatrixMut, Number, PositiveInteger, SparseMatrixMut, SparseValuedMatrix2D,
    },
};

/// Internal implementation behind
/// [`super::MinimumCostBalancedFlow::minimum_cost_balanced_flow`].
///
/// The function first normalizes and validates the input, then solves the
/// lexicographic objective with a sequence of exact special cases before
/// falling back to the general matching reduction:
/// 1. split disconnected inputs into independent connected components;
/// 2. compute the maximum attainable total flow with [`crate::traits::Kocay`];
/// 3. handle cheap exact cases directly:
///    - zero-flow instances,
///    - connected trees via dynamic programming,
///    - instances where every capacity unit must be used,
///    - uniform-cost instances,
///    - connected bipartite instances via min-cost max-flow;
/// 4. solve the remaining connected non-bipartite case by reducing exact
///    cardinality `b`-matching to weighted perfect matching and calling
///    [`crate::traits::BlossomV`].
///
/// Recursion only happens during connected-component decomposition. Once the
/// input is connected, all work stays within that single component.
#[allow(clippy::too_many_lines)]
pub(super) fn minimum_cost_balanced_flow_impl<M, C>(
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

    // Validate matrix shape invariants once, before any recursive split.
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

    // Both the maximum-flow objective and the minimum-cost tie-break decompose
    // exactly across connected components, so solve each component separately.
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

    // From here on the input is connected. Kocay gives the target total flow
    // value for the lexicographic problem.
    let maximum_flow = capacities.kocay(vertex_budgets);
    let maximum_total_flow: usize = maximum_flow.iter().map(|&(_, _, flow)| flow.as_()).sum();
    if maximum_total_flow == 0 {
        return Vec::new();
    }

    // Normalize each undirected edge to a single record `(u < v)` carrying the
    // capacity and symmetric cost. All remaining specialized solvers consume
    // this representation.
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

    // Trees admit a direct dynamic program and are much cheaper than the
    // general matching reduction.
    if component_count == 1 && original_edges.len() + 1 == n_rows {
        let tree_budgets: Vec<usize> =
            vertex_budgets.iter().map(|budget| (*budget).as_()).collect();
        return tree_minimum_cost_balanced_flow(n_rows, &original_edges, &tree_budgets)
            .into_iter()
            .map(|(u, v, flow)| {
                (
                    convert_index::<M::RowIndex>(u),
                    convert_index::<M::ColumnIndex>(v),
                    convert_index::<M::Value>(flow),
                )
            })
            .collect();
    }

    // If the maximum-flow solution already uses every capacity unit, then the
    // cost tie-break is vacuous and each edge must be fully saturated.
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

    // Uniform costs also make the second stage vacuous, so Kocay's solution is
    // already lexicographically optimal.
    if all_costs_equal {
        return maximum_flow;
    }

    // Connected bipartite instances are solved exactly via min-cost max-flow,
    // avoiding the heavier Blossom-based reduction.
    if let Some(bipartite_coloring) =
        super::super::bipartite_detection::sparse_matrix_bipartite_coloring(capacities)
    {
        let bipartite_budgets: Vec<usize> =
            vertex_budgets.iter().map(|budget| (*budget).as_()).collect();
        let flow = bipartite_minimum_cost_balanced_flow(
            n_rows,
            &original_edges,
            &bipartite_budgets,
            &bipartite_coloring,
        );
        let recovered_total_flow: usize =
            flow.iter().map(|&(_, _, assigned_flow)| assigned_flow).sum();
        assert!(
            recovered_total_flow == maximum_total_flow,
            "bipartite fast path recovered flow {recovered_total_flow} != maximum flow {maximum_total_flow}"
        );
        return flow
            .into_iter()
            .map(|(u, v, assigned_flow)| {
                (
                    convert_index::<M::RowIndex>(u),
                    convert_index::<M::ColumnIndex>(v),
                    convert_index::<M::Value>(assigned_flow),
                )
            })
            .collect();
    }

    let total_budget: usize = vertex_budgets.iter().map(|budget| (*budget).as_()).sum();
    let slack_degree = total_budget - 2 * maximum_total_flow;
    let used_slack_vertex = slack_degree > 0;
    let slack_vertex = n_rows;
    let augmented_order = n_rows + usize::from(used_slack_vertex);

    // General connected non-bipartite case:
    // expand capacities into unit edges and optionally add a slack vertex so
    // the reduction can enforce exact vertex degrees.
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

    // Build the exact-degree gadget. Every incident expanded edge gets an
    // `a`-node, each local degree surplus becomes a `b`-node, zero-cost
    // `a-b` edges encode feasible local choices, and weighted cross edges
    // encode the original edge costs.
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

    // Solve the resulting weighted perfect-matching instance and project the
    // chosen cross edges back onto the original undirected graph.
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
