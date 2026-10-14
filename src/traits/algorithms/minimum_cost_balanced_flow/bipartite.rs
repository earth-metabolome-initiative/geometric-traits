use alloc::{collections::VecDeque, vec, vec::Vec};

use super::shared::OriginalEdge;

#[derive(Clone)]
struct MinCostFlowArc {
    to: usize,
    reverse_arc: usize,
    residual_capacity: usize,
    cost: i64,
}

/// Adds a forward residual arc and its reverse companion to the min-cost-flow
/// network used by the bipartite fast path.
///
/// Returns the index of the forward arc in `graph[source]`, which lets the
/// caller recover the final flow assigned to original problem edges after the
/// residual augmentations finish.
#[inline]
fn add_min_cost_flow_arc(
    graph: &mut [Vec<MinCostFlowArc>],
    source: usize,
    destination: usize,
    capacity: usize,
    cost: i64,
) -> usize {
    let source_index = graph[source].len();
    let destination_index = graph[destination].len();
    graph[source].push(MinCostFlowArc {
        to: destination,
        reverse_arc: destination_index,
        residual_capacity: capacity,
        cost,
    });
    graph[destination].push(MinCostFlowArc {
        to: source,
        reverse_arc: source_index,
        residual_capacity: 0,
        cost: -cost,
    });
    source_index
}

/// Solves a connected bipartite component exactly as ordinary min-cost
/// max-flow.
///
/// The bipartition comes from `bipartite_coloring`: color `0` vertices become
/// the left side, color `1` vertices become the right side. The constructed
/// network is the standard capacitated bipartite `b`-matching formulation:
/// - `source -> left vertex` arcs have capacity equal to the left budget;
/// - `left -> right` arcs correspond to original undirected edges, with the
///   original capacities and costs;
/// - `right vertex -> sink` arcs have capacity equal to the right budget.
///
/// The routine then repeatedly augments along minimum-cost residual paths until
/// no `source -> sink` path remains. Because this helper is only used after the
/// outer solver has established that the support is bipartite, any edge whose
/// endpoints fail to cross the partition is treated as a bug and triggers a
/// panic.
///
/// The returned triples refer to the original undirected edges and keep the
/// original endpoint ordering from `original_edges`.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_lines)]
pub(super) fn bipartite_minimum_cost_balanced_flow(
    number_of_vertices: usize,
    original_edges: &[OriginalEdge],
    budgets: &[usize],
    bipartite_coloring: &[u8],
) -> Vec<(usize, usize, usize)> {
    let left_vertices: Vec<usize> = bipartite_coloring
        .iter()
        .enumerate()
        .filter_map(|(vertex, &color)| (color == 0).then_some(vertex))
        .collect();
    let right_vertices: Vec<usize> = bipartite_coloring
        .iter()
        .enumerate()
        .filter_map(|(vertex, &color)| (color == 1).then_some(vertex))
        .collect();

    let mut network_node_of_vertex = vec![usize::MAX; number_of_vertices];
    let source = 0usize;
    for (offset, &vertex) in left_vertices.iter().enumerate() {
        network_node_of_vertex[vertex] = 1 + offset;
    }
    let right_offset = 1 + left_vertices.len();
    for (offset, &vertex) in right_vertices.iter().enumerate() {
        network_node_of_vertex[vertex] = right_offset + offset;
    }
    let sink = right_offset + right_vertices.len();

    let mut residual_graph = vec![Vec::new(); sink + 1];
    for &vertex in &left_vertices {
        let node = network_node_of_vertex[vertex];
        add_min_cost_flow_arc(&mut residual_graph, source, node, budgets[vertex], 0);
    }
    for &vertex in &right_vertices {
        let node = network_node_of_vertex[vertex];
        add_min_cost_flow_arc(&mut residual_graph, node, sink, budgets[vertex], 0);
    }

    let mut original_arc_locations = Vec::with_capacity(original_edges.len());
    for edge in original_edges {
        let (left_vertex, right_vertex) =
            match (bipartite_coloring[edge.u], bipartite_coloring[edge.v]) {
                (0, 1) => (edge.u, edge.v),
                (1, 0) => (edge.v, edge.u),
                _ => panic!("bipartite fast path requires every edge to cross the partition"),
            };
        let left_node = network_node_of_vertex[left_vertex];
        let right_node = network_node_of_vertex[right_vertex];
        let arc_index = add_min_cost_flow_arc(
            &mut residual_graph,
            left_node,
            right_node,
            edge.capacity,
            edge.cost,
        );
        original_arc_locations.push((left_node, arc_index));
    }

    let mut parent_vertex = vec![usize::MAX; residual_graph.len()];
    let mut parent_arc = vec![usize::MAX; residual_graph.len()];
    let mut distances = vec![i64::MAX; residual_graph.len()];
    let mut queued = vec![false; residual_graph.len()];

    loop {
        parent_vertex.fill(usize::MAX);
        parent_arc.fill(usize::MAX);
        distances.fill(i64::MAX);
        queued.fill(false);
        distances[source] = 0;

        let mut frontier = VecDeque::new();
        frontier.push_back(source);
        queued[source] = true;

        while let Some(vertex) = frontier.pop_front() {
            queued[vertex] = false;
            let distance = distances[vertex];

            for (arc_index, arc) in residual_graph[vertex].iter().enumerate() {
                if arc.residual_capacity == 0 {
                    continue;
                }
                let candidate = distance + arc.cost;
                if candidate < distances[arc.to] {
                    distances[arc.to] = candidate;
                    parent_vertex[arc.to] = vertex;
                    parent_arc[arc.to] = arc_index;
                    if !queued[arc.to] {
                        frontier.push_back(arc.to);
                        queued[arc.to] = true;
                    }
                }
            }
        }

        if distances[sink] == i64::MAX {
            break;
        }

        let mut augmenting_flow = usize::MAX;
        let mut vertex = sink;
        while vertex != source {
            let predecessor = parent_vertex[vertex];
            let arc_index = parent_arc[vertex];
            augmenting_flow =
                augmenting_flow.min(residual_graph[predecessor][arc_index].residual_capacity);
            vertex = predecessor;
        }

        let mut vertex = sink;
        while vertex != source {
            let predecessor = parent_vertex[vertex];
            let arc_index = parent_arc[vertex];
            let reverse_arc = residual_graph[predecessor][arc_index].reverse_arc;
            residual_graph[predecessor][arc_index].residual_capacity -= augmenting_flow;
            residual_graph[vertex][reverse_arc].residual_capacity += augmenting_flow;
            vertex = predecessor;
        }
    }

    let mut flow = Vec::with_capacity(original_edges.len());
    for (edge_index, &(left_node, arc_index)) in original_arc_locations.iter().enumerate() {
        let edge = original_edges[edge_index];
        let assigned_flow = edge.capacity - residual_graph[left_node][arc_index].residual_capacity;
        if assigned_flow == 0 {
            continue;
        }
        flow.push((edge.u, edge.v, assigned_flow));
    }
    flow
}
