use alloc::{vec, vec::Vec};

use super::shared::OriginalEdge;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TreeObjective {
    flow: usize,
    cost: i64,
}

#[inline]
fn prefer_tree_objective(candidate: TreeObjective, incumbent: Option<TreeObjective>) -> bool {
    incumbent.is_none_or(|best| {
        candidate.flow > best.flow || (candidate.flow == best.flow && candidate.cost < best.cost)
    })
}

#[inline]
fn combine_tree_objectives(left: TreeObjective, right: TreeObjective) -> TreeObjective {
    TreeObjective { flow: left.flow + right.flow, cost: left.cost + right.cost }
}

/// Reconstruction context for the tree dynamic program.
///
/// `subtree_dp[vertex][parent_flow]` stores the best lexicographic objective
/// reachable inside the subtree rooted at `vertex` once `parent_flow` units are
/// reserved for the edge connecting `vertex` to its parent. Reconstruction
/// walks the tree top-down and replays the knapsack choices that produced the
/// optimal state.
struct TreeReconstructionContext<'a> {
    parent: &'a [usize],
    adjacency: &'a [Vec<(usize, usize)>],
    original_edges: &'a [OriginalEdge],
    budgets: &'a [usize],
    subtree_dp: &'a [Vec<Option<TreeObjective>>],
    assigned_flow_per_edge: &'a mut [usize],
}

impl TreeReconstructionContext<'_> {
    /// Replays the dynamic-program decisions for the subtree rooted at
    /// `vertex`, assuming that `parent_flow` units are committed to the parent
    /// edge of `vertex`.
    fn reconstruct(&mut self, vertex: usize, parent_flow: usize) {
        let mut children = Vec::new();
        for &(child, edge_index) in &self.adjacency[vertex] {
            if self.parent[child] == vertex {
                children.push((child, edge_index));
            }
        }

        let mut states = vec![None; self.budgets[vertex] + 1];
        states[0] = Some(TreeObjective::default());
        let mut choices: Vec<Vec<Option<(usize, usize)>>> =
            vec![vec![None; self.budgets[vertex] + 1]; children.len() + 1];

        for (child_offset, &(child, edge_index)) in children.iter().enumerate() {
            let edge = self.original_edges[edge_index];
            let max_child_flow = self.budgets[child].min(edge.capacity);
            let mut child_options = vec![None; max_child_flow + 1];
            for (assigned_flow, slot) in child_options.iter_mut().enumerate() {
                if let Some(child_objective) = self.subtree_dp[child][assigned_flow] {
                    *slot = Some(TreeObjective {
                        flow: assigned_flow + child_objective.flow,
                        cost: edge.cost
                            * i64::try_from(assigned_flow)
                                .expect("tree assigned flow must fit into i64")
                            + child_objective.cost,
                    });
                }
            }

            let mut next_states = vec![None; self.budgets[vertex] + 1];
            for used_budget in 0..=self.budgets[vertex] {
                let Some(current_objective) = states[used_budget] else {
                    continue;
                };
                for (assigned_flow, child_objective) in child_options.iter().enumerate() {
                    if used_budget + assigned_flow > self.budgets[vertex] {
                        break;
                    }
                    let Some(child_objective) = child_objective else {
                        continue;
                    };
                    let candidate = combine_tree_objectives(current_objective, *child_objective);
                    if prefer_tree_objective(candidate, next_states[used_budget + assigned_flow]) {
                        next_states[used_budget + assigned_flow] = Some(candidate);
                        choices[child_offset + 1][used_budget + assigned_flow] =
                            Some((used_budget, assigned_flow));
                    }
                }
            }
            states = next_states;
        }

        let mut selected_used_budget = 0usize;
        let mut best = None;
        for (used_budget, candidate) in
            states.iter().copied().enumerate().take((self.budgets[vertex] - parent_flow) + 1)
        {
            let Some(candidate) = candidate else {
                continue;
            };
            if prefer_tree_objective(candidate, best) {
                best = Some(candidate);
                selected_used_budget = used_budget;
            }
        }

        for child_offset in (0..children.len()).rev() {
            let (previous_used_budget, assigned_flow) = choices[child_offset + 1]
                [selected_used_budget]
                .expect("tree reconstruction requires a recorded knapsack choice");
            let (child, edge_index) = children[child_offset];
            self.assigned_flow_per_edge[edge_index] = assigned_flow;
            self.reconstruct(child, assigned_flow);
            selected_used_budget = previous_used_budget;
        }
    }
}

/// Solves a connected tree component exactly by dynamic programming.
///
/// Root the tree at vertex `0`. For every vertex, the DP enumerates how much
/// flow is committed to the edge towards its parent and then solves a knapsack
/// over the children:
/// - choosing a child-edge flow consumes local budget,
/// - contributes that flow to the objective value,
/// - and adds the corresponding edge cost plus the best achievable objective in
///   the child's subtree.
///
/// Objectives are compared lexicographically: higher total flow wins first,
/// lower total cost breaks ties. After the bottom-up pass, reconstruction
/// replays the recorded choices and returns only the positive-flow edges.
///
/// The helper assumes that `original_edges` describe a connected tree on
/// `number_of_vertices` vertices. Disconnected input is a caller bug and
/// triggers a panic.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_lines)]
pub(super) fn tree_minimum_cost_balanced_flow(
    number_of_vertices: usize,
    original_edges: &[OriginalEdge],
    budgets: &[usize],
) -> Vec<(usize, usize, usize)> {
    if number_of_vertices == 0 {
        return Vec::new();
    }

    let mut adjacency = vec![Vec::new(); number_of_vertices];
    for (edge_index, edge) in original_edges.iter().enumerate() {
        adjacency[edge.u].push((edge.v, edge_index));
        adjacency[edge.v].push((edge.u, edge_index));
    }

    let mut parent = vec![usize::MAX; number_of_vertices];
    let mut order = Vec::with_capacity(number_of_vertices);
    let mut stack = vec![0usize];
    parent[0] = 0;

    while let Some(vertex) = stack.pop() {
        order.push(vertex);
        for &(neighbour, _) in &adjacency[vertex] {
            if parent[neighbour] != usize::MAX {
                continue;
            }
            parent[neighbour] = vertex;
            stack.push(neighbour);
        }
    }

    assert!(
        order.len() == number_of_vertices,
        "tree dynamic program requires a connected support graph"
    );

    let mut subtree_dp: Vec<Vec<Option<TreeObjective>>> =
        budgets.iter().map(|&budget| vec![None; budget + 1]).collect();

    for &vertex in order.iter().rev() {
        let mut states = vec![None; budgets[vertex] + 1];
        states[0] = Some(TreeObjective::default());

        for &(child, edge_index) in &adjacency[vertex] {
            if parent[child] != vertex {
                continue;
            }

            let edge = original_edges[edge_index];
            let max_child_flow = budgets[child].min(edge.capacity);
            let mut child_options = vec![None; max_child_flow + 1];
            for (assigned_flow, slot) in child_options.iter_mut().enumerate() {
                if let Some(child_objective) = subtree_dp[child][assigned_flow] {
                    *slot = Some(TreeObjective {
                        flow: assigned_flow + child_objective.flow,
                        cost: edge.cost
                            * i64::try_from(assigned_flow)
                                .expect("tree assigned flow must fit into i64")
                            + child_objective.cost,
                    });
                }
            }

            let mut next_states = vec![None; budgets[vertex] + 1];
            for used_budget in 0..=budgets[vertex] {
                let Some(current_objective) = states[used_budget] else {
                    continue;
                };
                for (assigned_flow, child_objective) in child_options.iter().enumerate() {
                    if used_budget + assigned_flow > budgets[vertex] {
                        break;
                    }
                    let Some(child_objective) = child_objective else {
                        continue;
                    };
                    let candidate = combine_tree_objectives(current_objective, *child_objective);
                    if prefer_tree_objective(candidate, next_states[used_budget + assigned_flow]) {
                        next_states[used_budget + assigned_flow] = Some(candidate);
                    }
                }
            }
            states = next_states;
        }

        let mut best_up_to_budget = vec![None; budgets[vertex] + 1];
        let mut best = None;
        for used_budget in 0..=budgets[vertex] {
            if let Some(candidate) = states[used_budget]
                && prefer_tree_objective(candidate, best)
            {
                best = Some(candidate);
            }
            best_up_to_budget[used_budget] = best;
        }

        for parent_flow in 0..=budgets[vertex] {
            subtree_dp[vertex][parent_flow] = best_up_to_budget[budgets[vertex] - parent_flow];
        }
    }

    let mut assigned_flow_per_edge = vec![0usize; original_edges.len()];
    TreeReconstructionContext {
        parent: &parent,
        adjacency: &adjacency,
        original_edges,
        budgets,
        subtree_dp: &subtree_dp,
        assigned_flow_per_edge: &mut assigned_flow_per_edge,
    }
    .reconstruct(0, 0);

    let mut flow = Vec::with_capacity(original_edges.len());
    for (edge_index, assigned_flow) in assigned_flow_per_edge.into_iter().enumerate() {
        if assigned_flow == 0 {
            continue;
        }
        let edge = original_edges[edge_index];
        flow.push((edge.u, edge.v, assigned_flow));
    }
    flow
}
