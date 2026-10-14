use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use super::NodeSorter;
use crate::traits::MonoplexMonopartiteGraph;

/// Strategy used to choose the next traversal seed when a new connected
/// component (or weakly connected region in the directed case) must be
/// started.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum TraversalSeedStrategy {
    /// Pick the smallest unvisited node id.
    #[default]
    NodeIdAscending,
    /// Pick the unvisited node with maximum out-degree, breaking ties by
    /// smaller node id.
    MaxOutDegree,
}

/// Ordering policy for successor expansion during traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum TraversalNeighborOrder {
    /// Visit successors in ascending node-id order.
    #[default]
    NodeIdAscending,
    /// Visit successors in descending node-id order.
    NodeIdDescending,
    /// Visit successors by descending out-degree, breaking ties by smaller node
    /// id.
    OutDegreeDescending,
}

/// Breadth-first traversal ordering with explicit deterministic policies.
///
/// On disconnected graphs, traversal restarts from a fresh seed chosen by the
/// same [`TraversalSeedStrategy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct BfsTraversalSorter {
    seed_strategy: TraversalSeedStrategy,
    neighbor_order: TraversalNeighborOrder,
}

impl BfsTraversalSorter {
    /// Creates a new breadth-first traversal sorter.
    #[inline]
    #[must_use]
    pub const fn new(
        seed_strategy: TraversalSeedStrategy,
        neighbor_order: TraversalNeighborOrder,
    ) -> Self {
        Self { seed_strategy, neighbor_order }
    }
}

/// Depth-first traversal ordering with explicit deterministic policies.
///
/// On disconnected graphs, traversal restarts from a fresh seed chosen by the
/// same [`TraversalSeedStrategy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct DfsTraversalSorter {
    seed_strategy: TraversalSeedStrategy,
    neighbor_order: TraversalNeighborOrder,
}

impl DfsTraversalSorter {
    /// Creates a new depth-first traversal sorter.
    #[inline]
    #[must_use]
    pub const fn new(
        seed_strategy: TraversalSeedStrategy,
        neighbor_order: TraversalNeighborOrder,
    ) -> Self {
        Self { seed_strategy, neighbor_order }
    }
}

fn pick_seed<G>(graph: &G, visited: &[bool], strategy: TraversalSeedStrategy) -> Option<G::NodeId>
where
    G: MonoplexMonopartiteGraph,
{
    let mut best_node: Option<G::NodeId> = None;
    let mut best_degree = 0usize;

    for node in graph.node_ids() {
        let node_index = node.as_();
        if visited[node_index] {
            continue;
        }

        if strategy == TraversalSeedStrategy::NodeIdAscending {
            return Some(node);
        }

        let node_degree = graph.out_degree(node).as_();
        let is_better = best_node.is_none()
            || node_degree > best_degree
            || (node_degree == best_degree
                && node_index < best_node.expect("best node must exist").as_());
        if is_better {
            best_node = Some(node);
            best_degree = node_degree;
        }
    }

    best_node
}

fn ordered_successors<G>(
    graph: &G,
    node: G::NodeId,
    neighbor_order: TraversalNeighborOrder,
) -> Vec<G::NodeId>
where
    G: MonoplexMonopartiteGraph,
{
    let mut successors: Vec<G::NodeId> = graph.successors(node).collect();
    if neighbor_order == TraversalNeighborOrder::NodeIdAscending {
        successors.sort_unstable_by_key(|left| (*left).as_());
        return successors;
    }

    if neighbor_order == TraversalNeighborOrder::NodeIdDescending {
        successors.sort_unstable_by_key(|right| core::cmp::Reverse((*right).as_()));
        return successors;
    }

    successors.sort_unstable_by(|left, right| {
        let left_index = (*left).as_();
        let right_index = (*right).as_();
        graph
            .out_degree(*right)
            .as_()
            .cmp(&graph.out_degree(*left).as_())
            .then_with(|| left_index.cmp(&right_index))
    });
    successors
}

impl<G> NodeSorter<G> for BfsTraversalSorter
where
    G: MonoplexMonopartiteGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        let n = graph.number_of_nodes().as_();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::with_capacity(n);
        let mut order = Vec::with_capacity(n);

        while let Some(seed) = pick_seed(graph, &visited, self.seed_strategy) {
            let seed_index = seed.as_();
            visited[seed_index] = true;
            queue.push_back(seed);

            while let Some(node) = queue.pop_front() {
                order.push(node);

                for successor in ordered_successors(graph, node, self.neighbor_order) {
                    let successor_index = successor.as_();
                    if visited[successor_index] {
                        continue;
                    }
                    visited[successor_index] = true;
                    queue.push_back(successor);
                }
            }
        }

        order
    }
}

impl<G> NodeSorter<G> for DfsTraversalSorter
where
    G: MonoplexMonopartiteGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        let n = graph.number_of_nodes().as_();
        let mut visited = vec![false; n];
        let mut stack = Vec::with_capacity(n);
        let mut order = Vec::with_capacity(n);

        while let Some(seed) = pick_seed(graph, &visited, self.seed_strategy) {
            let seed_index = seed.as_();
            visited[seed_index] = true;
            stack.push(seed);

            while let Some(node) = stack.pop() {
                order.push(node);

                let successors = ordered_successors(graph, node, self.neighbor_order);
                for successor in successors.into_iter().rev() {
                    let successor_index = successor.as_();
                    if visited[successor_index] {
                        continue;
                    }
                    visited[successor_index] = true;
                    stack.push(successor);
                }
            }
        }

        order
    }
}
