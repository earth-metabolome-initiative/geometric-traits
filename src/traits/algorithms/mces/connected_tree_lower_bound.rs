use alloc::{collections::BTreeMap, vec::Vec};
use core::marker::PhantomData;

use num_traits::AsPrimitive;

use super::super::weighted_assignment::Crouse;
use crate::{
    impls::ValuedCSR2D,
    traits::{MatrixMut, MonoplexMonopartiteGraph, SparseMatrixMut},
};

struct ConnectedTreeData {
    adjacency: Vec<Vec<usize>>,
}

fn connected_tree_data<G>(graph: &G) -> Option<ConnectedTreeData>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize> + Copy,
{
    if graph.has_self_loops() {
        return None;
    }

    let node_ids: Vec<G::NodeId> = graph.node_ids().collect();
    let number_of_nodes = node_ids.len();
    if number_of_nodes == 0 {
        return None;
    }

    let mut adjacency = vec![Vec::new(); number_of_nodes];
    for &node_id in &node_ids {
        let node_index = node_id.as_();
        for neighbour_id in graph.successors(node_id) {
            let neighbour_index = neighbour_id.as_();
            if !graph.has_successor(node_ids[neighbour_index], node_ids[node_index]) {
                return None;
            }
            adjacency[node_index].push(neighbour_index);
        }
        adjacency[node_index].sort_unstable();
    }

    let mut visited = vec![false; number_of_nodes];
    let mut frontier = vec![0usize];
    let mut number_of_visited_nodes = 0usize;
    let mut degree_sum = 0usize;
    visited[0] = true;

    while let Some(node) = frontier.pop() {
        number_of_visited_nodes += 1;
        degree_sum += adjacency[node].len();
        for &neighbour in &adjacency[node] {
            if !visited[neighbour] {
                visited[neighbour] = true;
                frontier.push(neighbour);
            }
        }
    }

    let number_of_edges = degree_sum / 2;
    if number_of_visited_nodes != number_of_nodes || number_of_edges + 1 != number_of_nodes {
        return None;
    }

    Some(ConnectedTreeData { adjacency })
}

fn undirected_edge_key(left: usize, right: usize) -> (usize, usize) {
    if left <= right { (left, right) } else { (right, left) }
}

fn build_edge_index_map<N>(edge_map: &[(N, N)]) -> BTreeMap<(usize, usize), usize>
where
    N: Copy + AsPrimitive<usize>,
{
    let mut edge_index_map = BTreeMap::new();
    for (index, (left, right)) in edge_map.iter().copied().enumerate() {
        edge_index_map.insert(undirected_edge_key(left.as_(), right.as_()), index);
    }
    edge_index_map
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ConnectedTreeEdgeKey {
    first_parent: usize,
    first_node: usize,
    second_parent: usize,
    second_node: usize,
}

struct ConnectedTreeBoundSolver<'a, N>
where
    N: Eq + Copy + Ord + core::fmt::Debug + AsPrimitive<usize>,
{
    adjacency_first: &'a [Vec<usize>],
    adjacency_second: &'a [Vec<usize>],
    edge_index_first: BTreeMap<(usize, usize), usize>,
    edge_index_second: BTreeMap<(usize, usize), usize>,
    second_node_count: usize,
    second_edge_count: usize,
    junction_compatible: &'a [bool],
    edge_pair_allowed: &'a [bool],
    memo: BTreeMap<ConnectedTreeEdgeKey, usize>,
    marker: PhantomData<N>,
}

impl<'a, N> ConnectedTreeBoundSolver<'a, N>
where
    N: Eq + Copy + Ord + core::fmt::Debug + AsPrimitive<usize>,
{
    fn new(
        adjacency_first: &'a [Vec<usize>],
        adjacency_second: &'a [Vec<usize>],
        edge_map_first: &'a [(N, N)],
        edge_map_second: &'a [(N, N)],
        junction_compatible: &'a [bool],
        edge_pair_allowed: &'a [bool],
    ) -> Self {
        Self {
            adjacency_first,
            adjacency_second,
            edge_index_first: build_edge_index_map(edge_map_first),
            edge_index_second: build_edge_index_map(edge_map_second),
            second_node_count: adjacency_second.len(),
            second_edge_count: edge_map_second.len(),
            junction_compatible,
            edge_pair_allowed,
            memo: BTreeMap::new(),
            marker: PhantomData,
        }
    }

    fn solve(&mut self, edge_map_first: &[(N, N)], edge_map_second: &[(N, N)]) -> usize {
        let mut best = 0usize;

        for &(first_src, first_dst) in edge_map_first {
            let first_src = first_src.as_();
            let first_dst = first_dst.as_();
            for &(second_src, second_dst) in edge_map_second {
                let second_src = second_src.as_();
                let second_dst = second_dst.as_();
                best = best.max(self.root_edge_score(first_src, first_dst, second_src, second_dst));
                best = best.max(self.root_edge_score(first_src, first_dst, second_dst, second_src));
            }
        }

        best
    }

    fn root_edge_score(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
    ) -> usize {
        #[rustfmt::skip]
        let left_score = self.oriented_edge_score(first_right, first_left, second_right, second_left);
        if left_score == 0 {
            return 0;
        }
        #[rustfmt::skip]
        let right_score = self.oriented_edge_score(first_left, first_right, second_left, second_right);
        left_score + right_score - 1
    }

    fn oriented_edge_score(
        &mut self,
        first_parent: usize,
        first_node: usize,
        second_parent: usize,
        second_node: usize,
    ) -> usize {
        let key = ConnectedTreeEdgeKey { first_parent, first_node, second_parent, second_node };
        if let Some(score) = self.memo.get(&key).copied() {
            return score;
        }

        let first_edge_key = undirected_edge_key(first_parent, first_node);
        let Some(&first_edge_index) = self.edge_index_first.get(&first_edge_key) else {
            self.memo.insert(key, 0);
            return 0;
        };
        let second_edge_key = undirected_edge_key(second_parent, second_node);
        let Some(&second_edge_index) = self.edge_index_second.get(&second_edge_key) else {
            self.memo.insert(key, 0);
            return 0;
        };
        if !self.edge_pair_allowed[first_edge_index * self.second_edge_count + second_edge_index] {
            self.memo.insert(key, 0);
            return 0;
        }

        let mut score = 1usize;
        if self.junction_compatible[first_node * self.second_node_count + second_node] {
            let first_children: Vec<usize> = self.adjacency_first[first_node]
                .iter()
                .copied()
                .filter(|&child| child != first_parent)
                .collect();
            let second_children: Vec<usize> = self.adjacency_second[second_node]
                .iter()
                .copied()
                .filter(|&child| child != second_parent)
                .collect();

            if !first_children.is_empty() && !second_children.is_empty() {
                let rows = first_children.len();
                let columns = second_children.len();
                let mut child_scores = vec![0usize; rows * columns];
                for (row, &first_child) in first_children.iter().enumerate() {
                    for (column, &second_child) in second_children.iter().enumerate() {
                        #[rustfmt::skip]
                        let child_score = self.oriented_edge_score(first_node, first_child, second_node, second_child);
                        child_scores[row * columns + column] = child_score;
                    }
                }
                score += positive_weight_matching_score(&child_scores, rows, columns);
            }
        }

        self.memo.insert(key, score);
        score
    }
}

#[allow(clippy::cast_precision_loss)]
fn positive_weight_matching_score(scores: &[usize], rows: usize, columns: usize) -> usize {
    debug_assert_eq!(scores.len(), rows * columns);
    if rows == 0 || columns == 0 {
        return 0;
    }

    let max_score =
        scores.iter().copied().max().expect("positive matrix dimensions imply at least one score");
    if max_score == 0 {
        return 0;
    }

    let non_zero_entries = scores.iter().filter(|&&score| score > 0).count();
    let mut matrix: ValuedCSR2D<usize, usize, usize, f64> =
        SparseMatrixMut::with_sparse_shaped_capacity((rows, columns), non_zero_entries);
    let max_real_cost = (max_score + 1) as f64;

    for row in 0..rows {
        for column in 0..columns {
            let score = scores[row * columns + column];
            if score == 0 {
                continue;
            }
            let cost = max_real_cost - score as f64;
            MatrixMut::add(&mut matrix, (row, column, cost))
                .expect("tree matching matrix must be built in sorted row-major order");
        }
    }

    let non_edge_cost = max_real_cost + 1.0;
    let max_cost = non_edge_cost + 1.0;
    matrix
        .crouse(non_edge_cost, max_cost)
        .expect("tree child matching must be feasible")
        .into_iter()
        .map(|(row, column)| scores[row * columns + column])
        .sum()
}

pub(super) fn connected_tree_lower_bound<G>(
    first: &G,
    second: &G,
    edge_map_first: &[(G::NodeId, G::NodeId)],
    edge_map_second: &[(G::NodeId, G::NodeId)],
    junction_compatible: &[bool],
    edge_pair_allowed: &[bool],
) -> Option<usize>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: Eq + Copy + Ord + core::fmt::Debug + AsPrimitive<usize>,
{
    let first_tree = connected_tree_data(first)?;
    let second_tree = connected_tree_data(second)?;
    #[rustfmt::skip]
    let mut solver = ConnectedTreeBoundSolver::new(&first_tree.adjacency, &second_tree.adjacency, edge_map_first, edge_map_second, junction_compatible, edge_pair_allowed);
    Some(solver.solve(edge_map_first, edge_map_second))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        ConnectedTreeBoundSolver, connected_tree_data, connected_tree_lower_bound,
        positive_weight_matching_score,
    };
    use crate::{
        impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
        naive_structs::{DiGraph, GenericVocabularyBuilder, UndiGraph},
        prelude::{DiEdgesBuilder, UndiEdgesBuilder},
        traits::{
            EdgesBuilder, LineGraph, MonopartiteGraph, VocabularyBuilder,
            algorithms::tree_detection::TreeDetection,
        },
    };

    fn build_undigraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
        let node_vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(nodes.len())
            .symbols(nodes.into_iter().enumerate())
            .build()
            .unwrap();
        let edge_mat: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape(node_vocab.len())
            .edges(edges.into_iter())
            .build()
            .unwrap();
        UndiGraph::from((node_vocab, edge_mat))
    }

    fn build_digraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> DiGraph<usize> {
        let node_vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
            .expected_number_of_symbols(nodes.len())
            .symbols(nodes.into_iter().enumerate())
            .build()
            .unwrap();
        let edge_mat: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
            .expected_number_of_edges(edges.len())
            .expected_shape(node_vocab.len())
            .edges(edges.into_iter())
            .build()
            .unwrap();
        DiGraph::from((node_vocab, edge_mat))
    }

    #[test]
    fn test_connected_tree_data_rejects_empty_graph() {
        let graph = build_undigraph(Vec::new(), Vec::new());
        assert!(connected_tree_data(&graph).is_none());
    }

    #[test]
    fn test_connected_tree_data_rejects_self_loop_graph() {
        let graph = build_undigraph(vec![0], vec![(0, 0)]);
        assert!(connected_tree_data(&graph).is_none());
    }

    #[test]
    fn test_connected_tree_data_rejects_asymmetric_graph() {
        let graph = build_digraph(vec![0, 1], vec![(0, 1)]);
        assert!(connected_tree_data(&graph).is_none());
    }

    #[test]
    fn test_connected_tree_data_rejects_disconnected_graph() {
        let graph = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (2, 3)]);
        assert!(connected_tree_data(&graph).is_none());
    }

    #[test]
    fn test_connected_tree_lower_bound_matches_simple_path_overlap() {
        let first = build_undigraph(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
        let second = build_undigraph(vec![0, 1, 2], vec![(0, 1), (1, 2)]);

        assert!(first.is_tree());
        assert!(second.is_tree());

        let first_line_graph = first.line_graph();
        let second_line_graph = second.line_graph();
        let junction_compatible = vec![true; first.number_of_nodes() * second.number_of_nodes()];
        let edge_pair_allowed =
            vec![true; first_line_graph.edge_map().len() * second_line_graph.edge_map().len()];

        assert_eq!(
            connected_tree_lower_bound(
                &first,
                &second,
                first_line_graph.edge_map(),
                second_line_graph.edge_map(),
                &junction_compatible,
                &edge_pair_allowed,
            ),
            Some(2),
        );
    }

    #[test]
    fn test_positive_weight_matching_score_handles_trivial_inputs() {
        assert_eq!(positive_weight_matching_score(&[], 0, 3), 0);
        assert_eq!(positive_weight_matching_score(&[0, 0, 0, 0], 2, 2), 0);
    }

    #[test]
    fn test_oriented_edge_score_returns_zero_when_first_edge_map_is_missing() {
        let adjacency_first = vec![vec![1], vec![0]];
        let adjacency_second = vec![vec![1], vec![0]];
        let edge_map_second = vec![(0usize, 1usize)];
        let junction_compatible = vec![true; 4];
        let edge_pair_allowed = vec![true];
        let mut solver = ConnectedTreeBoundSolver::new(
            &adjacency_first,
            &adjacency_second,
            &[],
            &edge_map_second,
            &junction_compatible,
            &edge_pair_allowed,
        );

        assert_eq!(solver.oriented_edge_score(0, 1, 0, 1), 0);
    }

    #[test]
    fn test_oriented_edge_score_returns_zero_when_second_edge_map_is_missing() {
        let adjacency_first = vec![vec![1], vec![0]];
        let adjacency_second = vec![vec![1], vec![0]];
        let edge_map_first = vec![(0usize, 1usize)];
        let junction_compatible = vec![true; 4];
        let edge_pair_allowed = vec![true];
        let mut solver = ConnectedTreeBoundSolver::new(
            &adjacency_first,
            &adjacency_second,
            &edge_map_first,
            &[],
            &junction_compatible,
            &edge_pair_allowed,
        );

        assert_eq!(solver.oriented_edge_score(0, 1, 0, 1), 0);
    }
}
