use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::NodeScorer;
use crate::traits::UndirectedMonopartiteMonoplexGraph;

pub(super) fn triangle_counts<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    if n == 0 {
        return Vec::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let mut forward_neighbors = vec![Vec::new(); n];

    for &node in &nodes {
        let node_index = node.as_();
        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if (degrees[node_index], node_index) < (degrees[neighbor_index], neighbor_index) {
                forward_neighbors[node_index].push(neighbor_index);
            }
        }
    }

    let mut counts = vec![0usize; n];
    let mut marks = vec![0usize; n];

    for node_index in 0..n {
        let stamp = node_index + 1;
        for &neighbor_index in &forward_neighbors[node_index] {
            marks[neighbor_index] = stamp;
        }

        for &neighbor_index in &forward_neighbors[node_index] {
            for &candidate_index in &forward_neighbors[neighbor_index] {
                if marks[candidate_index] == stamp {
                    counts[node_index] += 1;
                    counts[neighbor_index] += 1;
                    counts[candidate_index] += 1;
                }
            }
        }
    }

    counts
}

/// Triangle-count scorer.
///
/// The score of a node is the number of distinct triangles incident to that
/// node. The implementation uses a degree-oriented forward-neighborhood
/// traversal so each triangle is counted exactly once and credited to all
/// three participating nodes.
#[derive(Clone, Copy, Debug, Default)]
pub struct TriangleCountScorer;

impl<G> NodeScorer<G> for TriangleCountScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        triangle_counts(graph)
    }
}
