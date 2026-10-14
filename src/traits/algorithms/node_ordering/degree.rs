use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::NodeScorer;
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Degree scorer.
///
/// The score of a node is its degree in the graph.
#[derive(Clone, Copy, Debug, Default)]
pub struct DegreeScorer;

impl<G> NodeScorer<G> for DegreeScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        let n = graph.number_of_nodes().as_();
        let nodes: Vec<G::NodeId> = graph.node_ids().collect();
        debug_assert_eq!(nodes.len(), n);
        debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

        nodes.iter().map(|&node| graph.degree(node).as_()).collect()
    }
}

/// Second-order degree scorer.
///
/// The score of a node is the sum of the degrees of its neighbors.
#[derive(Clone, Copy, Debug, Default)]
pub struct SecondOrderDegreeScorer;

impl<G> NodeScorer<G> for SecondOrderDegreeScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        let n = graph.number_of_nodes().as_();
        let nodes: Vec<G::NodeId> = graph.node_ids().collect();
        debug_assert_eq!(nodes.len(), n);
        debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
        let mut scores = vec![0usize; n];

        for &node in &nodes {
            let node_index = node.as_();
            scores[node_index] =
                graph.neighbors(node).map(|neighbor| degrees[neighbor.as_()]).sum();
        }

        scores
    }
}
