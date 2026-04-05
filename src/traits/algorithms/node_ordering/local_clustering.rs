use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{LOCAL_CLUSTERING_SCORE_SCALE, NodeScorer, triangles::triangle_counts, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Local clustering-coefficient scorer.
///
/// This matches the exact unweighted `NetworkX` local clustering coefficient on
/// undirected simple graphs. Weighted variants are not part of this scorer.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalClusteringCoefficientScorer;

impl<G> NodeScorer<G> for LocalClusteringCoefficientScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = f64;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        let n = graph.number_of_nodes().as_();
        if n == 0 {
            return Vec::new();
        }

        let nodes: Vec<G::NodeId> = graph.node_ids().collect();
        debug_assert_eq!(nodes.len(), n);
        debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
        let triangle_counts = triangle_counts(graph);
        let mut scores = vec![0.0; n];

        for node_index in 0..n {
            let degree = degrees[node_index];
            let triangles = triangle_counts[node_index];
            if degree < 2 || triangles == 0 {
                continue;
            }

            let numerator = 2.0 * usize_to_f64(triangles);
            let denominator = usize_to_f64(degree * (degree - 1));
            scores[node_index] = (numerator / denominator * LOCAL_CLUSTERING_SCORE_SCALE).round()
                / LOCAL_CLUSTERING_SCORE_SCALE;
        }

        scores
    }
}
