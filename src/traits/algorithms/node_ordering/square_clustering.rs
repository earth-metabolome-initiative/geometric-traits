use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{NodeScorer, SQUARE_CLUSTERING_SCORE_SCALE, usize_to_f64};
use crate::{impls::SortedIterator, traits::UndirectedMonopartiteMonoplexGraph};

/// Square clustering-coefficient scorer.
///
/// This matches the exact unweighted `NetworkX` `square_clustering()`
/// coefficient on undirected simple graphs. Self loops are ignored.
///
/// # Examples
/// ```
/// use geometric_traits::{
///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
///     prelude::*,
///     traits::{
///         SquareMatrix, VocabularyBuilder,
///         algorithms::{
///             NodeScorer, SquareClusteringCoefficientScorer, randomized_graphs::cycle_graph,
///         },
///     },
/// };
///
/// fn wrap_undi(matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>>) -> UndiGraph<usize> {
///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
///         .expected_number_of_symbols(matrix.order())
///         .symbols((0..matrix.order()).enumerate())
///         .build()
///         .unwrap();
///     UndiGraph::from((nodes, matrix))
/// }
///
/// let graph = wrap_undi(cycle_graph(4));
///
/// assert_eq!(SquareClusteringCoefficientScorer.score_nodes(&graph), vec![1.0, 1.0, 1.0, 1.0]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct SquareClusteringCoefficientScorer;

impl<G> NodeScorer<G> for SquareClusteringCoefficientScorer
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

        let neighbors: Vec<Vec<usize>> = nodes
            .iter()
            .map(|&node| {
                let node_index = node.as_();
                graph
                    .neighbors(node)
                    .filter(|neighbor| neighbor.as_() != node_index)
                    .map(AsPrimitive::as_)
                    .collect()
            })
            .collect();
        let degrees: Vec<usize> = neighbors.iter().map(Vec::len).collect();
        let mut scores = vec![0.0; n];

        for root_index in 0..n {
            let root_neighbors = &neighbors[root_index];
            if root_neighbors.len() < 2 {
                continue;
            }

            let mut actual_squares = 0usize;
            let mut actual_plus_possible = 0usize;

            for left_offset in 0..root_neighbors.len() {
                let left = root_neighbors[left_offset];
                for &right in &root_neighbors[(left_offset + 1)..] {
                    let common_neighbors = neighbors[left]
                        .iter()
                        .copied()
                        .sorted_intersection(neighbors[right].iter().copied())
                        .filter(|&common| common != root_index && common != left && common != right)
                        .count();
                    let adjacent = usize::from(neighbors[left].binary_search(&right).is_ok());
                    let available_left =
                        degrees[left].saturating_sub(1 + common_neighbors + adjacent);
                    let available_right =
                        degrees[right].saturating_sub(1 + common_neighbors + adjacent);

                    actual_squares += common_neighbors;
                    actual_plus_possible += common_neighbors + available_left + available_right;
                }
            }

            if actual_plus_possible == 0 {
                continue;
            }

            scores[root_index] = (usize_to_f64(actual_squares)
                / usize_to_f64(actual_plus_possible)
                * SQUARE_CLUSTERING_SCORE_SCALE)
                .round()
                / SQUARE_CLUSTERING_SCORE_SCALE;
        }

        scores
    }
}
