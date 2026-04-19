use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{
    NodeScorer,
    motifs::{MotifCountOrdering, build_undirected_motif_context},
};
use crate::{impls::SortedIterator, traits::UndirectedMonopartiteMonoplexGraph};

pub(super) fn square_counts_with_ordering<G>(graph: &G, ordering: MotifCountOrdering) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let context = build_undirected_motif_context(graph, ordering);
    let n = context.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut counts = vec![0usize; n];
    let mut visited = vec![0usize; n];

    for root_index in 0..n {
        if !context.in_cover[root_index] {
            continue;
        }

        let root_node = context.nodes[root_index];
        let stamp = root_index + 1;

        for via in graph.neighbors(root_node) {
            let via_index = via.as_();
            if via_index == root_index {
                continue;
            }

            for candidate in graph.neighbors(via) {
                let candidate_index = candidate.as_();
                if candidate_index == root_index
                    || candidate_index == via_index
                    || !context.in_cover[candidate_index]
                    || context.rank[candidate_index] >= context.rank[root_index]
                    || visited[candidate_index] == stamp
                {
                    continue;
                }

                visited[candidate_index] = stamp;

                let candidate_node = context.nodes[candidate_index];
                let mut cover_common = 0usize;
                let mut common_nodes = Vec::new();

                for common in
                    graph.neighbors(root_node).sorted_intersection(graph.neighbors(candidate_node))
                {
                    let common_index = common.as_();
                    if common_index == root_index || common_index == candidate_index {
                        continue;
                    }

                    common_nodes.push(common_index);
                    if context.in_cover[common_index] {
                        cover_common += 1;
                    }
                }

                let common_count = common_nodes.len();
                if common_count < 2 {
                    continue;
                }

                let outside_common = common_count - cover_common;
                let root_contribution = outside_common * outside_common.saturating_sub(1) / 2
                    + outside_common * cover_common;
                counts[root_index] += root_contribution;
                counts[candidate_index] += root_contribution;

                let common_contribution = common_count - 1;
                for common_index in common_nodes {
                    counts[common_index] += common_contribution;
                }
            }
        }
    }

    counts
}

/// Square-count scorer.
///
/// The score of a node is the number of distinct 4-cycles incident to that
/// node. The ordering selects the paper's vertex-cover construction heuristic.
/// The paper does not show a clear globally best schema for square counting, so
/// the default is [`MotifCountOrdering::Natural`]. Chorded 4-cycles are counted
/// in all modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SquareCountScorer {
    ordering: MotifCountOrdering,
}

impl SquareCountScorer {
    /// Creates a square-count scorer with the selected cover-ordering
    /// heuristic.
    ///
    /// # Examples
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{
    ///         SquareMatrix, VocabularyBuilder,
    ///         algorithms::{
    ///             MotifCountOrdering, NodeScorer, SquareCountScorer, randomized_graphs::cycle_graph,
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
    /// let scorer = SquareCountScorer::new(MotifCountOrdering::Natural);
    ///
    /// assert_eq!(scorer.score_nodes(&graph), vec![1, 1, 1, 1]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(ordering: MotifCountOrdering) -> Self {
        Self { ordering }
    }

    /// Returns the selected cover-ordering heuristic.
    ///
    /// # Examples
    /// ```
    /// use geometric_traits::traits::algorithms::{MotifCountOrdering, SquareCountScorer};
    ///
    /// let scorer = SquareCountScorer::new(MotifCountOrdering::Natural);
    /// assert_eq!(scorer.ordering(), MotifCountOrdering::Natural);
    /// ```
    #[inline]
    #[must_use]
    pub const fn ordering(self) -> MotifCountOrdering {
        self.ordering
    }
}

impl Default for SquareCountScorer {
    #[inline]
    fn default() -> Self {
        Self::new(MotifCountOrdering::Natural)
    }
}

impl<G> NodeScorer<G> for SquareCountScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        square_counts_with_ordering(graph, self.ordering)
    }
}
