use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{NodeScorer, motifs::MotifCountOrdering};
use crate::{
    impls::SortedIterator,
    traits::{
        UndirectedMonopartiteMonoplexGraph,
        algorithms::node_ordering::motifs::build_undirected_motif_context,
    },
};

pub(super) fn triangle_counts<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    triangle_counts_with_ordering(graph, MotifCountOrdering::IncreasingDegree)
}

pub(super) fn triangle_counts_with_ordering<G>(
    graph: &G,
    ordering: MotifCountOrdering,
) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let context = build_undirected_motif_context(graph, ordering);
    let n = context.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut counts = vec![0usize; n];

    for root_index in 0..n {
        if !context.in_cover[root_index] {
            continue;
        }

        let root_node = context.nodes[root_index];
        for neighbor in graph.neighbors(root_node) {
            let neighbor_index = neighbor.as_();
            if neighbor_index == root_index
                || !context.in_cover[neighbor_index]
                || context.rank[neighbor_index] >= context.rank[root_index]
            {
                continue;
            }

            for common in graph.neighbors(root_node).sorted_intersection(graph.neighbors(neighbor))
            {
                let common_index = common.as_();
                if common_index == root_index || common_index == neighbor_index {
                    continue;
                }
                if context.in_cover[common_index]
                    && context.rank[common_index] >= context.rank[neighbor_index]
                {
                    continue;
                }

                counts[root_index] += 1;
                counts[neighbor_index] += 1;
                counts[common_index] += 1;
            }
        }
    }

    counts
}

/// Triangle-count scorer.
///
/// The score of a node is the number of distinct triangles incident to that
/// node. The ordering selects the paper's vertex-cover construction heuristic.
/// [`MotifCountOrdering::IncreasingDegree`] is the preferred and default mode
/// for triangle counting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TriangleCountScorer {
    ordering: MotifCountOrdering,
}

impl TriangleCountScorer {
    /// Creates a triangle-count scorer with the selected cover-ordering
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
    ///             MotifCountOrdering, NodeScorer, TriangleCountScorer, randomized_graphs::cycle_graph,
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
    /// let graph = wrap_undi(cycle_graph(3));
    /// let scorer = TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree);
    ///
    /// assert_eq!(scorer.score_nodes(&graph), vec![1, 1, 1]);
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
    /// use geometric_traits::traits::algorithms::{MotifCountOrdering, TriangleCountScorer};
    ///
    /// let scorer = TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree);
    /// assert_eq!(scorer.ordering(), MotifCountOrdering::IncreasingDegree);
    /// ```
    #[inline]
    #[must_use]
    pub const fn ordering(self) -> MotifCountOrdering {
        self.ordering
    }
}

impl Default for TriangleCountScorer {
    #[inline]
    fn default() -> Self {
        Self::new(MotifCountOrdering::IncreasingDegree)
    }
}

impl<G> NodeScorer<G> for TriangleCountScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        triangle_counts_with_ordering(graph, self.ordering)
    }
}
