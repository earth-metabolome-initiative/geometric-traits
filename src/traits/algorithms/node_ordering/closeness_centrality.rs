use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use super::{CLOSENESS_SCORE_SCALE, NodeScorer, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Closeness centrality scorer.
///
/// This matches the exact unweighted `NetworkX` closeness-centrality contract
/// on undirected simple graphs for the supported `wf_improved` parameter.
/// Weighted distances are not part of this scorer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosenessCentralityScorer {
    wf_improved: bool,
}

/// Builder for [`ClosenessCentralityScorer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosenessCentralityScorerBuilder {
    wf_improved: bool,
}

impl ClosenessCentralityScorer {
    /// Creates a new closeness-centrality scorer with the provided parameters.
    #[inline]
    #[must_use]
    pub const fn new(wf_improved: bool) -> Self {
        Self { wf_improved }
    }

    /// Creates a builder for configuring closeness-centrality parameters.
    #[inline]
    #[must_use]
    pub const fn builder() -> ClosenessCentralityScorerBuilder {
        ClosenessCentralityScorerBuilder::new()
    }
}

impl Default for ClosenessCentralityScorer {
    #[inline]
    fn default() -> Self {
        Self::new(true)
    }
}

impl ClosenessCentralityScorerBuilder {
    /// Creates a builder initialized with NetworkX-compatible defaults.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { wf_improved: true }
    }

    /// Sets whether to apply the Wasserman-Faust connected-component scaling.
    #[inline]
    #[must_use]
    pub const fn wf_improved(mut self, wf_improved: bool) -> Self {
        self.wf_improved = wf_improved;
        self
    }

    /// Builds the configured scorer.
    #[inline]
    #[must_use]
    pub const fn build(self) -> ClosenessCentralityScorer {
        ClosenessCentralityScorer::new(self.wf_improved)
    }
}

impl Default for ClosenessCentralityScorerBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<G> NodeScorer<G> for ClosenessCentralityScorer
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

        let mut scores = vec![0.0; n];
        let mut distance = vec![usize::MAX; n];
        let mut queue = VecDeque::with_capacity(n);

        for source_index in 0..n {
            distance.fill(usize::MAX);
            queue.clear();
            distance[source_index] = 0;
            queue.push_back(source_index);

            let mut total_shortest_path_distance = 0usize;
            let mut reachable_nodes = 1usize;

            while let Some(node_index) = queue.pop_front() {
                let node = nodes[node_index];
                let node_distance = distance[node_index];

                for neighbor in graph.neighbors(node) {
                    let neighbor_index = neighbor.as_();
                    if distance[neighbor_index] != usize::MAX {
                        continue;
                    }

                    let neighbor_distance = node_distance + 1;
                    distance[neighbor_index] = neighbor_distance;
                    total_shortest_path_distance += neighbor_distance;
                    reachable_nodes += 1;
                    queue.push_back(neighbor_index);
                }
            }

            if total_shortest_path_distance > 0 && n > 1 {
                let reachable_minus_source = reachable_nodes - 1;
                let mut score = usize_to_f64(reachable_minus_source)
                    / usize_to_f64(total_shortest_path_distance);
                if self.wf_improved {
                    score *= usize_to_f64(reachable_minus_source) / usize_to_f64(n - 1);
                }
                scores[source_index] =
                    (score * CLOSENESS_SCORE_SCALE).round() / CLOSENESS_SCORE_SCALE;
            }
        }

        scores
    }
}
