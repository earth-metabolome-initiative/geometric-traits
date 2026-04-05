use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use super::{BETWEENNESS_SCORE_SCALE, NodeScorer, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Betweenness centrality scorer.
///
/// This matches the exact unweighted `NetworkX` betweenness-centrality
/// contract on undirected simple graphs, including the `normalized` and
/// `endpoints` parameter behavior. Weighted edges and source sampling are not
/// part of this scorer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetweennessCentralityScorer {
    normalized: bool,
    endpoints: bool,
}

/// Builder for [`BetweennessCentralityScorer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetweennessCentralityScorerBuilder {
    normalized: bool,
    endpoints: bool,
}

impl BetweennessCentralityScorer {
    /// Creates a new betweenness-centrality scorer with the provided
    /// parameters.
    #[inline]
    #[must_use]
    pub const fn new(normalized: bool, endpoints: bool) -> Self {
        Self { normalized, endpoints }
    }

    /// Creates a builder for configuring betweenness-centrality parameters.
    #[inline]
    #[must_use]
    pub const fn builder() -> BetweennessCentralityScorerBuilder {
        BetweennessCentralityScorerBuilder::new()
    }
}

impl Default for BetweennessCentralityScorer {
    #[inline]
    fn default() -> Self {
        Self::new(true, false)
    }
}

impl BetweennessCentralityScorerBuilder {
    /// Creates a builder initialized with NetworkX-compatible defaults.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { normalized: true, endpoints: false }
    }

    /// Sets whether to normalize the final scores.
    #[inline]
    #[must_use]
    pub const fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Sets whether to include path endpoints in the centrality counts.
    #[inline]
    #[must_use]
    pub const fn endpoints(mut self, endpoints: bool) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// Builds the configured scorer.
    #[inline]
    #[must_use]
    pub const fn build(self) -> BetweennessCentralityScorer {
        BetweennessCentralityScorer::new(self.normalized, self.endpoints)
    }
}

impl Default for BetweennessCentralityScorerBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<G> NodeScorer<G> for BetweennessCentralityScorer
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
        let mut stack = Vec::with_capacity(n);
        let mut predecessors = vec![Vec::<usize>::new(); n];
        let mut sigma = vec![0.0; n];
        let mut distance = vec![usize::MAX; n];
        let mut queue = VecDeque::with_capacity(n);
        let mut delta = vec![0.0; n];

        for source_index in 0..n {
            stack.clear();
            queue.clear();
            sigma.fill(0.0);
            distance.fill(usize::MAX);
            delta.fill(0.0);
            for node_predecessors in &mut predecessors {
                node_predecessors.clear();
            }

            sigma[source_index] = 1.0;
            distance[source_index] = 0;
            queue.push_back(source_index);

            while let Some(node_index) = queue.pop_front() {
                stack.push(node_index);
                let node = nodes[node_index];
                let node_distance = distance[node_index];
                let node_sigma = sigma[node_index];

                for neighbor in graph.neighbors(node) {
                    let neighbor_index = neighbor.as_();
                    if distance[neighbor_index] == usize::MAX {
                        queue.push_back(neighbor_index);
                        distance[neighbor_index] = node_distance + 1;
                    }
                    if distance[neighbor_index] == node_distance + 1 {
                        sigma[neighbor_index] += node_sigma;
                        predecessors[neighbor_index].push(node_index);
                    }
                }
            }

            if self.endpoints {
                scores[source_index] += usize_to_f64(stack.len().saturating_sub(1));
            }

            while let Some(node_index) = stack.pop() {
                let coefficient = (1.0 + delta[node_index]) / sigma[node_index];
                for &predecessor_index in &predecessors[node_index] {
                    delta[predecessor_index] += sigma[predecessor_index] * coefficient;
                }
                if node_index != source_index {
                    scores[node_index] +=
                        if self.endpoints { delta[node_index] + 1.0 } else { delta[node_index] };
                }
            }
        }

        let scale = if self.normalized {
            if self.endpoints {
                if n < 2 { None } else { Some(1.0 / (usize_to_f64(n) * usize_to_f64(n - 1))) }
            } else if n <= 2 {
                None
            } else {
                Some(1.0 / (usize_to_f64(n - 1) * usize_to_f64(n - 2)))
            }
        } else {
            Some(0.5)
        };

        if let Some(scale) = scale {
            for score in &mut scores {
                *score *= scale;
                *score = (*score * BETWEENNESS_SCORE_SCALE).round() / BETWEENNESS_SCORE_SCALE;
            }
        } else {
            for score in &mut scores {
                *score = (*score * BETWEENNESS_SCORE_SCALE).round() / BETWEENNESS_SCORE_SCALE;
            }
        }

        scores
    }
}
