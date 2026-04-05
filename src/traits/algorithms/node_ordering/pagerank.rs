use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{NodeScorer, PAGERANK_SCORE_SCALE, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// PageRank scorer.
///
/// This matches NetworkX's default undirected PageRank contract:
/// `alpha=0.85`, `max_iter=100`, `tol=1e-6`, uniform initialization,
/// uniform personalization, and uniform dangling redistribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageRankScorer {
    alpha: f64,
    max_iter: usize,
    tolerance: f64,
}

/// Builder for [`PageRankScorer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageRankScorerBuilder {
    alpha: f64,
    max_iter: usize,
    tolerance: f64,
}

impl PageRankScorer {
    /// Creates a new PageRank scorer with the provided parameters.
    #[inline]
    #[must_use]
    pub const fn new(alpha: f64, max_iter: usize, tolerance: f64) -> Self {
        Self { alpha, max_iter, tolerance }
    }

    /// Creates a builder for configuring PageRank parameters.
    #[inline]
    #[must_use]
    pub const fn builder() -> PageRankScorerBuilder {
        PageRankScorerBuilder::new()
    }
}

impl Default for PageRankScorer {
    #[inline]
    fn default() -> Self {
        Self::new(0.85, 100, 1.0e-6)
    }
}

impl PageRankScorerBuilder {
    /// Creates a builder initialized with NetworkX-compatible defaults.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { alpha: 0.85, max_iter: 100, tolerance: 1.0e-6 }
    }

    /// Sets the damping factor.
    #[inline]
    #[must_use]
    pub const fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the maximum number of power iterations.
    #[inline]
    #[must_use]
    pub const fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[inline]
    #[must_use]
    pub const fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Builds the configured scorer.
    #[inline]
    #[must_use]
    pub const fn build(self) -> PageRankScorer {
        PageRankScorer::new(self.alpha, self.max_iter, self.tolerance)
    }
}

impl Default for PageRankScorerBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<G> NodeScorer<G> for PageRankScorer
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

        let inv_n = 1.0 / usize_to_f64(n);
        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
        let teleport = (1.0 - self.alpha) * inv_n;
        let mut scores = vec![inv_n; n];

        for _ in 0..self.max_iter {
            let previous_scores = scores.clone();
            let dangling_mass: f64 = previous_scores
                .iter()
                .enumerate()
                .filter(|(node_index, _)| degrees[*node_index] == 0)
                .map(|(_, score)| *score)
                .sum();
            let dangling_contribution = self.alpha * dangling_mass * inv_n;

            scores.fill(teleport + dangling_contribution);

            for (node_index, &node) in nodes.iter().enumerate() {
                let degree = degrees[node_index];
                if degree == 0 {
                    continue;
                }

                let contribution = self.alpha * previous_scores[node_index] / usize_to_f64(degree);
                for neighbor in graph.neighbors(node) {
                    scores[neighbor.as_()] += contribution;
                }
            }

            let error: f64 = scores
                .iter()
                .zip(previous_scores.iter())
                .map(|(left, right)| (left - right).abs())
                .sum();
            if error < usize_to_f64(n) * self.tolerance {
                for score in &mut scores {
                    *score = (*score * PAGERANK_SCORE_SCALE).round() / PAGERANK_SCORE_SCALE;
                }
                return scores;
            }
        }

        panic!(
            "PageRankScorer failed to converge within {} iterations (alpha={}, tolerance={})",
            self.max_iter, self.alpha, self.tolerance
        );
    }
}
