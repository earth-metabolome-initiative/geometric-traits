use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{NodeScorer, POWER_ITERATION_EIGENVECTOR_SCORE_SCALE, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Eigenvector-centrality scorer computed with shifted power iteration.
///
/// This matches the unweighted `NetworkX` power-method contract on
/// undirected simple graphs: it starts from the all-ones vector, iterates with
/// `(A + I)` for improved convergence behavior on bipartite graphs, normalizes
/// each iterate to unit Euclidean norm, and checks convergence using the L1
/// norm of the change vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerIterationEigenvectorCentralityScorer {
    max_iter: usize,
    tolerance: f64,
}

/// Builder for [`PowerIterationEigenvectorCentralityScorer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerIterationEigenvectorCentralityScorerBuilder {
    max_iter: usize,
    tolerance: f64,
}

impl PowerIterationEigenvectorCentralityScorer {
    /// Creates a new power-iteration eigenvector-centrality scorer with the
    /// provided parameters.
    #[inline]
    #[must_use]
    pub const fn new(max_iter: usize, tolerance: f64) -> Self {
        Self { max_iter, tolerance }
    }

    /// Creates a builder for configuring power-iteration
    /// eigenvector-centrality parameters.
    #[inline]
    #[must_use]
    pub const fn builder() -> PowerIterationEigenvectorCentralityScorerBuilder {
        PowerIterationEigenvectorCentralityScorerBuilder::new()
    }
}

impl Default for PowerIterationEigenvectorCentralityScorer {
    #[inline]
    fn default() -> Self {
        Self::new(100, 1.0e-6)
    }
}

impl PowerIterationEigenvectorCentralityScorerBuilder {
    /// Creates a builder initialized with `NetworkX`-compatible defaults.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { max_iter: 100, tolerance: 1.0e-6 }
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
    pub const fn build(self) -> PowerIterationEigenvectorCentralityScorer {
        PowerIterationEigenvectorCentralityScorer::new(self.max_iter, self.tolerance)
    }
}

impl Default for PowerIterationEigenvectorCentralityScorerBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<G> NodeScorer<G> for PowerIterationEigenvectorCentralityScorer
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

        let initial_score = 1.0 / usize_to_f64(n);
        let mut previous_scores = vec![initial_score; n];
        let mut scores = vec![0.0; n];
        let convergence_threshold = usize_to_f64(n) * self.tolerance;

        for _ in 0..self.max_iter {
            scores.copy_from_slice(&previous_scores);

            for (node_index, &node) in nodes.iter().enumerate() {
                let node_score = previous_scores[node_index];
                for neighbor in graph.neighbors(node) {
                    scores[neighbor.as_()] += node_score;
                }
            }

            let inverse_norm = {
                let norm_squared = scores
                    .iter()
                    .fold(0.0, |accumulator, score| (*score).mul_add(*score, accumulator));
                if norm_squared == 0.0 { 1.0 } else { norm_squared.sqrt().recip() }
            };

            let mut error = 0.0;
            for (score, previous_score) in scores.iter_mut().zip(previous_scores.iter()) {
                *score *= inverse_norm;
                error += (*score - *previous_score).abs();
            }
            if error < convergence_threshold {
                for score in &mut scores {
                    *score = (*score * POWER_ITERATION_EIGENVECTOR_SCORE_SCALE).round()
                        / POWER_ITERATION_EIGENVECTOR_SCORE_SCALE;
                }
                return scores;
            }

            core::mem::swap(&mut scores, &mut previous_scores);
        }

        panic!(
            "PowerIterationEigenvectorCentralityScorer failed to converge within {} iterations (tolerance={})",
            self.max_iter, self.tolerance
        );
    }
}
