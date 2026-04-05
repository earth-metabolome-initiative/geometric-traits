use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{KATZ_SCORE_SCALE, NodeScorer, usize_to_f64};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

/// Katz centrality scorer.
///
/// This matches the scalar-`beta`, unweighted `NetworkX` Katz centrality
/// contract on undirected simple graphs. The iterative update is
/// `x_{k+1} = alpha * A * x_k + beta`, starting from the all-zero vector.
/// When `normalized` is enabled, the converged score vector is rescaled to
/// unit Euclidean norm.
///
/// The default parameters intentionally mirror `NetworkX`, including
/// `alpha=0.1`. That default is not convergence-safe on arbitrary graphs:
/// for undirected graphs, a sufficient condition is `alpha < 1 / Delta`,
/// where `Delta` is the maximum degree. Use
/// [`KatzCentralityScorer::safe_alpha_from_max_degree`] or
/// [`KatzCentralityScorer::safe_alpha_for_graph`] when you want a conservative
/// attenuation factor that converges across a wider range of graphs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KatzCentralityScorer {
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tolerance: f64,
    normalized: bool,
}

/// Builder for [`KatzCentralityScorer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KatzCentralityScorerBuilder {
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tolerance: f64,
    normalized: bool,
}

impl KatzCentralityScorer {
    /// Creates a new Katz centrality scorer with the provided parameters.
    #[inline]
    #[must_use]
    pub const fn new(
        alpha: f64,
        beta: f64,
        max_iter: usize,
        tolerance: f64,
        normalized: bool,
    ) -> Self {
        Self { alpha, beta, max_iter, tolerance, normalized }
    }

    /// Creates a builder for configuring Katz centrality parameters.
    #[inline]
    #[must_use]
    pub const fn builder() -> KatzCentralityScorerBuilder {
        KatzCentralityScorerBuilder::new()
    }

    /// Returns a conservative Katz attenuation factor from a maximum-degree
    /// bound.
    ///
    /// For undirected graphs, `rho(A) <= Delta`, so `alpha < 1 / Delta` is a
    /// sufficient convergence condition. This helper uses a 10% safety margin:
    /// `0.9 / Delta`. When `max_degree == 0`, the adjacency matrix is zero and
    /// Katz converges for any finite `alpha`; this helper returns `0.1`.
    #[inline]
    #[must_use]
    pub fn safe_alpha_from_max_degree(max_degree: usize) -> f64 {
        if max_degree == 0 { 0.1 } else { 0.9 / usize_to_f64(max_degree) }
    }

    /// Returns a conservative Katz attenuation factor for the given graph.
    ///
    /// This scans the graph's maximum degree and applies
    /// [`KatzCentralityScorer::safe_alpha_from_max_degree`].
    #[must_use]
    pub fn safe_alpha_for_graph<G>(graph: &G) -> f64
    where
        G: UndirectedMonopartiteMonoplexGraph,
    {
        let max_degree = graph.node_ids().map(|node| graph.degree(node).as_()).max().unwrap_or(0);
        Self::safe_alpha_from_max_degree(max_degree)
    }
}

impl Default for KatzCentralityScorer {
    #[inline]
    fn default() -> Self {
        Self::new(0.1, 1.0, 1000, 1.0e-6, true)
    }
}

impl KatzCentralityScorerBuilder {
    /// Creates a builder initialized with `NetworkX`-compatible defaults.
    ///
    /// Note that the default `alpha=0.1` is not guaranteed to converge on
    /// arbitrary graphs. If you want a conservative graph-dependent value, use
    /// [`KatzCentralityScorerBuilder::safe_alpha_from_graph`].
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { alpha: 0.1, beta: 1.0, max_iter: 1000, tolerance: 1.0e-6, normalized: true }
    }

    /// Sets the attenuation factor.
    #[inline]
    #[must_use]
    pub const fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets `alpha` from the graph's maximum-degree bound.
    ///
    /// This uses [`KatzCentralityScorer::safe_alpha_for_graph`] and is the
    /// recommended entry point when you want a conservative `alpha` without
    /// estimating the spectral radius directly.
    #[must_use]
    pub fn safe_alpha_from_graph<G>(mut self, graph: &G) -> Self
    where
        G: UndirectedMonopartiteMonoplexGraph,
    {
        self.alpha = KatzCentralityScorer::safe_alpha_for_graph(graph);
        self
    }

    /// Sets the immediate-neighborhood weight.
    #[inline]
    #[must_use]
    pub const fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
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

    /// Sets whether to normalize the converged vector.
    #[inline]
    #[must_use]
    pub const fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Builds the configured scorer.
    #[inline]
    #[must_use]
    pub const fn build(self) -> KatzCentralityScorer {
        KatzCentralityScorer::new(
            self.alpha,
            self.beta,
            self.max_iter,
            self.tolerance,
            self.normalized,
        )
    }
}

impl Default for KatzCentralityScorerBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<G> NodeScorer<G> for KatzCentralityScorer
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

        for _ in 0..self.max_iter {
            let previous_scores = scores.clone();
            scores.fill(0.0);

            for (node_index, &node) in nodes.iter().enumerate() {
                for neighbor in graph.neighbors(node) {
                    scores[neighbor.as_()] += previous_scores[node_index];
                }
            }

            for score in &mut scores {
                *score = self.alpha * *score + self.beta;
            }

            let error: f64 = scores
                .iter()
                .zip(previous_scores.iter())
                .map(|(left, right)| (left - right).abs())
                .sum();
            if error < usize_to_f64(n) * self.tolerance {
                let scale = if self.normalized {
                    let norm = scores.iter().map(|score| score * score).sum::<f64>().sqrt();
                    if norm == 0.0 { 1.0 } else { 1.0 / norm }
                } else {
                    1.0
                };

                for score in &mut scores {
                    *score *= scale;
                    *score = (*score * KATZ_SCORE_SCALE).round() / KATZ_SCORE_SCALE;
                }

                return scores;
            }
        }

        panic!(
            "KatzCentralityScorer failed to converge within {} iterations (alpha={}, beta={}, tolerance={}, normalized={})",
            self.max_iter, self.alpha, self.beta, self.tolerance, self.normalized
        );
    }
}
