//! Graph-level node ordering and node scoring primitives.
//!
//! This module separates:
//! - direct ordering algorithms, represented by [`NodeSorter`]
//! - reusable per-node metrics, represented by [`NodeScorer`]
//!
//! Not every ordering is naturally "sort by one local score". For example,
//! degeneracy is a procedural ordering and is therefore modeled directly as a
//! sorter. Metrics such as second-order degree are modeled as scorers and can
//! be paired with generic ascending or descending score sorters.

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::{AsPrimitive, cast};

use crate::traits::{MonopartiteGraph, TotalOrd, UndirectedMonopartiteMonoplexGraph};

/// Trait for algorithms that return a complete node ordering for a graph.
pub trait NodeSorter<G>
where
    G: MonopartiteGraph,
{
    /// Returns a complete ordering of the graph's node ids.
    #[must_use]
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId>;
}

/// Trait for algorithms that compute one score per node.
///
/// The returned vector is dense-indexed by node id via `node_id.as_()`.
pub trait NodeScorer<G>
where
    G: MonopartiteGraph,
{
    /// Score type produced for each node.
    type Score;

    /// Returns one score per node, indexed by dense node id.
    #[must_use]
    fn score_nodes(&self, graph: &G) -> Vec<Self::Score>;
}

/// Generic ascending sorter built from a node scorer.
#[derive(Clone, Copy, Debug, Default)]
pub struct AscendingScoreSorter<S> {
    scorer: S,
}

impl<S> AscendingScoreSorter<S> {
    /// Creates a new ascending score sorter from the provided scorer.
    #[inline]
    #[must_use]
    pub const fn new(scorer: S) -> Self {
        Self { scorer }
    }
}

/// Generic descending sorter built from a node scorer.
#[derive(Clone, Copy, Debug, Default)]
pub struct DescendingScoreSorter<S> {
    scorer: S,
}

impl<S> DescendingScoreSorter<S> {
    /// Creates a new descending score sorter from the provided scorer.
    #[inline]
    #[must_use]
    pub const fn new(scorer: S) -> Self {
        Self { scorer }
    }
}

/// Generic descending lexicographic sorter built from two node scorers.
#[derive(Clone, Copy, Debug, Default)]
pub struct DescendingLexicographicScoreSorter<Primary, Secondary> {
    primary_scorer: Primary,
    secondary_scorer: Secondary,
}

impl<Primary, Secondary> DescendingLexicographicScoreSorter<Primary, Secondary> {
    /// Creates a new descending lexicographic sorter from two scorers.
    #[inline]
    #[must_use]
    pub const fn new(primary_scorer: Primary, secondary_scorer: Secondary) -> Self {
        Self { primary_scorer, secondary_scorer }
    }
}

fn sort_nodes_by_scores<G, S, F>(graph: &G, scorer: &S, compare_scores: F) -> Vec<G::NodeId>
where
    G: MonopartiteGraph,
    S: NodeScorer<G>,
    S::Score: TotalOrd,
    F: Fn(&S::Score, &S::Score) -> core::cmp::Ordering,
{
    let node_scores = scorer.score_nodes(graph);
    let mut nodes: Vec<G::NodeId> = graph.node_ids().collect();
    assert_eq!(
        node_scores.len(),
        graph.number_of_nodes().as_(),
        "node scorer must return one score per node"
    );

    nodes.sort_unstable_by(|left, right| {
        let left_index = (*left).as_();
        let right_index = (*right).as_();
        compare_scores(&node_scores[left_index], &node_scores[right_index])
            .then_with(|| left_index.cmp(&right_index))
    });

    nodes
}

fn sort_nodes_by_score_pairs<G, Primary, Secondary>(
    graph: &G,
    primary: &Primary,
    secondary: &Secondary,
) -> Vec<G::NodeId>
where
    G: MonopartiteGraph,
    Primary: NodeScorer<G>,
    Secondary: NodeScorer<G>,
    Primary::Score: TotalOrd,
    Secondary::Score: TotalOrd,
{
    let primary_scores = primary.score_nodes(graph);
    let secondary_scores = secondary.score_nodes(graph);
    let mut nodes: Vec<G::NodeId> = graph.node_ids().collect();
    assert_eq!(
        primary_scores.len(),
        graph.number_of_nodes().as_(),
        "primary node scorer must return one score per node"
    );
    assert_eq!(
        secondary_scores.len(),
        graph.number_of_nodes().as_(),
        "secondary node scorer must return one score per node"
    );

    nodes.sort_unstable_by(|left, right| {
        let left_index = (*left).as_();
        let right_index = (*right).as_();
        primary_scores[right_index]
            .total_cmp(&primary_scores[left_index])
            .then_with(|| secondary_scores[right_index].total_cmp(&secondary_scores[left_index]))
            .then_with(|| left_index.cmp(&right_index))
    });

    nodes
}

impl<G, S> NodeSorter<G> for AscendingScoreSorter<S>
where
    G: MonopartiteGraph,
    S: NodeScorer<G>,
    S::Score: TotalOrd,
{
    #[inline]
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        sort_nodes_by_scores(graph, &self.scorer, TotalOrd::total_cmp)
    }
}

impl<G, S> NodeSorter<G> for DescendingScoreSorter<S>
where
    G: MonopartiteGraph,
    S: NodeScorer<G>,
    S::Score: TotalOrd,
{
    #[inline]
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        sort_nodes_by_scores(graph, &self.scorer, |left, right| right.total_cmp(left))
    }
}

impl<G, Primary, Secondary> NodeSorter<G> for DescendingLexicographicScoreSorter<Primary, Secondary>
where
    G: MonopartiteGraph,
    Primary: NodeScorer<G>,
    Secondary: NodeScorer<G>,
    Primary::Score: TotalOrd,
    Secondary::Score: TotalOrd,
{
    #[inline]
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        sort_nodes_by_score_pairs(graph, &self.primary_scorer, &self.secondary_scorer)
    }
}

const NO_NODE: usize = usize::MAX;
const PAGERANK_SCORE_SCALE: f64 = 1.0e12;
const KATZ_SCORE_SCALE: f64 = 1.0e12;
const BETWEENNESS_SCORE_SCALE: f64 = 1.0e12;

#[inline]
fn usize_to_f64(value: usize) -> f64 {
    cast::<usize, f64>(value).expect("graph sizes and degrees must fit into f64 for PageRank")
}

struct DegeneracyDecomposition<NodeId> {
    smallest_last_order: Vec<NodeId>,
}

#[inline]
fn bucket_insert_head(
    node_index: usize,
    degree: usize,
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    prev_in_bucket: &mut [usize],
) {
    let previous_head = bucket_heads[degree];
    bucket_heads[degree] = node_index;
    next_in_bucket[node_index] = previous_head;
    prev_in_bucket[node_index] = NO_NODE;
    if previous_head != NO_NODE {
        prev_in_bucket[previous_head] = node_index;
    }
}

#[inline]
fn bucket_remove(
    node_index: usize,
    degree: usize,
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    prev_in_bucket: &mut [usize],
) {
    let previous = prev_in_bucket[node_index];
    let next = next_in_bucket[node_index];

    if previous == NO_NODE {
        bucket_heads[degree] = next;
    } else {
        next_in_bucket[previous] = next;
    }

    if next != NO_NODE {
        prev_in_bucket[next] = previous;
    }

    next_in_bucket[node_index] = NO_NODE;
    prev_in_bucket[node_index] = NO_NODE;
}

fn degeneracy_decomposition<G>(graph: &G) -> DegeneracyDecomposition<G::NodeId>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    if n == 0 {
        return DegeneracyDecomposition { smallest_last_order: Vec::new() };
    }

    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let mut degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let mut bucket_heads = vec![NO_NODE; max_degree + 1];
    let mut next_in_bucket = vec![NO_NODE; n];
    let mut prev_in_bucket = vec![NO_NODE; n];
    let mut removed = vec![false; n];

    for node_index in (0..n).rev() {
        bucket_insert_head(
            node_index,
            degrees[node_index],
            &mut bucket_heads,
            &mut next_in_bucket,
            &mut prev_in_bucket,
        );
    }

    let mut min_degree = degrees.iter().copied().min().unwrap_or(0);
    let mut removal_order = Vec::with_capacity(n);

    for _ in 0..n {
        while min_degree < bucket_heads.len() && bucket_heads[min_degree] == NO_NODE {
            min_degree += 1;
        }
        debug_assert!(min_degree < bucket_heads.len());

        let node_index = bucket_heads[min_degree];
        debug_assert_ne!(node_index, NO_NODE);
        bucket_remove(
            node_index,
            min_degree,
            &mut bucket_heads,
            &mut next_in_bucket,
            &mut prev_in_bucket,
        );

        removed[node_index] = true;
        let node = nodes[node_index];
        removal_order.push(node);

        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if removed[neighbor_index] {
                continue;
            }

            let previous_degree = degrees[neighbor_index];
            bucket_remove(
                neighbor_index,
                previous_degree,
                &mut bucket_heads,
                &mut next_in_bucket,
                &mut prev_in_bucket,
            );

            let new_degree = previous_degree.saturating_sub(1);
            degrees[neighbor_index] = new_degree;
            bucket_insert_head(
                neighbor_index,
                new_degree,
                &mut bucket_heads,
                &mut next_in_bucket,
                &mut prev_in_bucket,
            );

            if new_degree < min_degree {
                min_degree = new_degree;
            }
        }
    }

    removal_order.reverse();
    DegeneracyDecomposition { smallest_last_order: removal_order }
}

fn core_numbers<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    if n == 0 {
        return Vec::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let mut core_numbers: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let max_degree = core_numbers.iter().copied().max().unwrap_or(0);
    let mut bins = vec![0usize; max_degree + 1];
    for &degree in &core_numbers {
        bins[degree] += 1;
    }

    let mut start = 0usize;
    for bin in &mut bins {
        let count = *bin;
        *bin = start;
        start += count;
    }

    let mut positions = vec![0usize; n];
    let mut ordering = vec![0usize; n];

    for node_index in 0..n {
        let degree = core_numbers[node_index];
        let position = bins[degree];
        positions[node_index] = position;
        ordering[position] = node_index;
        bins[degree] += 1;
    }

    for degree in (1..=max_degree).rev() {
        bins[degree] = bins[degree - 1];
    }
    bins[0] = 0;

    for position in 0..n {
        let node_index = ordering[position];
        let node = nodes[node_index];

        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if core_numbers[neighbor_index] <= core_numbers[node_index] {
                continue;
            }

            let neighbor_degree = core_numbers[neighbor_index];
            let neighbor_position = positions[neighbor_index];
            let first_in_bin_position = bins[neighbor_degree];
            let first_in_bin_node = ordering[first_in_bin_position];

            if neighbor_index != first_in_bin_node {
                ordering[neighbor_position] = first_in_bin_node;
                ordering[first_in_bin_position] = neighbor_index;
                positions[neighbor_index] = first_in_bin_position;
                positions[first_in_bin_node] = neighbor_position;
            }

            bins[neighbor_degree] += 1;
            core_numbers[neighbor_index] -= 1;
        }
    }

    core_numbers
}

/// Degeneracy (smallest-last) ordering.
///
/// This is the linear-time bucket-queue smallest-last algorithm of Matula and
/// Beck. It returns the final ordering obtained by reversing the minimum-degree
/// removal sequence, so denser-core vertices appear first. Exact tie order
/// within the same degree bucket is not part of the contract; the guarantee is
/// that the returned order is a valid smallest-last ordering.
///
/// References:
/// - Matula, D. W., & Beck, L. L. (1983). Smallest-last ordering and clustering
///   and graph coloring algorithms. *Journal of the ACM*, 30(3), 417-427. DOI:
///   `10.1145/2402.322385`
/// - Batagelj, V., & Zaversnik, M. (2003). An O(m) algorithm for cores
///   decomposition of networks. [arXiv:cs/0310049](https://arxiv.org/abs/cs/0310049)
#[derive(Clone, Copy, Debug, Default)]
pub struct DegeneracySorter;

impl<G> NodeSorter<G> for DegeneracySorter
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        degeneracy_decomposition(graph).smallest_last_order
    }
}

/// Core-number scorer.
///
/// The score of a node is its k-core number. This is computed from the same
/// linear bucket-queue decomposition used by [`DegeneracySorter`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreNumberScorer;

impl<G> NodeScorer<G> for CoreNumberScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        core_numbers(graph)
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

/// Katz centrality scorer.
///
/// This matches the scalar-`beta`, unweighted `NetworkX` Katz centrality
/// contract on undirected simple graphs. The iterative update is
/// `x_{k+1} = alpha * A * x_k + beta`, starting from the all-zero vector.
/// When `normalized` is enabled, the converged score vector is rescaled to
/// unit Euclidean norm.
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
}

impl Default for KatzCentralityScorer {
    #[inline]
    fn default() -> Self {
        Self::new(0.1, 1.0, 1000, 1.0e-6, true)
    }
}

impl KatzCentralityScorerBuilder {
    /// Creates a builder initialized with NetworkX-compatible defaults.
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
