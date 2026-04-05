//! Graph-level node ordering and node scoring primitives.
//!
//! This module separates:
//! - direct ordering algorithms, represented by [`NodeSorter`]
//! - reusable per-node metrics, represented by [`NodeScorer`]
//!
//! Not every ordering is naturally "sort by one local score". For example,
//! degeneracy and DSATUR are procedural orderings and are therefore modeled
//! directly as sorters. Metrics such as second-order degree are modeled as
//! scorers and can be paired with generic ascending or descending score
//! sorters.

mod betweenness_centrality;
mod closeness_centrality;
mod degeneracy;
mod degree;
mod dsatur;
mod katz_centrality;
mod local_clustering;
mod pagerank;
mod triangles;

use alloc::vec::Vec;

pub use betweenness_centrality::{BetweennessCentralityScorer, BetweennessCentralityScorerBuilder};
pub use closeness_centrality::{ClosenessCentralityScorer, ClosenessCentralityScorerBuilder};
pub use degeneracy::{CoreNumberScorer, DegeneracySorter};
pub use degree::{DegreeScorer, SecondOrderDegreeScorer};
pub use dsatur::DsaturSorter;
pub use katz_centrality::{KatzCentralityScorer, KatzCentralityScorerBuilder};
pub use local_clustering::LocalClusteringCoefficientScorer;
use num_traits::{AsPrimitive, cast};
pub use pagerank::{PageRankScorer, PageRankScorerBuilder};
pub use triangles::TriangleCountScorer;

use crate::traits::{MonopartiteGraph, TotalOrd};

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
///
/// This also covers named largest-first orderings such as Welsh-Powell when
/// composed with [`DegreeScorer`]:
/// `DescendingScoreSorter::new(DegreeScorer)`.
///
/// # Examples
/// ```
/// use geometric_traits::{
///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
///     prelude::*,
///     traits::{
///         SquareMatrix, VocabularyBuilder,
///         algorithms::{DegreeScorer, DescendingScoreSorter, randomized_graphs::star_graph},
///     },
/// };
///
/// let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = star_graph(5);
/// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
///     .expected_number_of_symbols(matrix.order())
///     .symbols((0..matrix.order()).enumerate())
///     .build()
///     .unwrap();
/// let graph = UndiGraph::from((nodes, matrix));
///
/// let order = DescendingScoreSorter::new(DegreeScorer).sort_nodes(&graph);
/// assert_eq!(order, vec![0, 1, 2, 3, 4]);
/// ```
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

pub(super) const PAGERANK_SCORE_SCALE: f64 = 1.0e12;
pub(super) const KATZ_SCORE_SCALE: f64 = 1.0e12;
pub(super) const BETWEENNESS_SCORE_SCALE: f64 = 1.0e12;
pub(super) const CLOSENESS_SCORE_SCALE: f64 = 1.0e12;
pub(super) const LOCAL_CLUSTERING_SCORE_SCALE: f64 = 1.0e12;

#[inline]
pub(super) fn usize_to_f64(value: usize) -> f64 {
    cast::<usize, f64>(value).expect("graph sizes and degrees must fit into f64 for PageRank")
}
