//! Composable clique ranking for MCES result selection.
//!
//! When the MCES pipeline finds multiple maximum cliques of the same size,
//! a ranking system selects the best one. This module provides:
//!
//! - [`CliqueInfo`] — trait for accessing precomputed clique data.
//! - [`EagerCliqueInfo`] — concrete implementation that eagerly computes
//!   matched edges, vertex matches, and fragment structure.
//! - [`CliqueRanker`] — trait for comparing two cliques; composable via
//!   [`then`](CliqueRankerExt::then) into zero-cost [`ChainedRanker`] chains.
//! - [`FragmentCountRanker`] — ranks by number of connected fragments (fewer =
//!   better).
//!
//! The ranking chain is fully monomorphized at compile time — no dynamic
//! dispatch, no vtable overhead.
//!
//! # Example
//!
//! ```
//! use core::cmp::Ordering;
//!
//! use geometric_traits::prelude::*;
//!
//! // Custom ranker: prefer more vertex matches.
//! struct MoreVerticesRanker;
//!
//! impl<I: CliqueInfo> CliqueRanker<I> for MoreVerticesRanker {
//!     fn compare(&self, a: &I, b: &I) -> Ordering {
//!         // More vertices = better → reverse order
//!         b.vertex_matches().len().cmp(&a.vertex_matches().len())
//!     }
//! }
//!
//! // Chain: fewer fragments first, then more vertices.
//! let _ranker = FragmentCountRanker.then(MoreVerticesRanker);
//! ```

use alloc::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    vec::Vec,
};
use core::cmp::Ordering;

use super::vertex_match_inference::infer_vertex_matches;

/// A matched edge pair: `((src1, dst1), (src2, dst2))` from two graphs.
pub type MatchedEdgePair<N> = ((N, N), (N, N));

/// Trait for accessing precomputed clique data used by rankers.
///
/// Implementors provide matched edges, vertex matches, and fragment
/// structure. All data is expected to be precomputed at construction.
pub trait CliqueInfo {
    /// Node ID type (same for both graphs).
    type NodeId: Eq + Copy + Ord + core::fmt::Debug;

    /// The clique vertex indices (into the modular product).
    fn clique(&self) -> &[usize];

    /// Matched edge pairs: `((src1, dst1), (src2, dst2))` from the
    /// original graphs.
    fn matched_edges(&self) -> &[MatchedEdgePair<Self::NodeId>];

    /// Matched vertex pairs `(v1, v2)` inferred from edge matches.
    fn vertex_matches(&self) -> &[(Self::NodeId, Self::NodeId)];

    /// Number of connected components (fragments) in the matched edge
    /// subgraph.
    fn fragment_count(&self) -> usize;

    /// Number of matched edges in the largest connected fragment.
    fn largest_fragment_edge_count(&self) -> usize;

    /// Number of matched vertices in the largest connected fragment.
    fn largest_fragment_atom_count(&self) -> usize;

    /// Legacy alias for the edge-based largest fragment size.
    #[inline]
    fn largest_fragment_size(&self) -> usize {
        self.largest_fragment_edge_count()
    }
}

/// Eagerly precomputed clique information.
///
/// Constructed from a maximum clique result, vertex pairs, edge maps, and
/// a disambiguation closure. All fields are computed at construction time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EagerCliqueInfo<N> {
    clique: Vec<usize>,
    matched_edges: Vec<MatchedEdgePair<N>>,
    vertex_matches: Vec<(N, N)>,
    fragment_count: usize,
    largest_fragment_edge_count: usize,
    largest_fragment_atom_count: usize,
}

impl<N> EagerCliqueInfo<N>
where
    N: Eq + Copy + Ord + core::fmt::Debug,
{
    /// Constructs a new `EagerCliqueInfo` from MCES pipeline outputs.
    ///
    /// # Parameters
    ///
    /// - `clique` — vertex indices in the modular product (max clique result).
    /// - `vertex_pairs` — from [`ModularProductResult::vertex_pairs()`], maps
    ///   product vertex → `(lg1_vertex, lg2_vertex)`.
    /// - `edge_map_first` — from [`LineGraphResult::edge_map()`] of the first
    ///   graph.
    /// - `edge_map_second` — same for the second graph.
    /// - `disambiguate` — closure for isolated edge disambiguation (see
    ///   [`infer_vertex_matches`]).
    ///
    /// [`ModularProductResult::vertex_pairs()`]: crate::traits::algorithms::modular_product::ModularProductResult::vertex_pairs
    /// [`LineGraphResult::edge_map()`]: crate::traits::algorithms::line_graph::LineGraphResult::edge_map
    #[must_use]
    pub fn new<F>(
        clique: Vec<usize>,
        vertex_pairs: &[(usize, usize)],
        edge_map_first: &[(N, N)],
        edge_map_second: &[(N, N)],
        disambiguate: F,
    ) -> Self
    where
        F: FnMut(N, N, N, N) -> bool,
    {
        // 1. Decode matched edges.
        let matched_edges: Vec<MatchedEdgePair<N>> = clique
            .iter()
            .map(|&k| {
                let (lg1, lg2) = vertex_pairs[k];
                (edge_map_first[lg1], edge_map_second[lg2])
            })
            .collect();

        // 2. Infer vertex matches.
        let vertex_matches = infer_vertex_matches(
            &clique,
            vertex_pairs,
            edge_map_first,
            edge_map_second,
            disambiguate,
        );

        // 3. Compute fragment structure from G1 edges.
        let g1_edges: Vec<(N, N)> = matched_edges.iter().map(|&(e1, _)| e1).collect();
        let (fragment_count, largest_fragment_edge_count, largest_fragment_atom_count) =
            compute_fragments(&g1_edges);

        Self {
            clique,
            matched_edges,
            vertex_matches,
            fragment_count,
            largest_fragment_edge_count,
            largest_fragment_atom_count,
        }
    }
}

impl<N> CliqueInfo for EagerCliqueInfo<N>
where
    N: Eq + Copy + Ord + core::fmt::Debug,
{
    type NodeId = N;

    #[inline]
    fn clique(&self) -> &[usize] {
        &self.clique
    }

    #[inline]
    fn matched_edges(&self) -> &[MatchedEdgePair<N>] {
        &self.matched_edges
    }

    #[inline]
    fn vertex_matches(&self) -> &[(N, N)] {
        &self.vertex_matches
    }

    #[inline]
    fn fragment_count(&self) -> usize {
        self.fragment_count
    }

    #[inline]
    fn largest_fragment_edge_count(&self) -> usize {
        self.largest_fragment_edge_count
    }

    #[inline]
    fn largest_fragment_atom_count(&self) -> usize {
        self.largest_fragment_atom_count
    }
}

/// Computes the number of connected components and the sizes of the largest
/// component in an edge list.
///
/// Uses BFS on a `BTreeMap`-based adjacency list to support arbitrary
/// (non-contiguous) node ID types.
///
/// Returns `(0, 0, 0)` for an empty edge list.
fn compute_fragments<N: Eq + Copy + Ord>(edges: &[(N, N)]) -> (usize, usize, usize) {
    if edges.is_empty() {
        return (0, 0, 0);
    }

    // Build adjacency list.
    let mut adj: BTreeMap<N, Vec<N>> = BTreeMap::new();
    for &(u, v) in edges {
        adj.entry(u).or_default().push(v);
        adj.entry(v).or_default().push(u);
    }

    // BFS to find connected components.
    let mut visited: BTreeSet<N> = BTreeSet::new();
    let mut queue: VecDeque<N> = VecDeque::new();
    let mut num_components = 0usize;
    let mut largest_component_edges = 0usize;
    let mut largest_component_vertices = 0usize;

    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }

        // BFS from start.
        visited.insert(start);
        queue.push_back(start);
        let mut component_vertices: Vec<N> = Vec::new();

        while let Some(v) = queue.pop_front() {
            component_vertices.push(v);
            if let Some(neighbors) = adj.get(&v) {
                for &u in neighbors {
                    if visited.insert(u) {
                        queue.push_back(u);
                    }
                }
            }
        }

        // Count edges in this component: edges whose both endpoints are in
        // this component. Since we built adjacency from the edge list, we
        // can count by summing degrees / 2.
        let component_set: BTreeSet<N> = component_vertices.iter().copied().collect();
        let component_edges = edges
            .iter()
            .filter(|&&(u, v)| component_set.contains(&u) && component_set.contains(&v))
            .count();

        num_components += 1;
        if component_edges > largest_component_edges {
            largest_component_edges = component_edges;
        }
        if component_vertices.len() > largest_component_vertices {
            largest_component_vertices = component_vertices.len();
        }
    }

    (num_components, largest_component_edges, largest_component_vertices)
}

/// Trait for comparing two cliques to determine ranking order.
///
/// Composable via [`then`](CliqueRankerExt::then) into zero-cost
/// [`ChainedRanker`] chains. The entire chain is monomorphized at
/// compile time — no dynamic dispatch.
pub trait CliqueRanker<I: CliqueInfo> {
    /// Compares two cliques. Returns `Less` if `a` is better, `Greater`
    /// if `b` is better, `Equal` if indistinguishable by this criterion.
    fn compare(&self, a: &I, b: &I) -> Ordering;
}

/// Extension trait providing the `.then()` chain combinator.
///
/// Separate from [`CliqueRanker`] to avoid requiring the `I` type parameter
/// at chain-construction time (it's only needed when `compare` is called).
pub trait CliqueRankerExt: Sized {
    /// Chains this ranker with another. The first ranker breaks ties first;
    /// the second ranker is consulted only when the first returns `Equal`.
    #[inline]
    fn then<R>(self, other: R) -> ChainedRanker<Self, R> {
        ChainedRanker { first: self, second: other }
    }
}

impl<T> CliqueRankerExt for T {}

/// A chained pair of rankers applied lexicographically.
///
/// Created by [`CliqueRankerExt::then`]. Fully monomorphized — the compiler
/// inlines both comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChainedRanker<R1, R2> {
    first: R1,
    second: R2,
}

impl<I: CliqueInfo, R1: CliqueRanker<I>, R2: CliqueRanker<I>> CliqueRanker<I>
    for ChainedRanker<R1, R2>
{
    #[inline]
    fn compare(&self, a: &I, b: &I) -> Ordering {
        self.first.compare(a, b).then_with(|| self.second.compare(a, b))
    }
}

/// Ranks cliques by fragment count: fewer connected fragments is better.
///
/// This is the first tiebreaker in the RASCAL-style ranking (criterion 2).
/// A single contiguous matched subgraph (1 fragment) is preferred over
/// a scattered match with the same number of edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FragmentCountRanker;

impl<I: CliqueInfo> CliqueRanker<I> for FragmentCountRanker {
    #[inline]
    fn compare(&self, a: &I, b: &I) -> Ordering {
        a.fragment_count().cmp(&b.fragment_count())
    }
}

/// Ranks cliques by largest fragment size: larger is better.
///
/// This is the second tiebreaker in the RASCAL-style ranking (criterion 3).
/// Among cliques with the same fragment count, prefer the one whose largest
/// connected component contains more edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LargestFragmentRanker;

impl<I: CliqueInfo> CliqueRanker<I> for LargestFragmentRanker {
    #[inline]
    fn compare(&self, a: &I, b: &I) -> Ordering {
        b.largest_fragment_edge_count().cmp(&a.largest_fragment_edge_count())
    }
}

/// Fragment-size metric used by [`LargestFragmentMetricRanker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LargestFragmentMetric {
    /// Prefer the clique with more matched edges in its largest fragment.
    Edges,
    /// Prefer the clique with more matched vertices in its largest fragment.
    Atoms,
}

/// Ranks cliques by largest fragment size using a selectable metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LargestFragmentMetricRanker {
    metric: LargestFragmentMetric,
}

impl LargestFragmentMetricRanker {
    /// Creates a new metric-selectable largest-fragment ranker.
    #[inline]
    #[must_use]
    pub const fn new(metric: LargestFragmentMetric) -> Self {
        Self { metric }
    }
}

impl Default for LargestFragmentMetricRanker {
    #[inline]
    fn default() -> Self {
        Self::new(LargestFragmentMetric::Edges)
    }
}

impl<I: CliqueInfo> CliqueRanker<I> for LargestFragmentMetricRanker {
    #[inline]
    fn compare(&self, a: &I, b: &I) -> Ordering {
        match self.metric {
            LargestFragmentMetric::Edges => {
                b.largest_fragment_edge_count().cmp(&a.largest_fragment_edge_count())
            }
            LargestFragmentMetric::Atoms => {
                b.largest_fragment_atom_count().cmp(&a.largest_fragment_atom_count())
            }
        }
    }
}

/// Ranks cliques using a user-provided comparison closure.
///
/// This is the escape hatch for any custom ranking logic: edge weight costs,
/// domain-specific scoring, or any criterion not covered by the built-in
/// rankers.
///
/// # Example
///
/// ```
/// use core::cmp::Ordering;
///
/// use geometric_traits::prelude::*;
///
/// // Prefer cliques with more vertex matches.
/// let _ranker = FnRanker::new(|a: &EagerCliqueInfo<u32>, b: &EagerCliqueInfo<u32>| {
///     b.vertex_matches().len().cmp(&a.vertex_matches().len())
/// });
/// ```
pub struct FnRanker<F> {
    compare_fn: F,
}

impl<F> FnRanker<F> {
    /// Creates a new `FnRanker` from a comparison closure.
    #[inline]
    #[must_use]
    pub fn new(compare_fn: F) -> Self {
        Self { compare_fn }
    }
}

impl<I: CliqueInfo, F: Fn(&I, &I) -> Ordering> CliqueRanker<I> for FnRanker<F> {
    #[inline]
    fn compare(&self, a: &I, b: &I) -> Ordering {
        (self.compare_fn)(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fragments_empty() {
        let (count, largest_edges, largest_atoms) = compute_fragments::<u32>(&[]);
        assert_eq!(count, 0);
        assert_eq!(largest_edges, 0);
        assert_eq!(largest_atoms, 0);
    }

    #[test]
    fn test_compute_fragments_single_edge() {
        let (count, largest_edges, largest_atoms) = compute_fragments(&[(0u32, 1)]);
        assert_eq!(count, 1);
        assert_eq!(largest_edges, 1);
        assert_eq!(largest_atoms, 2);
    }

    #[test]
    fn test_compute_fragments_path() {
        // Path: 0-1-2-3 → 1 component, 3 edges.
        let (count, largest_edges, largest_atoms) = compute_fragments(&[(0u32, 1), (1, 2), (2, 3)]);
        assert_eq!(count, 1);
        assert_eq!(largest_edges, 3);
        assert_eq!(largest_atoms, 4);
    }

    #[test]
    fn test_compute_fragments_two_disjoint() {
        // Two disjoint edges: (0,1) and (5,6) → 2 components, 1 edge each.
        let (count, largest_edges, largest_atoms) = compute_fragments(&[(0u32, 1), (5, 6)]);
        assert_eq!(count, 2);
        assert_eq!(largest_edges, 1);
        assert_eq!(largest_atoms, 2);
    }

    #[test]
    fn test_compute_fragments_uneven() {
        // Path (0,1,2) + isolated edge (5,6) → 2 components, largest has 2 edges.
        let (count, largest_edges, largest_atoms) = compute_fragments(&[(0u32, 1), (1, 2), (5, 6)]);
        assert_eq!(count, 2);
        assert_eq!(largest_edges, 2);
        assert_eq!(largest_atoms, 3);
    }

    #[test]
    fn test_compute_fragments_triangle() {
        let (count, largest_edges, largest_atoms) = compute_fragments(&[(0u32, 1), (1, 2), (0, 2)]);
        assert_eq!(count, 1);
        assert_eq!(largest_edges, 3);
        assert_eq!(largest_atoms, 3);
    }
}
