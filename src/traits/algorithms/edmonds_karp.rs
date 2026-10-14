//! The [`EdmondsKarp`] trait: the Edmonds-Karp algorithm for the maximum s-t
//! flow problem on a directed, capacitated graph.
//!
//! Reference: J. Edmonds, R. M. Karp, "Theoretical improvements in algorithmic
//! efficiency for network flow problems", Journal of the ACM 19 (1972),
//! 248-264. The same shortest-augmenting-path rule was independently given by
//! E. A. Dinic (1970).
//!
//! Edmonds-Karp is the Ford-Fulkerson method with a specific augmenting-path
//! rule: each iteration finds a *shortest* (fewest-arc) augmenting path in the
//! residual graph by breadth-first search and saturates it. Because the
//! source-sink residual distance never decreases and strictly increases every
//! `O(E)` augmentations, the algorithm runs in `O(V * E^2)` time regardless of
//! the capacity magnitudes.
//!
//! This is the same maximum-flow problem solved by [`Dinic`], and it shares the
//! residual graph, result, and error types from the `max_flow` module.
//! Dinic is asymptotically faster (`O(V^2 * E)`); Edmonds-Karp is kept as a
//! second, structurally different implementation, which makes the two strong
//! cross-checks for each other.
//!
//! [`Dinic`]: crate::traits::algorithms::Dinic

use alloc::{collections::VecDeque, vec::Vec};

use super::max_flow::{MaxFlowError, MaxFlowResult, build_residual};
use crate::traits::{Finite, Number, SparseValuedMatrix2D, SquareMatrix};

/// Repeatedly augments along shortest residual paths until the sink is
/// unreachable, returning the total flow pushed. Mutates `arc_cap` in place.
fn shortest_augmenting_paths<V: Number>(
    source: usize,
    sink: usize,
    adjacency: &[Vec<usize>],
    arc_to: &[usize],
    arc_cap: &mut [V],
) -> V {
    let zero = V::zero();
    let order = adjacency.len();
    let mut total = zero;
    let mut parent_arc = alloc::vec![usize::MAX; order];
    let mut visited = alloc::vec![false; order];
    let mut queue: VecDeque<usize> = VecDeque::new();

    loop {
        parent_arc.fill(usize::MAX);
        visited.fill(false);
        queue.clear();
        visited[source] = true;
        queue.push_back(source);

        // Breadth-first search records, per vertex, the arc used to reach it,
        // which yields a shortest residual path to the sink.
        while let Some(node) = queue.pop_front() {
            if node == sink {
                break;
            }
            for &arc in &adjacency[node] {
                let next = arc_to[arc];
                if !visited[next] && arc_cap[arc] > zero {
                    visited[next] = true;
                    parent_arc[next] = arc;
                    queue.push_back(next);
                }
            }
        }
        if !visited[sink] {
            break;
        }

        // Bottleneck of the located path, walking back via reverse arcs. `<`
        // and `<=` pick the same minimum (an equivalent mutant, skipped
        // in .cargo/mutants.toml).
        let mut bottleneck = V::max_value();
        let mut node = sink;
        while node != source {
            let arc = parent_arc[node];
            if arc_cap[arc] < bottleneck {
                bottleneck = arc_cap[arc];
            }
            node = arc_to[arc ^ 1];
        }
        // Augment the path by the bottleneck.
        let mut node = sink;
        while node != source {
            let arc = parent_arc[node];
            arc_cap[arc] -= bottleneck;
            arc_cap[arc ^ 1] += bottleneck;
            node = arc_to[arc ^ 1];
        }
        total += bottleneck;
    }
    total
}

/// The Edmonds-Karp algorithm for the maximum s-t flow problem.
///
/// Entry `(i, j)` of the matrix is the capacity of the directed arc `i -> j`.
/// Capacities must be finite and non-negative, and self-loops are ignored. The
/// reverse residual arcs are created internally with zero capacity, so
/// antiparallel input arcs remain independent.
pub trait EdmondsKarp: SparseValuedMatrix2D + SquareMatrix {
    /// Computes a maximum flow from `source` to `sink`.
    ///
    /// # Arguments
    ///
    /// * `source` - the index of the source node, in `0..order`.
    /// * `sink` - the index of the sink node, in `0..order`.
    ///
    /// # Complexity
    ///
    /// `O(V * E^2)` time and `O(V + E)` space, independent of the capacity
    /// magnitudes.
    ///
    /// # Errors
    ///
    /// Returns an error when `source` or `sink` is out of range, when they
    /// coincide, or when a non-self-loop arc carries a non-finite or negative
    /// capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// // Diamond network: two internal paths each bottlenecked to 2 units.
    /// let edges: ValuedCSR2D<usize, usize, usize, u64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, u64>>::default()
    ///         .expected_number_of_edges(4)
    ///         .expected_shape((4, 4))
    ///         .edges(vec![(0, 1, 3u64), (0, 2, 2), (1, 3, 2), (2, 3, 3)].into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    /// let result = edges.edmonds_karp(0, 3).unwrap();
    /// assert_eq!(result.max_flow(), 4);
    /// ```
    fn edmonds_karp(
        &self,
        source: usize,
        sink: usize,
    ) -> Result<MaxFlowResult<Self::Value>, MaxFlowError>
    where
        Self::Value: Number + Finite;
}

impl<M: SparseValuedMatrix2D + SquareMatrix + ?Sized> EdmondsKarp for M {
    fn edmonds_karp(
        &self,
        source: usize,
        sink: usize,
    ) -> Result<MaxFlowResult<Self::Value>, MaxFlowError>
    where
        Self::Value: Number + Finite,
    {
        let mut residual = build_residual(self, source, sink)?;
        let max_flow = shortest_augmenting_paths(
            source,
            sink,
            &residual.adjacency,
            &residual.arc_to,
            &mut residual.arc_cap,
        );
        Ok(residual.finish(max_flow))
    }
}
