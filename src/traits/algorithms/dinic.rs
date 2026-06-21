//! The [`Dinic`] trait: Dinic's (Dinitz') algorithm for the maximum s-t flow
//! problem on a directed, capacitated graph.
//!
//! Reference: E. A. Dinic, "Algorithm for solution of a problem of maximum
//! flow in networks with power estimation", Soviet Mathematics Doklady 11
//! (1970), 1277-1280.
//!
//! The algorithm repeats two steps until the sink is no longer reachable in the
//! residual graph: a breadth-first search assigns each vertex a level equal to
//! its residual distance from the source, and a blocking flow is then pushed
//! through the level graph (only arcs that go from one level to the next are
//! admissible). Each blocking-flow phase strictly increases the source-sink
//! distance, so at most `V - 1` phases run, each costing `O(V * E)` with the
//! current-arc optimization, for an overall `O(V^2 * E)` bound that is
//! independent of the capacity magnitudes.
//!
//! The shared residual graph, result, and error types live in the `max_flow`
//! module, which this algorithm reuses.

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::Zero;

use super::max_flow::{MaxFlowError, MaxFlowResult, build_residual};
use crate::traits::{Finite, Number, SparseValuedMatrix2D, SquareMatrix};

/// Assigns each vertex its residual breadth-first distance from `source` in
/// `levels` (`usize::MAX` marks an unreached vertex), and returns whether the
/// sink is reachable. A false return is the Dinic phase-loop termination test.
fn bfs_levels<V: Number>(
    source: usize,
    sink: usize,
    adjacency: &[Vec<usize>],
    arc_to: &[usize],
    arc_cap: &[V],
    levels: &mut [usize],
    queue: &mut VecDeque<usize>,
) -> bool {
    let zero = V::zero();
    levels.fill(usize::MAX);
    levels[source] = 0;
    queue.clear();
    queue.push_back(source);
    while let Some(node) = queue.pop_front() {
        for &arc in &adjacency[node] {
            let next = arc_to[arc];
            if arc_cap[arc] > zero && levels[next] == usize::MAX {
                levels[next] = levels[node] + 1;
                queue.push_back(next);
            }
        }
    }
    levels[sink] != usize::MAX
}

/// Pushes a blocking flow through the current level graph and returns its
/// value.
///
/// This is an iterative depth-first search with the current-arc optimization:
/// `current_arc` records, per vertex, the next outgoing arc to try, so each arc
/// is advanced past at most once per phase. On reaching the sink the path
/// bottleneck is augmented and the search retreats to the tail of the first arc
/// the augmentation saturated. Reaching a dead end retreats one vertex and
/// skips the arc that led into it.
fn blocking_flow<V: Number>(
    source: usize,
    sink: usize,
    adjacency: &[Vec<usize>],
    arc_to: &[usize],
    arc_cap: &mut [V],
    levels: &[usize],
    current_arc: &mut [usize],
) -> V {
    let zero = V::zero();
    let mut total = zero;
    let mut node_stack: Vec<usize> = Vec::new();
    let mut arc_stack: Vec<usize> = Vec::new();
    node_stack.push(source);

    while let Some(&node) = node_stack.last() {
        if node == sink {
            // Bottleneck of the located augmenting path.
            let mut bottleneck = V::max_value();
            for &arc in &arc_stack {
                if arc_cap[arc] < bottleneck {
                    bottleneck = arc_cap[arc];
                }
            }
            // Augment every arc, recording the first one that saturates.
            let path_len = arc_stack.len();
            let mut first_saturated = path_len;
            for (position, &arc) in arc_stack.iter().enumerate() {
                arc_cap[arc] -= bottleneck;
                arc_cap[arc ^ 1] += bottleneck;
                if first_saturated == path_len && arc_cap[arc] == zero {
                    first_saturated = position;
                }
            }
            total += bottleneck;
            // Retreat to the tail of the first saturated arc and continue.
            node_stack.truncate(first_saturated + 1);
            arc_stack.truncate(first_saturated);
            continue;
        }

        let mut advanced = false;
        while current_arc[node] < adjacency[node].len() {
            let arc = adjacency[node][current_arc[node]];
            let next = arc_to[arc];
            if arc_cap[arc] > zero && levels[next] == levels[node] + 1 {
                node_stack.push(next);
                arc_stack.push(arc);
                advanced = true;
                break;
            }
            current_arc[node] += 1;
        }
        if !advanced {
            node_stack.pop();
            if let Some(&parent) = node_stack.last() {
                arc_stack.pop();
                current_arc[parent] += 1;
            }
        }
    }
    total
}

/// Dinic's (Dinitz') algorithm for the maximum s-t flow problem.
///
/// Entry `(i, j)` of the matrix is the capacity of the directed arc `i -> j`.
/// Capacities must be finite and non-negative, and self-loops are ignored. The
/// reverse residual arcs are created internally with zero capacity, so
/// antiparallel input arcs remain independent.
pub trait Dinic: SparseValuedMatrix2D + SquareMatrix {
    /// Computes a maximum flow from `source` to `sink`.
    ///
    /// # Arguments
    ///
    /// * `source` - the index of the source node, in `0..order`.
    /// * `sink` - the index of the sink node, in `0..order`.
    ///
    /// # Complexity
    ///
    /// `O(V^2 * E)` time and `O(V + E)` space, independent of the capacity
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
    /// let result = edges.dinic(0, 3).unwrap();
    /// assert_eq!(result.max_flow(), 4);
    /// ```
    fn dinic(&self, source: usize, sink: usize) -> Result<MaxFlowResult<Self::Value>, MaxFlowError>
    where
        Self::Value: Number + Finite;
}

impl<M: SparseValuedMatrix2D + SquareMatrix + ?Sized> Dinic for M {
    fn dinic(&self, source: usize, sink: usize) -> Result<MaxFlowResult<Self::Value>, MaxFlowError>
    where
        Self::Value: Number + Finite,
    {
        let mut residual = build_residual(self, source, sink)?;
        let order = residual.adjacency.len();

        let mut max_flow = Self::Value::zero();
        let mut levels = alloc::vec![usize::MAX; order];
        let mut current_arc = alloc::vec![0usize; order];
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Phase loop: rebuild the level graph, push a blocking flow, repeat
        // until the sink is no longer reachable in the residual graph.
        while bfs_levels(
            source,
            sink,
            &residual.adjacency,
            &residual.arc_to,
            &residual.arc_cap,
            &mut levels,
            &mut queue,
        ) {
            current_arc.fill(0);
            // A blocking flow always pushes a positive amount while the sink is
            // reachable, and each phase strictly increases the source-sink
            // residual distance, so the loop runs at most `V - 1` times.
            max_flow += blocking_flow(
                source,
                sink,
                &residual.adjacency,
                &residual.arc_to,
                &mut residual.arc_cap,
                &levels,
                &mut current_arc,
            );
        }

        Ok(residual.finish(max_flow))
    }
}
