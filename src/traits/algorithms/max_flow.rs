//! Shared machinery for the maximum s-t flow algorithms.
//!
//! Every concrete max-flow algorithm in this crate (for example [`Dinic`] and
//! [`EdmondsKarp`]) computes the same object: a maximum flow value, a feasible
//! per-arc flow, and a minimum cut on a directed, capacitated graph. The result
//! and error types and the residual-graph construction are therefore factored
//! out here, so each algorithm only has to supply its own augmentation loop.
//!
//! Entry `(i, j)` of the matrix is the capacity of the directed arc `i -> j`.
//! Capacities must be finite and non-negative, and self-loops are ignored. The
//! matrix is read forward only (its rows are the out-arcs), and the reverse
//! residual arcs are created internally with zero capacity, so antiparallel
//! input arcs `i -> j` and `j -> i` stay independent.
//!
//! [`Dinic`]: crate::traits::algorithms::Dinic
//! [`EdmondsKarp`]: crate::traits::algorithms::EdmondsKarp

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::{AsPrimitive, Zero};

use crate::traits::{Finite, Number, SparseValuedMatrix2D, SquareMatrix};

/// Error enumeration shared by the maximum-flow algorithms.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MaxFlowError {
    /// The source node index is not smaller than the number of nodes.
    #[error("The source node {source_id} is out of range for a graph with {order} nodes.")]
    SourceOutOfRange {
        /// The offending source index.
        source_id: usize,
        /// The number of nodes in the graph.
        order: usize,
    },
    /// The sink node index is not smaller than the number of nodes.
    #[error("The sink node {sink} is out of range for a graph with {order} nodes.")]
    SinkOutOfRange {
        /// The offending sink index.
        sink: usize,
        /// The number of nodes in the graph.
        order: usize,
    },
    /// The source and the sink coincide, so the flow is undefined.
    #[error("The source and the sink must differ, but both are {node}.")]
    SourceEqualsSink {
        /// The shared source and sink index.
        node: usize,
    },
    /// An arc carries a non-finite capacity.
    #[error("Found a non-finite capacity on arc ({source_id}, {destination_id}).")]
    NonFiniteCapacity {
        /// Source node identifier of the offending arc.
        source_id: usize,
        /// Destination node identifier of the offending arc.
        destination_id: usize,
    },
    /// An arc carries a negative capacity.
    #[error("Found a negative capacity on arc ({source_id}, {destination_id}).")]
    NegativeCapacity {
        /// Source node identifier of the offending arc.
        source_id: usize,
        /// Destination node identifier of the offending arc.
        destination_id: usize,
    },
}

/// Result of a maximum s-t flow computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxFlowResult<V> {
    /// The value of the maximum flow from the source to the sink.
    max_flow: V,
    /// Per-arc flow `(source, destination, flow)` for every original arc that
    /// carries a strictly positive flow.
    flows: Vec<(usize, usize, V)>,
    /// The arcs `(source, destination)` of a minimum cut: the original arcs
    /// that leave the source side of the residual partition. Their
    /// capacities sum to the maximum flow.
    min_cut: Vec<(usize, usize)>,
    /// The source side of the minimum cut: `source_side[node]` is `true` when
    /// `node` is still reachable from the source in the residual graph.
    source_side: Vec<bool>,
}

impl<V: Copy> MaxFlowResult<V> {
    /// The value of the maximum flow from the source to the sink.
    #[must_use]
    pub fn max_flow(&self) -> V {
        self.max_flow
    }

    /// The per-arc flow on every original arc that carries positive flow, as
    /// `(source, destination, flow)` triples.
    #[must_use]
    pub fn flows(&self) -> &[(usize, usize, V)] {
        &self.flows
    }

    /// The arcs crossing a minimum cut, as `(source, destination)` pairs. Their
    /// capacities sum to [`MaxFlowResult::max_flow`].
    #[must_use]
    pub fn min_cut(&self) -> &[(usize, usize)] {
        &self.min_cut
    }

    /// The source side of the minimum cut: `source_side()[node]` is `true` when
    /// `node` remains reachable from the source in the residual graph.
    #[must_use]
    pub fn source_side(&self) -> &[bool] {
        &self.source_side
    }
}

/// Residual graph in paired-arc form shared by the max-flow algorithms.
///
/// Arc `e` has reverse arc `e ^ 1`, and `arc_to[e ^ 1]` is the tail of `e`.
/// Forward arcs start at their capacity, reverse arcs at zero. Algorithms push
/// flow by mutating [`Residual::arc_cap`] in place, then call
/// [`Residual::finish`] to read out the flow and the minimum cut.
pub(super) struct Residual<V> {
    /// Per-node lists of incident residual arc indices.
    pub(super) adjacency: Vec<Vec<usize>>,
    /// Head node of each residual arc.
    pub(super) arc_to: Vec<usize>,
    /// Remaining residual capacity of each arc.
    pub(super) arc_cap: Vec<V>,
    /// Original forward arcs as `(tail, head, arc_index, original_capacity)`.
    forward_arcs: Vec<(usize, usize, usize, V)>,
    /// The source node index.
    source: usize,
}

impl<V: Number> Residual<V> {
    /// Reads out the flow and the minimum cut after an algorithm has pushed a
    /// maximum flow of value `max_flow` into [`Residual::arc_cap`].
    pub(super) fn finish(self, max_flow: V) -> MaxFlowResult<V> {
        let zero = V::zero();
        let order = self.adjacency.len();

        // Per-arc flow is the capacity consumed on each original forward arc.
        let mut flows: Vec<(usize, usize, V)> = Vec::new();
        for &(tail, head, forward, capacity) in &self.forward_arcs {
            let flow = capacity - self.arc_cap[forward];
            if flow > zero {
                flows.push((tail, head, flow));
            }
        }

        // The source side is everything still reachable from the source in the
        // residual graph. Original arcs leaving it form the minimum cut.
        let mut source_side = alloc::vec![false; order];
        source_side[self.source] = true;
        let mut queue = VecDeque::new();
        queue.push_back(self.source);
        while let Some(node) = queue.pop_front() {
            for &arc in &self.adjacency[node] {
                let next = self.arc_to[arc];
                if self.arc_cap[arc] > zero && !source_side[next] {
                    source_side[next] = true;
                    queue.push_back(next);
                }
            }
        }
        let mut min_cut: Vec<(usize, usize)> = Vec::new();
        for &(tail, head, _forward, _capacity) in &self.forward_arcs {
            if source_side[tail] && !source_side[head] {
                min_cut.push((tail, head));
            }
        }

        MaxFlowResult { max_flow, flows, min_cut, source_side }
    }
}

/// Builds the residual graph from a square capacity matrix, validating the
/// source, the sink, and every capacity.
///
/// # Errors
///
/// Returns an error when `source` or `sink` is out of range, when they
/// coincide, or when a non-self-loop arc carries a non-finite or negative
/// capacity.
pub(super) fn build_residual<M>(
    matrix: &M,
    source: usize,
    sink: usize,
) -> Result<Residual<M::Value>, MaxFlowError>
where
    M: SparseValuedMatrix2D + SquareMatrix + ?Sized,
    M::Value: Number + Finite,
{
    let order: usize = matrix.order().as_();
    if source >= order {
        return Err(MaxFlowError::SourceOutOfRange { source_id: source, order });
    }
    if sink >= order {
        return Err(MaxFlowError::SinkOutOfRange { sink, order });
    }
    if source == sink {
        return Err(MaxFlowError::SourceEqualsSink { node: source });
    }

    let zero = M::Value::zero();

    // Map each node index to its matrix row handle so a row can be addressed by
    // `usize`.
    let handles: Vec<M::Index> = matrix.row_indices().collect();

    let mut adjacency: Vec<Vec<usize>> = alloc::vec![Vec::new(); order];
    let mut arc_to: Vec<usize> = Vec::new();
    let mut arc_cap: Vec<M::Value> = Vec::new();
    let mut forward_arcs: Vec<(usize, usize, usize, M::Value)> = Vec::new();

    for (tail, &row) in handles.iter().enumerate() {
        for (column, capacity) in matrix.sparse_row(row).zip(matrix.sparse_row_values(row)) {
            let head = column.as_();
            if tail == head {
                // Self-loops cannot carry s-t flow and are ignored.
                continue;
            }
            if !capacity.is_finite() {
                return Err(MaxFlowError::NonFiniteCapacity {
                    source_id: tail,
                    destination_id: head,
                });
            }
            if capacity < zero {
                return Err(MaxFlowError::NegativeCapacity {
                    source_id: tail,
                    destination_id: head,
                });
            }
            if capacity == zero {
                continue;
            }
            let forward = arc_to.len();
            adjacency[tail].push(forward);
            arc_to.push(head);
            arc_cap.push(capacity);
            adjacency[head].push(forward + 1);
            arc_to.push(tail);
            arc_cap.push(zero);
            forward_arcs.push((tail, head, forward, capacity));
        }
    }

    Ok(Residual { adjacency, arc_to, arc_cap, forward_arcs, source })
}
