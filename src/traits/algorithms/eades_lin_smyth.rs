//! The [`EadesLinSmyth`] trait: the GR greedy heuristic for the feedback arc
//! set problem.
//!
//! Reference: P. Eades, X. Lin, W. F. Smyth, "A fast and effective heuristic
//! for the feedback arc set problem", Information Processing Letters 47 (1993),
//! 319-323.
//!
//! GR builds a linear vertex arrangement by peeling sinks to the right, sources
//! to the left, and otherwise placing the maximum `out_weight - in_weight`
//! vertex on the left; the feedback arcs are the edges pointing backward in
//! that arrangement. Sources and sinks are held in FIFO queues so peeling is
//! linear, and only the tangled core costs a scan; with the smallest-id
//! tie-break the arrangement, and hence the cut, is deterministic.
//!
//! The graph is read directly through the bidirectional valued-matrix trait:
//! out-edges come from `sparse_row` / `sparse_row_values`, in-edges from
//! `sparse_column` / `sparse_column_values`, so no private adjacency is
//! materialized.

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::{SparseValuedMatrix2D, SquareMatrix, ValuedSizedSparseBiMatrix2D};

/// Per-node bookkeeping mutated during peeling: integer counts drive exact
/// source/sink detection, the float weights rank the core.
struct Degrees {
    out_count: Vec<usize>,
    in_count: Vec<usize>,
    out_weight: Vec<f64>,
    in_weight: Vec<f64>,
    removed: Vec<bool>,
}

/// Removes `node`'s out-edges: each surviving target loses an in-edge and is
/// enqueued as a source once its in-degree hits zero. Out-edges are read
/// straight from the matrix rows; self-loops are skipped.
fn relax_out_edges<M>(
    node: usize,
    matrix: &M,
    handles: &[M::Index],
    degrees: &mut Degrees,
    sources: &mut VecDeque<usize>,
) where
    M: SparseValuedMatrix2D + SquareMatrix + ?Sized,
    M::Index: AsPrimitive<usize>,
    M::Value: AsPrimitive<f64>,
{
    let row = handles[node];
    for (column_id, value) in matrix.sparse_row(row).zip(matrix.sparse_row_values(row)) {
        let target = column_id.as_();
        if node == target || degrees.removed[target] {
            continue;
        }
        degrees.in_count[target] -= 1;
        degrees.in_weight[target] -= value.as_();
        if degrees.in_count[target] == 0 {
            sources.push_back(target);
        }
    }
}

/// Removes `node`'s in-edges: each surviving source loses an out-edge and is
/// enqueued as a sink once its out-degree hits zero (while it still has
/// in-edges). In-edges are read straight from the matrix columns; self-loops
/// are skipped.
fn relax_in_edges<M>(
    node: usize,
    matrix: &M,
    handles: &[M::Index],
    degrees: &mut Degrees,
    sinks: &mut VecDeque<usize>,
) where
    M: SparseValuedMatrix2D + ValuedSizedSparseBiMatrix2D + SquareMatrix + ?Sized,
    M::Index: AsPrimitive<usize>,
    M::Value: AsPrimitive<f64>,
{
    let column = handles[node];
    for (row_id, value) in matrix.sparse_column(column).zip(matrix.sparse_column_values(column)) {
        let source = row_id.as_();
        if node == source || degrees.removed[source] {
            continue;
        }
        degrees.out_count[source] -= 1;
        degrees.out_weight[source] -= value.as_();
        if degrees.out_count[source] == 0 && degrees.in_count[source] > 0 {
            sinks.push_back(source);
        }
    }
}

/// GR peeling loop: returns the arrangement (`order[i]` is the node at position
/// `i`). Each round drains sources (FIFO) to the left and sinks (FIFO) to the
/// right, then places one max-`delta` core vertex left, breaking ties toward
/// the smallest id so the result is deterministic.
fn greedy_linear_arrangement<M>(
    matrix: &M,
    handles: &[M::Index],
    mut degrees: Degrees,
) -> Vec<usize>
where
    M: SparseValuedMatrix2D + ValuedSizedSparseBiMatrix2D + SquareMatrix + ?Sized,
    M::Index: AsPrimitive<usize>,
    M::Value: AsPrimitive<f64>,
{
    let number_of_nodes = degrees.removed.len();
    let mut left = Vec::with_capacity(number_of_nodes);
    let mut right = Vec::with_capacity(number_of_nodes);
    let mut sources = VecDeque::new();
    let mut sinks = VecDeque::new();
    let mut remaining = number_of_nodes;

    // Classify every vertex once; isolated vertices go straight to the left.
    for node in 0..number_of_nodes {
        match (degrees.in_count[node], degrees.out_count[node]) {
            (0, 0) => {
                degrees.removed[node] = true;
                left.push(node);
                remaining -= 1;
            }
            (0, _) => sources.push_back(node),
            (_, 0) => sinks.push_back(node),
            _ => {}
        }
    }

    while remaining > 0 {
        // Drain every source (in-degree zero) to the left. Removing one can
        // expose another, which the queue picks up.
        while let Some(source) = sources.pop_front() {
            if degrees.removed[source] {
                continue;
            }
            degrees.removed[source] = true;
            left.push(source);
            remaining -= 1;
            relax_out_edges(source, matrix, handles, &mut degrees, &mut sources);
        }

        // Drain every sink (out-degree zero) to the right.
        while let Some(sink) = sinks.pop_front() {
            if degrees.removed[sink] {
                continue;
            }
            degrees.removed[sink] = true;
            right.push(sink);
            remaining -= 1;
            relax_in_edges(sink, matrix, handles, &mut degrees, &mut sinks);
        }

        if remaining == 0 {
            break;
        }

        // Tangled core: place the max weighted-`delta` vertex left. Zipping the
        // three slices lets the bounds checks hoist out of this hot loop.
        let mut best = usize::MAX;
        let mut best_delta = f64::NEG_INFINITY;
        for (node, ((&removed, &out_weight), &in_weight)) in
            degrees.removed.iter().zip(&degrees.out_weight).zip(&degrees.in_weight).enumerate()
        {
            if removed {
                continue;
            }
            let delta = out_weight - in_weight;
            if best == usize::MAX || delta > best_delta {
                best = node;
                best_delta = delta;
            }
        }
        degrees.removed[best] = true;
        left.push(best);
        remaining -= 1;
        relax_out_edges(best, matrix, handles, &mut degrees, &mut sources);
        relax_in_edges(best, matrix, handles, &mut degrees, &mut sinks);
    }

    // Sinks were pushed front-to-back but belong at the far right, so reverse.
    let mut order = left;
    order.extend(right.into_iter().rev());
    order
}

/// Error enumeration for the Eades-Lin-Smyth feedback-arc-set heuristic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EadesLinSmythError {
    /// The edge weight is not finite.
    #[error("Found a non-finite edge weight on ({source_id}, {destination_id}).")]
    NonFiniteWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// The edge weight must be strictly positive.
    #[error("Found a non-positive edge weight on ({source_id}, {destination_id}).")]
    NonPositiveWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
}

/// Result of the Eades-Lin-Smyth greedy feedback-arc-set heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct EadesLinSmythResult {
    /// Linear arrangement: `order[i]` is the node placed at position `i`.
    order: Vec<usize>,
    /// Inverse of `order`: `positions[node]` is the position of `node`.
    positions: Vec<usize>,
    /// Backward (feedback) edges `(source, destination)`.
    feedback_edges: Vec<(usize, usize)>,
    /// Total weight of the backward edges.
    feedback_weight: f64,
    /// Total weight of all edges (self-loops excluded).
    total_weight: f64,
}

impl EadesLinSmythResult {
    /// The linear arrangement, where `order()[i]` is the node at position `i`.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The position of each node within the arrangement.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// The backward (feedback) edges as `(source, destination)` pairs.
    #[must_use]
    pub fn feedback_edges(&self) -> &[(usize, usize)] {
        &self.feedback_edges
    }

    /// The total weight of the backward edges.
    #[must_use]
    pub fn feedback_weight(&self) -> f64 {
        self.feedback_weight
    }

    /// The total weight of all edges (self-loops excluded).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// The fraction of edge weight that points backward, in `[0, 1]`.
    #[must_use]
    pub fn tangle_fraction(&self) -> f64 {
        if self.total_weight <= 0.0 { 0.0 } else { self.feedback_weight / self.total_weight }
    }

    /// Whether the arrangement has no backward edges.
    #[must_use]
    pub fn is_acyclic(&self) -> bool {
        self.feedback_edges.is_empty()
    }
}

/// The Eades-Lin-Smyth (GR) greedy heuristic for the feedback arc set problem.
///
/// Entry `(i, j)` of the matrix is the weight of edge `i -> j`. Weights must be
/// finite and strictly positive, and self-loops are ignored. The matrix is read
/// in both directions through the bidirectional valued-matrix trait: out-edges
/// from its rows, in-edges from its columns.
pub trait EadesLinSmyth: SparseValuedMatrix2D + ValuedSizedSparseBiMatrix2D + SquareMatrix {
    /// Computes a greedy linear arrangement and its backward (feedback) edges.
    ///
    /// # Complexity
    ///
    /// `O(V + E + c * V)` time, `O(V + E)` space, where `c` is the number of
    /// tangled-core vertices needing a max-`delta` scan. Source/sink peeling is
    /// linear, so this is `O(V + E)` on a DAG and `O(V^2 + E)` worst case.
    ///
    /// # Errors
    ///
    /// Returns an error when a (non-self-loop) weight is non-finite or
    /// non-positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{GenericBiMatrix2D, ValuedCSR2D},
    ///     prelude::*,
    /// };
    ///
    /// // A single 3-cycle 0 -> 1 -> 2 -> 0 needs exactly one edge cut to layer.
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///         .expected_number_of_edges(3)
    ///         .expected_shape((3, 3))
    ///         .edges(vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)].into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    /// let result = GenericBiMatrix2D::new(edges).eades_lin_smyth().unwrap();
    /// assert_eq!(result.feedback_edges(), &[(2, 0)]);
    /// assert!((result.tangle_fraction() - 1.0 / 3.0).abs() < 1e-12);
    /// ```
    fn eades_lin_smyth(&self) -> Result<EadesLinSmythResult, EadesLinSmythError>
    where
        Self::Index: AsPrimitive<usize>,
        Self::Value: AsPrimitive<f64>;
}

impl<M: SparseValuedMatrix2D + ValuedSizedSparseBiMatrix2D + SquareMatrix + ?Sized> EadesLinSmyth
    for M
{
    fn eades_lin_smyth(&self) -> Result<EadesLinSmythResult, EadesLinSmythError>
    where
        Self::Index: AsPrimitive<usize>,
        Self::Value: AsPrimitive<f64>,
    {
        let number_of_nodes = self.order().as_();

        // Map node usize -> Index once so the max-delta core pick (found as a
        // usize) can be turned back into an Index to address its row / column.
        let handles: Vec<Self::Index> = self.row_indices().collect();

        let mut out_count = vec![0usize; number_of_nodes];
        let mut in_count = vec![0usize; number_of_nodes];
        let mut out_weight = vec![0.0; number_of_nodes];
        let mut in_weight = vec![0.0; number_of_nodes];
        let mut total_weight = 0.0;

        for (source, &row) in handles.iter().enumerate() {
            for (column_id, value) in self.sparse_row(row).zip(self.sparse_row_values(row)) {
                let destination = column_id.as_();

                // Self-loops are dropped before validation: an ignored edge's
                // weight is irrelevant, and a loop would pin a node out of both
                // the source and sink sets.
                if source == destination {
                    continue;
                }

                let weight: f64 = value.as_();
                if !weight.is_finite() {
                    return Err(EadesLinSmythError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                if weight <= 0.0 {
                    return Err(EadesLinSmythError::NonPositiveWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }

                out_count[source] += 1;
                in_count[destination] += 1;
                out_weight[source] += weight;
                in_weight[destination] += weight;
                total_weight += weight;
            }
        }

        let degrees = Degrees {
            out_count,
            in_count,
            out_weight,
            in_weight,
            removed: vec![false; number_of_nodes],
        };

        let order = greedy_linear_arrangement(self, &handles, degrees);

        let mut positions = vec![0usize; number_of_nodes];
        for (position, &node) in order.iter().enumerate() {
            positions[node] = position;
        }

        let mut feedback_edges = Vec::new();
        let mut feedback_weight = 0.0;
        for (source, &row) in handles.iter().enumerate() {
            for (column_id, value) in self.sparse_row(row).zip(self.sparse_row_values(row)) {
                let destination = column_id.as_();
                if source == destination {
                    continue;
                }
                if positions[source] > positions[destination] {
                    feedback_edges.push((source, destination));
                    feedback_weight += value.as_();
                }
            }
        }

        Ok(EadesLinSmythResult { order, positions, feedback_edges, feedback_weight, total_weight })
    }
}
