//! Submodule providing the `PairwiseDijkstra` trait and its blanket
//! implementation for sparse valued matrices with a shared node index type.
//!
//! The algorithm runs one Dijkstra shortest-path search from each source node
//! and returns the resulting all-pairs shortest-path distances for the
//! non-negative weighted case.
use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use num_traits::{AsPrimitive, Zero};

use crate::{
    impls::VecMatrix2D,
    traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D, TotalOrd},
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur while executing the pairwise Dijkstra algorithm.
pub enum PairwiseDijkstraError {
    /// The input matrix is not square.
    #[error("The matrix must be square, but has {rows} rows and {columns} columns.")]
    NonSquareMatrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// An input edge weight is not finite.
    #[error("Found a non-finite weight on ({source_id}, {destination_id}).")]
    NonFiniteWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// An input edge weight is negative.
    #[error("Found a negative weight on ({source_id}, {destination_id}).")]
    NegativeWeight {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
    },
    /// A tentative path distance overflowed or otherwise became non-finite.
    #[error(
        "Found a non-finite tentative distance from {source_id} to {destination_id} via {via_id}."
    )]
    NonFiniteDistance {
        /// Source node identifier of the shortest-path search.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
        /// Intermediate node identifier whose outgoing edge caused the issue.
        via_id: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct QueueEntry<V, I> {
    distance: V,
    node_id: I,
}

impl<V, I> PartialEq for QueueEntry<V, I>
where
    V: TotalOrd,
    I: PositiveInteger,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance).is_eq() && self.node_id == other.node_id
    }
}

impl<V, I> Eq for QueueEntry<V, I>
where
    V: TotalOrd,
    I: PositiveInteger,
{
}

impl<V, I> PartialOrd for QueueEntry<V, I>
where
    V: TotalOrd,
    I: PositiveInteger,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<V, I> Ord for QueueEntry<V, I>
where
    V: TotalOrd,
    I: PositiveInteger,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.total_cmp(&self.distance).then_with(|| other.node_id.cmp(&self.node_id))
    }
}

/// Trait providing all-pairs shortest-path distances for non-negative weighted
/// graphs via repeated Dijkstra shortest-path search.
///
/// Missing entries in the sparse matrix are interpreted as absent edges. The
/// result is returned as a dense matrix whose entries are `Option<Value>`:
/// - `Some(distance)` when a path exists;
/// - `None` when the destination is unreachable.
///
/// The trait is available for sparse valued matrices whose row and column
/// index types coincide, since repeated Dijkstra must be able to revisit a
/// destination node as a later source node.
///
/// # Complexity
///
/// O(V * (V + E) * log V) time and O(V²) space.
///
/// # Examples
///
/// ```
/// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
///
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(4)
///         .expected_shape((4, 4))
///         .edges(vec![(0, 1, 2.0), (0, 3, 10.0), (1, 2, 3.0), (2, 3, 4.0)].into_iter())
///         .build()
///         .unwrap();
///
/// let distances = csr.pairwise_dijkstra().unwrap();
/// assert_eq!(distances.value((0, 3)), Some(9.0));
/// assert_eq!(distances.value((3, 0)), None);
/// ```
pub trait PairwiseDijkstra: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite + TotalOrd,
    Self::RowIndex: PositiveInteger,
    Self::ColumnIndex: AsPrimitive<usize> + Into<Self::RowIndex>,
{
    /// Computes all-pairs shortest-path distances in the non-negative weighted
    /// case.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, if an input weight is not
    /// finite, if an input weight is negative, or if a tentative path distance
    /// becomes non-finite.
    #[inline]
    fn pairwise_dijkstra(&self) -> Result<VecMatrix2D<Option<Self::Value>>, PairwiseDijkstraError> {
        let rows = self.number_of_rows().as_();
        let columns = self.number_of_columns().as_();

        if rows != columns {
            return Err(PairwiseDijkstraError::NonSquareMatrix { rows, columns });
        }

        let order = rows;
        let mut all_distances = vec![None; order * order];
        if order == 0 {
            return Ok(VecMatrix2D::new(0, 0, all_distances));
        }

        let zero = Self::Value::zero();
        let mut heap = BinaryHeap::new();

        for source_id in self.row_indices() {
            let source = source_id.as_();
            let source_offset = source * order;
            let distances = &mut all_distances[source_offset..source_offset + order];
            distances.fill(None);
            distances[source] = Some(zero);

            heap.clear();
            heap.push(QueueEntry { distance: zero, node_id: source_id });

            while let Some(entry) = heap.pop() {
                let node_id = entry.node_id;
                let node = node_id.as_();
                let best_distance =
                    distances[node].expect("Dijkstra heap only contains already-reached nodes");
                if entry.distance.total_cmp(&best_distance).is_gt() {
                    continue;
                }

                for (destination_id, weight) in
                    self.sparse_row(node_id).zip(self.sparse_row_values(node_id))
                {
                    let destination = destination_id.as_();
                    if !weight.is_finite() {
                        return Err(PairwiseDijkstraError::NonFiniteWeight {
                            source_id: node,
                            destination_id: destination,
                        });
                    }
                    if weight < zero {
                        return Err(PairwiseDijkstraError::NegativeWeight {
                            source_id: node,
                            destination_id: destination,
                        });
                    }

                    let candidate = entry.distance + weight;
                    if !candidate.is_finite() {
                        return Err(PairwiseDijkstraError::NonFiniteDistance {
                            source_id: source,
                            destination_id: destination,
                            via_id: node,
                        });
                    }

                    let should_update = match distances[destination] {
                        Some(current) => candidate.total_cmp(&current).is_lt(),
                        None => true,
                    };
                    if should_update {
                        distances[destination] = Some(candidate);
                        heap.push(QueueEntry {
                            distance: candidate,
                            node_id: destination_id.into(),
                        });
                    }
                }
            }
        }

        Ok(VecMatrix2D::new(order, order, all_distances))
    }
}

impl<M> PairwiseDijkstra for M
where
    M: SparseValuedMatrix2D + Sized,
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: AsPrimitive<usize> + Into<M::RowIndex>,
{
}
