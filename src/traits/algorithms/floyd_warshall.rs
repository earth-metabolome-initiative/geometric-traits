//! Submodule providing the `FloydWarshall` trait and its blanket
//! implementation for sparse valued matrices.
//!
//! The algorithm computes all-pairs shortest-path distances for a weighted
//! adjacency matrix. The result is returned as a dense matrix whose entries are
//! `Option<Value>`:
//! - `Some(distance)` when a path exists;
//! - `None` when the destination is unreachable.
use num_traits::{AsPrimitive, Zero};

use crate::{
    impls::VecMatrix2D,
    traits::{Finite, Number, SparseValuedMatrix2D},
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur while executing the Floyd-Warshall algorithm.
pub enum FloydWarshallError {
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
    /// A tentative path distance overflowed or otherwise became non-finite.
    #[error(
        "Found a non-finite tentative distance for ({source_id}, {destination_id}) via {pivot_id}."
    )]
    NonFiniteDistance {
        /// Source node identifier.
        source_id: usize,
        /// Destination node identifier.
        destination_id: usize,
        /// Intermediate node identifier.
        pivot_id: usize,
    },
    /// The graph contains a negative cycle.
    #[error("Found a negative cycle reachable from node {node_id}.")]
    NegativeCycle {
        /// Node identifier whose diagonal distance became negative.
        node_id: usize,
    },
}

/// Trait providing Floyd-Warshall all-pairs shortest paths for sparse valued
/// matrices.
///
/// The matrix is interpreted as a weighted adjacency matrix. Missing entries
/// represent absent edges. The diagonal is initialized to zero, so positive
/// self-loops do not change the distance from a node to itself, while negative
/// self-loops correctly trigger a negative-cycle error.
///
/// # Complexity
///
/// O(n³) time and O(n²) space.
///
/// # Examples
///
/// ```
/// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
///
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(3)
///         .expected_shape((3, 3))
///         .edges(vec![(0, 1, 5.0), (0, 2, 10.0), (1, 2, 2.0)].into_iter())
///         .build()
///         .unwrap();
///
/// let distances = csr.floyd_warshall().unwrap();
/// assert_eq!(distances.value((0, 2)), Some(7.0));
/// assert_eq!(distances.value((2, 0)), None);
/// ```
pub trait FloydWarshall: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite,
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Computes all-pairs shortest-path distances.
    ///
    /// The returned matrix is row-major and has the same order as the input
    /// matrix. Entry `(i, j)` is:
    /// - `Some(distance)` when `j` is reachable from `i`;
    /// - `None` when no path exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, if an input weight is not
    /// finite, if a tentative path distance becomes non-finite, or if a
    /// negative cycle is detected.
    #[inline]
    fn floyd_warshall(&self) -> Result<VecMatrix2D<Option<Self::Value>>, FloydWarshallError> {
        let rows = self.number_of_rows().as_();
        let columns = self.number_of_columns().as_();

        if rows != columns {
            return Err(FloydWarshallError::NonSquareMatrix { rows, columns });
        }

        let order = rows;
        let mut distances = vec![None; order * order];
        for node_id in 0..order {
            distances[node_id * order + node_id] = Some(Self::Value::zero());
        }

        for row_id in self.row_indices() {
            let source_id = row_id.as_();
            for (column_id, weight) in self.sparse_row(row_id).zip(self.sparse_row_values(row_id)) {
                let destination_id = column_id.as_();
                if !weight.is_finite() {
                    return Err(FloydWarshallError::NonFiniteWeight { source_id, destination_id });
                }

                let distance = &mut distances[source_id * order + destination_id];
                match *distance {
                    Some(current) if current <= weight => {}
                    _ => *distance = Some(weight),
                }
            }
        }

        for pivot_id in 0..order {
            for source_id in 0..order {
                let Some(source_to_pivot) = distances[source_id * order + pivot_id] else {
                    continue;
                };

                for destination_id in 0..order {
                    let Some(pivot_to_destination) = distances[pivot_id * order + destination_id]
                    else {
                        continue;
                    };

                    let through_pivot = source_to_pivot + pivot_to_destination;
                    if !through_pivot.is_finite() {
                        return Err(FloydWarshallError::NonFiniteDistance {
                            source_id,
                            destination_id,
                            pivot_id,
                        });
                    }

                    let distance = &mut distances[source_id * order + destination_id];
                    match *distance {
                        Some(current) if current <= through_pivot => {}
                        _ => *distance = Some(through_pivot),
                    }
                }
            }

            if let Some(diagonal) = distances[pivot_id * order + pivot_id] {
                if diagonal < Self::Value::zero() {
                    return Err(FloydWarshallError::NegativeCycle { node_id: pivot_id });
                }
            }
        }

        Ok(VecMatrix2D::new(order, order, distances))
    }
}

impl<M: SparseValuedMatrix2D> FloydWarshall for M
where
    M::Value: Number + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}
