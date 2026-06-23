//! Shared error, result, and graph-extraction types for the
//! minimum-spanning-tree algorithms.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use crate::traits::{Finite, SparseValuedMatrix2D};

/// Error returned by the minimum spanning tree traits.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MinimumSpanningTreeError {
    /// The adjacency matrix is not square.
    #[error("The adjacency matrix is not square: {rows} rows by {columns} columns.")]
    NonSquareMatrix {
        /// Row count.
        rows: usize,
        /// Column count.
        columns: usize,
    },
    /// An edge weight is NaN or infinite.
    #[error("Edge ({source_id}, {destination_id}) has a non-finite weight.")]
    NonFiniteWeight {
        /// Source endpoint.
        source_id: usize,
        /// Destination endpoint.
        destination_id: usize,
    },
    /// An edge weight cannot be represented as an `f64`.
    #[error("Edge ({source_id}, {destination_id}) has a weight not representable as f64.")]
    UnrepresentableWeight {
        /// Source endpoint.
        source_id: usize,
        /// Destination endpoint.
        destination_id: usize,
    },
}

/// A minimum spanning forest: the selected tree edges (canonical, `usize`
/// indices) and summary statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningForest {
    edges: Vec<(usize, usize, f64)>,
    total_weight: f64,
    number_of_nodes: usize,
    number_of_components: usize,
}

impl MinimumSpanningForest {
    /// Builds the result, deriving the weight and the component count
    /// (`node_count - edges.len()` for any forest).
    pub(super) fn from_edges(node_count: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        let total_weight = edges.iter().map(|&(_, _, weight)| weight).sum();
        let number_of_components = node_count - edges.len();
        Self { edges, total_weight, number_of_nodes: node_count, number_of_components }
    }

    /// The tree edges as `(source, destination, weight)` triples, `source <
    /// destination`.
    #[inline]
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
    }

    /// The total weight (unique even when the edge set is not).
    #[inline]
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// The number of nodes in the input graph.
    #[inline]
    #[must_use]
    pub fn number_of_nodes(&self) -> usize {
        self.number_of_nodes
    }

    /// The number of trees, equal to the graph's connected components.
    #[inline]
    #[must_use]
    pub fn number_of_components(&self) -> usize {
        self.number_of_components
    }

    /// The number of tree edges.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the forest has no edges.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Whether the graph is connected (the forest is a single spanning tree).
    #[inline]
    #[must_use]
    pub fn is_spanning_tree(&self) -> bool {
        self.number_of_components <= 1
    }
}

/// The undirected graph extracted from a matrix. Crossing the matrix generics
/// once here keeps the three algorithms non-generic (compiled once, not per
/// matrix type).
pub(super) struct CollectedGraph {
    pub(super) node_count: usize,
    /// Canonical edges (`source < destination`), one per pair, sorted by
    /// endpoint.
    pub(super) edges: Vec<(usize, usize, f64)>,
}

impl CollectedGraph {
    /// Reads the matrix as an undirected weighted graph, validating it and
    /// folding each unordered pair to one canonical edge.
    pub(super) fn from_matrix<M>(matrix: &M) -> Result<Self, MinimumSpanningTreeError>
    where
        M: SparseValuedMatrix2D,
        M::RowIndex: AsPrimitive<usize>,
        M::ColumnIndex: AsPrimitive<usize>,
        M::Value: ToPrimitive + Finite,
    {
        let node_count = matrix.number_of_rows().as_();
        let columns = matrix.number_of_columns().as_();
        if node_count != columns {
            return Err(MinimumSpanningTreeError::NonSquareMatrix { rows: node_count, columns });
        }

        let mut edges = Vec::new();
        for row_id in matrix.row_indices() {
            let source = row_id.as_();
            for (column_id, weight) in
                matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id))
            {
                let destination = column_id.as_();
                if source == destination {
                    continue;
                }
                if !weight.is_finite() {
                    return Err(MinimumSpanningTreeError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                let weight =
                    weight.to_f64().ok_or(MinimumSpanningTreeError::UnrepresentableWeight {
                        source_id: source,
                        destination_id: destination,
                    })?;
                let (low, high) = if source < destination {
                    (source, destination)
                } else {
                    (destination, source)
                };
                edges.push((low, high, weight));
            }
        }

        // Sorting by (endpoints, weight) makes equal pairs adjacent and
        // weight-ascending, so `dedup_by` keeps the minimum-weight copy.
        edges.sort_by(|left, right| {
            (left.0, left.1).cmp(&(right.0, right.1)).then_with(|| left.2.total_cmp(&right.2))
        });
        edges.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);

        Ok(Self { node_count, edges })
    }
}
