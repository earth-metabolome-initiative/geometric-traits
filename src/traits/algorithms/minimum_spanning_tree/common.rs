//! Shared error, graph extraction, and forest construction for the
//! minimum-spanning-tree algorithms.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, Zero};

use crate::{
    impls::WeightedForest,
    traits::{Finite, SparseValuedMatrix2D, TotalOrd},
};

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
}

/// The undirected graph extracted from a matrix. Crossing the matrix generics
/// once here keeps the three algorithms non-generic over the matrix type. The
/// weight type `F` is the matrix value type, carried through without
/// conversion.
pub(super) struct CollectedGraph<F> {
    pub(super) node_count: usize,
    /// Canonical edges (`source < destination`), one per pair, sorted by
    /// endpoint.
    pub(super) edges: Vec<(usize, usize, F)>,
}

impl<F: Copy + TotalOrd + Finite> CollectedGraph<F> {
    /// Reads the matrix as an undirected weighted graph, validating it and
    /// folding each unordered pair to one canonical edge.
    pub(super) fn from_matrix<M>(matrix: &M) -> Result<Self, MinimumSpanningTreeError>
    where
        M: SparseValuedMatrix2D<Value = F>,
        M::RowIndex: AsPrimitive<usize>,
        M::ColumnIndex: AsPrimitive<usize>,
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

/// Roots the selected tree edges into a parent-array forest.
///
/// Each connected component is rooted at its lowest-index node, and edges are
/// oriented away from that root. Because a tree has a unique parent structure
/// once rooted, the traversal order does not affect the result, so the three
/// algorithms produce the same forest on any graph with a unique minimum
/// spanning tree.
pub(super) fn root_forest<F: Copy + Zero>(
    node_count: usize,
    tree_edges: &[(usize, usize, F)],
) -> WeightedForest<usize, F> {
    let mut adjacency: Vec<Vec<(usize, F)>> = alloc::vec![Vec::new(); node_count];
    for &(source, destination, weight) in tree_edges {
        adjacency[source].push((destination, weight));
        adjacency[destination].push((source, weight));
    }

    // Every node starts as its own root with a zero weight to itself.
    let mut parent: Vec<(usize, F)> = (0..node_count).map(|node| (node, F::zero())).collect();
    let mut visited = alloc::vec![false; node_count];
    let mut stack: Vec<usize> = Vec::new();

    for root in 0..node_count {
        if visited[root] {
            continue;
        }
        // The lowest unvisited index is its component's root and keeps its
        // self-parent.
        visited[root] = true;
        stack.push(root);
        while let Some(node) = stack.pop() {
            for &(neighbor, weight) in &adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = (node, weight);
                    stack.push(neighbor);
                }
            }
        }
    }

    WeightedForest::from_parent_array(parent)
}
