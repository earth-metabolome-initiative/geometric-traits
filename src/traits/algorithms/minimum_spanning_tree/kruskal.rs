//! Kruskal's algorithm: sort edges by weight, accept each that joins two
//! distinct components.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::common::{CollectedGraph, MinimumSpanningForest, MinimumSpanningTreeError};
use crate::traits::{Finite, SparseValuedMatrix2D, algorithms::union_find::UnionFind};

/// Minimum spanning forest via Kruskal's algorithm (sort edges, union-find).
pub trait Kruskal: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: ToPrimitive + Finite,
{
    /// Minimum spanning forest via Kruskal's algorithm.
    ///
    /// # Errors
    ///
    /// If the matrix is not square or a weight is non-finite or unrepresentable
    /// as `f64`.
    #[inline]
    fn minimum_spanning_tree_kruskal(
        &self,
    ) -> Result<MinimumSpanningForest, MinimumSpanningTreeError> {
        Ok(CollectedGraph::from_matrix(self)?.kruskal())
    }
}

impl<M: SparseValuedMatrix2D> Kruskal for M
where
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: ToPrimitive + Finite,
{
}

impl CollectedGraph {
    pub(super) fn kruskal(self) -> MinimumSpanningForest {
        let Self { node_count, mut edges } = self;
        // Ascending weight, with a deterministic endpoint tie-break.
        edges.sort_by(|left, right| {
            left.2.total_cmp(&right.2).then_with(|| (left.0, left.1).cmp(&(right.0, right.1)))
        });

        let mut disjoint = UnionFind::new(node_count);
        let mut tree = Vec::new();
        for (source, destination, weight) in edges {
            if disjoint.union(source, destination) {
                tree.push((source, destination, weight));
            }
        }

        MinimumSpanningForest::from_edges(node_count, tree)
    }
}
