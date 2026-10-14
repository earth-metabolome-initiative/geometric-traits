//! Kruskal's algorithm: sort edges by weight, accept each that joins two
//! distinct components.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, Zero};

use super::common::{CollectedGraph, MinimumSpanningTreeError, root_forest};
use crate::{
    impls::WeightedForest,
    traits::{Finite, SparseValuedMatrix2D, TotalOrd, algorithms::union_find::UnionFind},
};

/// Minimum spanning forest via Kruskal's algorithm (sort edges, union-find).
pub trait Kruskal: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: TotalOrd + Finite + Copy + Zero,
{
    /// Minimum spanning forest via Kruskal's algorithm.
    ///
    /// # Errors
    ///
    /// If the matrix is not square or a weight is non-finite.
    #[inline]
    fn minimum_spanning_tree_kruskal(
        &self,
    ) -> Result<WeightedForest<usize, Self::Value>, MinimumSpanningTreeError> {
        Ok(CollectedGraph::from_matrix(self)?.kruskal())
    }
}

impl<M: SparseValuedMatrix2D> Kruskal for M
where
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: TotalOrd + Finite + Copy + Zero,
{
}

impl<F: Copy + TotalOrd + Zero> CollectedGraph<F> {
    pub(super) fn kruskal(self) -> WeightedForest<usize, F> {
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

        root_forest(node_count, &tree)
    }
}
