//! Boruvka's algorithm: in each round, every component selects its cheapest
//! outgoing edge and the chosen edges are merged.

use alloc::vec::Vec;
use core::cmp::Ordering;

use num_traits::{AsPrimitive, ToPrimitive};

use super::common::{CollectedGraph, MinimumSpanningForest, MinimumSpanningTreeError};
use crate::traits::{Finite, SparseValuedMatrix2D, algorithms::union_find::UnionFind};

/// Minimum spanning forest via Boruvka's algorithm (round-based component
/// merging).
pub trait Boruvka: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: ToPrimitive + Finite,
{
    /// Minimum spanning forest via Boruvka's algorithm.
    ///
    /// # Errors
    ///
    /// If the matrix is not square or a weight is non-finite or unrepresentable
    /// as `f64`.
    #[inline]
    fn minimum_spanning_tree_boruvka(
        &self,
    ) -> Result<MinimumSpanningForest, MinimumSpanningTreeError> {
        Ok(CollectedGraph::from_matrix(self)?.boruvka())
    }
}

impl<M: SparseValuedMatrix2D> Boruvka for M
where
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: ToPrimitive + Finite,
{
}

impl CollectedGraph {
    pub(super) fn boruvka(self) -> MinimumSpanningForest {
        let Self { node_count, edges } = self;
        let mut disjoint = UnionFind::new(node_count);
        let mut tree = Vec::new();

        loop {
            // Cheapest outgoing edge per component root. The edge-index
            // tie-break is a strict total order, so the chosen edges cannot
            // form a cycle.
            let mut cheapest: Vec<Option<(f64, usize)>> = alloc::vec![None; node_count];
            for (index, &(source, destination, weight)) in edges.iter().enumerate() {
                let source_root = disjoint.find(source);
                let destination_root = disjoint.find(destination);
                if source_root == destination_root {
                    continue;
                }
                for root in [source_root, destination_root] {
                    let is_better = match cheapest[root] {
                        None => true,
                        Some((best_weight, best_index)) => {
                            weight.total_cmp(&best_weight).then_with(|| index.cmp(&best_index))
                                == Ordering::Less
                        }
                    };
                    if is_better {
                        cheapest[root] = Some((weight, index));
                    }
                }
            }

            let mut merged_any = false;
            for slot in &cheapest {
                let Some((weight, index)) = *slot else {
                    continue;
                };
                let (source, destination, _) = edges[index];
                if disjoint.union(source, destination) {
                    tree.push((source, destination, weight));
                    merged_any = true;
                }
            }

            if !merged_any {
                break;
            }
        }

        MinimumSpanningForest::from_edges(node_count, tree)
    }
}
