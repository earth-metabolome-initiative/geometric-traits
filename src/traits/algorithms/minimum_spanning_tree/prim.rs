//! Prim's algorithm: grow one tree per component from a binary-heap frontier of
//! the lightest crossing edge.

use alloc::{collections::BinaryHeap, vec::Vec};
use core::cmp::Ordering;

use num_traits::{AsPrimitive, Zero};

use super::common::{CollectedGraph, MinimumSpanningTreeError, root_forest};
use crate::{
    impls::WeightedForest,
    traits::{Finite, SparseValuedMatrix2D, TotalOrd},
};

/// Minimum spanning forest via Prim's algorithm (binary-heap frontier growth).
pub trait Prim: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: TotalOrd + Finite + Copy + Zero,
{
    /// Minimum spanning forest via Prim's algorithm.
    ///
    /// # Errors
    ///
    /// If the matrix is not square or a weight is non-finite.
    #[inline]
    fn minimum_spanning_tree_prim(
        &self,
    ) -> Result<WeightedForest<usize, Self::Value>, MinimumSpanningTreeError> {
        Ok(CollectedGraph::from_matrix(self)?.prim())
    }
}

impl<M: SparseValuedMatrix2D> Prim for M
where
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: TotalOrd + Finite + Copy + Zero,
{
}

/// A binary-heap frontier candidate. The `Ord` impl reverses the weight so the
/// max-heap pops the lightest edge first.
#[derive(Debug)]
struct PrimEntry<F> {
    weight: F,
    from: usize,
    to: usize,
}

impl<F: TotalOrd> Ord for PrimEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .weight
            .total_cmp(&self.weight)
            .then_with(|| self.from.cmp(&other.from))
            .then_with(|| self.to.cmp(&other.to))
    }
}

impl<F: TotalOrd> PartialOrd for PrimEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: TotalOrd> PartialEq for PrimEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<F: TotalOrd> Eq for PrimEntry<F> {}

impl<F: Copy + TotalOrd + Zero> CollectedGraph<F> {
    pub(super) fn prim(self) -> WeightedForest<usize, F> {
        let Self { node_count, edges } = self;

        let mut adjacency: Vec<Vec<(usize, F)>> = alloc::vec![Vec::new(); node_count];
        for &(source, destination, weight) in &edges {
            adjacency[source].push((destination, weight));
            adjacency[destination].push((source, weight));
        }

        let mut visited = alloc::vec![false; node_count];
        let mut tree = Vec::new();
        let mut heap: BinaryHeap<PrimEntry<F>> = BinaryHeap::new();

        // Each unvisited seed starts a new tree, so the loop covers every
        // connected component.
        for start in 0..node_count {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            for &(neighbor, weight) in &adjacency[start] {
                if !visited[neighbor] {
                    heap.push(PrimEntry { weight, from: start, to: neighbor });
                }
            }

            while let Some(PrimEntry { weight, from, to }) = heap.pop() {
                if visited[to] {
                    continue;
                }
                visited[to] = true;
                // The rooting pass treats edges as undirected, so the endpoint
                // order does not matter here.
                tree.push((from, to, weight));
                for &(neighbor, neighbor_weight) in &adjacency[to] {
                    if !visited[neighbor] {
                        heap.push(PrimEntry { weight: neighbor_weight, from: to, to: neighbor });
                    }
                }
            }
        }

        root_forest(node_count, &tree)
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use super::PrimEntry;

    fn entry(weight: f64, from: usize, to: usize) -> PrimEntry<f64> {
        PrimEntry { weight, from, to }
    }

    #[test]
    fn lighter_edge_ranks_higher_for_the_max_heap() {
        let light = entry(1.0, 0, 1);
        let heavy = entry(2.0, 0, 1);
        // The weight order is reversed so a `BinaryHeap` pops the lightest
        // edge.
        assert_eq!(light.cmp(&heavy), Ordering::Greater);
        assert_eq!(heavy.cmp(&light), Ordering::Less);
        assert_eq!(light.partial_cmp(&heavy), Some(Ordering::Greater));
        assert!(light > heavy);
        assert_ne!(light, heavy);
    }

    #[test]
    fn equal_weight_breaks_ties_on_endpoints() {
        let a = entry(1.0, 0, 1);
        let b = entry(1.0, 0, 2);
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_ne!(a, b);
    }

    #[test]
    fn identical_entries_are_equal() {
        let a = entry(1.5, 2, 3);
        let b = entry(1.5, 2, 3);
        assert_eq!(a.cmp(&b), Ordering::Equal);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_eq!(a, b);
    }
}
