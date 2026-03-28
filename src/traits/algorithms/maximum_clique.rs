//! Exact maximum clique search on [`BitSquareMatrix`].
//!
//! The public API is split into two traits:
//! - [`MaximumClique`] for the generic bit-parallel solver.
//! - [`PartitionedMaximumClique`] for the MCES-oriented partition-driven
//!   solver.
//!
//! This keeps the generic clique implementation clean while letting the
//! partitioned MCES path use a dedicated search state modeled after RDKit's
//! `PartitionSet`.

mod generic;
mod partitioned;

use alloc::vec::Vec;

pub use partitioned::{
    OwnedPartitionLabels, PartitionInfo, PartitionSearchProfile, PartitionSearchStats,
    PartitionSearchTrace, PartitionSide, all_best_search, choose_partition_side,
    greedy_lower_bound, partial_search, partial_search_with_root_pruning, profile_search,
    profile_search_with_bounds, trace_partial_search_to_target,
};

use crate::{impls::BitSquareMatrix, traits::SquareMatrix};

/// Trait for finding maximum cliques in an undirected graph.
pub trait MaximumClique {
    /// Returns one maximum clique (a clique of the largest possible size).
    #[must_use]
    fn maximum_clique(&self) -> Vec<usize>;

    /// Returns all maximum cliques (all cliques whose size equals ω(G)).
    #[must_use]
    fn all_maximum_cliques(&self) -> Vec<Vec<usize>>;

    /// Returns one accepted maximum clique.
    ///
    /// The callback receives a sorted clique. Returning `true` accepts the
    /// clique and allows it to update the current best size. Returning `false`
    /// rejects it and forces the search to continue.
    #[must_use]
    fn maximum_clique_where<F>(&self, accept_clique: F) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> bool;

    /// Returns all accepted maximum cliques.
    ///
    /// The callback receives a sorted clique. Only accepted cliques
    /// participate in the maximum-size result set.
    #[must_use]
    fn all_maximum_cliques_where<F>(&self, accept_clique: F) -> Vec<Vec<usize>>
    where
        F: FnMut(&[usize]) -> bool;
}

/// Trait for partition-aware maximum clique search.
pub trait PartitionedMaximumClique {
    /// Returns one maximum clique subject to a partition constraint with a
    /// label-aware upper bound.
    ///
    /// `partition.pairs.len()` must equal `self.order()`.
    #[must_use]
    fn maximum_clique_with_partition(&self, partition: &PartitionInfo<'_>) -> Vec<usize>;

    /// Returns all maximum cliques subject to a partition constraint with a
    /// label-aware upper bound.
    ///
    /// `partition.pairs.len()` must equal `self.order()`.
    #[must_use]
    fn all_maximum_cliques_with_partition(&self, partition: &PartitionInfo<'_>) -> Vec<Vec<usize>>;

    /// Returns one accepted maximum clique subject to a partition constraint
    /// with a label-aware upper bound.
    ///
    /// The callback receives a sorted clique in the solver graph's vertex
    /// index space.
    #[must_use]
    fn maximum_clique_with_partition_where<F>(
        &self,
        partition: &PartitionInfo<'_>,
        accept_clique: F,
    ) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> bool;

    /// Returns all accepted maximum cliques subject to a partition constraint
    /// with a label-aware upper bound.
    ///
    /// The callback receives a sorted clique in the solver graph's vertex
    /// index space.
    #[must_use]
    fn all_maximum_cliques_with_partition_where<F>(
        &self,
        partition: &PartitionInfo<'_>,
        accept_clique: F,
    ) -> Vec<Vec<usize>>
    where
        F: FnMut(&[usize]) -> bool;

    /// Convenience: partition constraint with all bonds labeled the same
    /// (unlabeled partition, double-sided injectivity only).
    #[must_use]
    fn maximum_clique_with_pairs(&self, pairs: &[(usize, usize)]) -> Vec<usize>
    where
        Self: Sized,
    {
        let labels = OwnedPartitionLabels::unlabeled(pairs);
        self.maximum_clique_with_partition(&labels.as_info(pairs))
    }

    /// Convenience: partition constraint with all bonds labeled the same plus
    /// an acceptance predicate.
    #[must_use]
    fn maximum_clique_with_pairs_where<F>(
        &self,
        pairs: &[(usize, usize)],
        accept_clique: F,
    ) -> Vec<usize>
    where
        Self: Sized,
        F: FnMut(&[usize]) -> bool,
    {
        let labels = OwnedPartitionLabels::unlabeled(pairs);
        self.maximum_clique_with_partition_where(&labels.as_info(pairs), accept_clique)
    }

    /// Convenience: enumerate all maximum cliques with an unlabeled partition.
    #[must_use]
    fn all_maximum_cliques_with_pairs(&self, pairs: &[(usize, usize)]) -> Vec<Vec<usize>>
    where
        Self: Sized,
    {
        let labels = OwnedPartitionLabels::unlabeled(pairs);
        self.all_maximum_cliques_with_partition(&labels.as_info(pairs))
    }

    /// Convenience: enumerate all accepted maximum cliques with an unlabeled
    /// partition.
    #[must_use]
    fn all_maximum_cliques_with_pairs_where<F>(
        &self,
        pairs: &[(usize, usize)],
        accept_clique: F,
    ) -> Vec<Vec<usize>>
    where
        Self: Sized,
        F: FnMut(&[usize]) -> bool,
    {
        let labels = OwnedPartitionLabels::unlabeled(pairs);
        self.all_maximum_cliques_with_partition_where(&labels.as_info(pairs), accept_clique)
    }
}

impl MaximumClique for BitSquareMatrix {
    fn maximum_clique(&self) -> Vec<usize> {
        self.maximum_clique_where(|_| true)
    }

    fn all_maximum_cliques(&self) -> Vec<Vec<usize>> {
        self.all_maximum_cliques_where(|_| true)
    }

    fn maximum_clique_where<F>(&self, accept_clique: F) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> bool,
    {
        generic::search(self, false, accept_clique).into_iter().next().unwrap_or_default()
    }

    fn all_maximum_cliques_where<F>(&self, accept_clique: F) -> Vec<Vec<usize>>
    where
        F: FnMut(&[usize]) -> bool,
    {
        generic::search(self, true, accept_clique)
    }
}

impl PartitionedMaximumClique for BitSquareMatrix {
    fn maximum_clique_with_partition(&self, partition: &PartitionInfo<'_>) -> Vec<usize> {
        self.maximum_clique_with_partition_where(partition, |_| true)
    }

    fn all_maximum_cliques_with_partition(&self, partition: &PartitionInfo<'_>) -> Vec<Vec<usize>> {
        self.all_maximum_cliques_with_partition_where(partition, |_| true)
    }

    fn maximum_clique_with_partition_where<F>(
        &self,
        partition: &PartitionInfo<'_>,
        accept_clique: F,
    ) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> bool,
    {
        assert_eq!(partition.pairs.len(), self.order(), "partition length must equal graph order");
        partitioned::search(self, partition, false, 0, accept_clique)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    fn all_maximum_cliques_with_partition_where<F>(
        &self,
        partition: &PartitionInfo<'_>,
        accept_clique: F,
    ) -> Vec<Vec<usize>>
    where
        F: FnMut(&[usize]) -> bool,
    {
        assert_eq!(partition.pairs.len(), self.order(), "partition length must equal graph order");
        partitioned::search(self, partition, true, 0, accept_clique)
    }
}
