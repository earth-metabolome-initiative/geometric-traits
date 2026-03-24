//! Submodule providing the `Gabow1976` trait for maximum cardinality matching
//! in general (non-bipartite) graphs using Gabow's 1976 implementation of
//! Edmonds' algorithm.
use alloc::vec::Vec;

mod inner;

use inner::Gabow1976State;

use crate::traits::SparseSquareMatrix;

/// Maximum cardinality matching in general graphs via Gabow's 1976
/// implementation of Edmonds' algorithm.
pub trait Gabow1976: SparseSquareMatrix {
    /// Returns a maximum cardinality matching as a list of edge pairs `(u, v)`
    /// with `u < v`.
    ///
    /// The input matrix is expected to be symmetric (representing an undirected
    /// graph). Self-loops are ignored. Non-symmetric input will not panic but
    /// the result is unspecified.
    ///
    /// # Complexity
    ///
    /// O(V^3) time, O(V) auxiliary space.
    #[inline]
    fn gabow_1976(&self) -> Vec<(Self::Index, Self::Index)> {
        Gabow1976State::new(self).solve()
    }
}

impl<M: SparseSquareMatrix + ?Sized> Gabow1976 for M {}
