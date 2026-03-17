//! Submodule providing the `Blossom` trait for maximum cardinality matching
//! in general (non-bipartite) graphs using the Edmonds blossom algorithm.
use alloc::vec::Vec;

mod inner;

use inner::BlossomState;

use crate::traits::SparseSquareMatrix;

/// Maximum cardinality matching in general graphs via the Edmonds blossom
/// algorithm.
pub trait Blossom: SparseSquareMatrix {
    /// Returns a maximum cardinality matching as a list of edge pairs `(u, v)`
    /// with `u < v`.
    ///
    /// The input matrix is expected to be symmetric (representing an undirected
    /// graph). Self-loops are ignored. Non-symmetric input will not panic but
    /// the result is unspecified.
    ///
    /// # Complexity
    ///
    /// O(V² · E) time, O(V) space.
    #[inline]
    fn blossom(&self) -> Vec<(Self::Index, Self::Index)> {
        BlossomState::new(self).solve()
    }
}

impl<M: SparseSquareMatrix> Blossom for M {}
