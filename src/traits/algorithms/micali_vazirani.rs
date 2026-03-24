//! Submodule providing the `MicaliVazirani` trait for maximum cardinality
//! matching in general (non-bipartite) graphs using the Micali-Vazirani
//! algorithm.
use alloc::vec::Vec;

mod inner;

use inner::MVState;

use crate::traits::SparseSquareMatrix;

/// Maximum cardinality matching in general graphs via the Micali-Vazirani
/// algorithm.
pub trait MicaliVazirani: SparseSquareMatrix {
    /// Returns a maximum cardinality matching as a list of edge pairs `(u, v)`
    /// with `u < v`.
    ///
    /// The input matrix is expected to be symmetric (representing an undirected
    /// graph). Self-loops are ignored. Non-symmetric input will not panic but
    /// the result is unspecified.
    ///
    /// # Complexity
    ///
    /// O(sqrt(V) * E) time, O(V + E) space.
    #[inline]
    fn micali_vazirani(&self) -> Vec<(Self::Index, Self::Index)> {
        MVState::new(self).solve()
    }
}

impl<M: SparseSquareMatrix + ?Sized> MicaliVazirani for M {}
