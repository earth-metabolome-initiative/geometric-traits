//! Submodule providing the `MicaliVazirani` trait for maximum cardinality
//! matching in general (non-bipartite) graphs using the Micali-Vazirani
//! algorithm.
use alloc::vec::Vec;

mod inner;

use inner::MVState;

use crate::traits::SparseSquareMatrix;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur while validating the input for the Micali-Vazirani
/// matching algorithm.
pub enum MicaliVaziraniError {
    /// The graph must be undirected, so every edge must have a matching reverse
    /// edge.
    #[error(
        "The matrix is not symmetric: edge ({source_id}, {destination_id}) has no matching reverse edge."
    )]
    NonSymmetricEdge {
        /// Source vertex of the missing reverse-edge check.
        source_id: usize,
        /// Destination vertex of the missing reverse-edge check.
        destination_id: usize,
    },
}

/// Matching edge pairs returned by the Micali-Vazirani solver.
pub type MatchingPairs<I> = Vec<(I, I)>;

/// Maximum cardinality matching in general graphs via the Micali-Vazirani
/// algorithm.
pub trait MicaliVazirani: SparseSquareMatrix {
    /// Returns a maximum cardinality matching as a list of edge pairs `(u, v)`
    /// with `u < v`.
    ///
    /// The input matrix must be symmetric, representing an undirected graph.
    /// Self-loops are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`MicaliVaziraniError::NonSymmetricEdge`] if the input does not
    /// represent an undirected graph.
    ///
    /// # Complexity
    ///
    /// The core solver follows the Micali-Vazirani phase structure. This
    /// implementation also validates symmetry before solving.
    #[inline]
    fn micali_vazirani(&self) -> Result<MatchingPairs<Self::Index>, MicaliVaziraniError> {
        Ok(MVState::try_new(self)?.solve())
    }
}

impl<M: SparseSquareMatrix + ?Sized> MicaliVazirani for M {}
