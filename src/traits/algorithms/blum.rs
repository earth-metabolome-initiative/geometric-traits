//! Submodule providing the `Blum` trait for maximum cardinality matching
//! in general (non-bipartite) graphs using Blum's algorithm.
//!
//! The algorithm reduces finding augmenting paths in a general graph to a
//! reachability problem in a directed bipartite graph G_M, then uses a
//! modified depth-first search (MDFS) to find strongly simple s-t paths
//! that correspond to augmenting paths in the original graph.
//!
//! Reference: Norbert Blum, "A New Approach to Maximum Matching in General
//! Graphs" (ICALP 1990); revised 2015 (arXiv:1509.04927).
use alloc::vec::Vec;

mod inner;

use inner::BlumState;

use crate::traits::SparseSquareMatrix;

/// Maximum cardinality matching in general graphs via Blum's algorithm.
pub trait Blum: SparseSquareMatrix {
    /// Returns a maximum cardinality matching as a list of edge pairs `(u, v)`
    /// with `u < v`.
    ///
    /// The input matrix is expected to be symmetric (representing an undirected
    /// graph). Self-loops are ignored. Non-symmetric input will not panic but
    /// the result is unspecified.
    ///
    /// # Complexity
    ///
    /// O(√V · (V + E) · α(V + E, V)) time, O(V + E) space, where α is
    /// the inverse Ackermann function (at most 4 for all practical inputs).
    ///
    /// Blum's Theorem 6 states O(√V · (V + E)) using Gabow-Tarjan
    /// incremental tree set union.  We use standard union-find instead:
    /// benchmarks show Gabow-Tarjan is 6-10× slower in practice.  See
    /// <https://github.com/LucaCappelletti94/incremental-tree-set-union>.
    #[inline]
    fn blum(&self) -> Vec<(Self::Index, Self::Index)> {
        BlumState::new(self).solve()
    }
}

impl<M: SparseSquareMatrix> Blum for M {}
