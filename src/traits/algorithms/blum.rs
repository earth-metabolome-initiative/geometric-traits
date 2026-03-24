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
    /// O(V · (V + E)) worst-case time, O(V + E) space.
    ///
    /// Paper claim: Blum's phased algorithm (Theorem 6 in the 1990 paper and
    /// the 2015 rewrite) claims O(√V · (V + E)) time by finding batches of
    /// augmenting paths across O(√V) phases and using Gabow-Tarjan
    /// incremental-tree-set union inside MBFS. Replacing Gabow-Tarjan with
    /// standard union-find would only add an inverse-Ackermann factor, so the
    /// phased path alone would be O(√V · (V + E) · α(V + E, V)).
    ///
    /// Implementation note: this crate keeps the phased MBFS+MDFS fast path,
    /// but it also adds stronger correctness safeguards than the published
    /// algorithm. In particular, it validates reconstructed augmenting paths
    /// and falls back to per-free-vertex single-path MDFS when MBFS or layered
    /// MDFS misses an augmenting path. Those fallbacks can require up to O(V)
    /// single-path searches, which brings the implementation's documented
    /// worst-case bound back to O(V · (V + E)).
    #[inline]
    fn blum(&self) -> Vec<(Self::Index, Self::Index)> {
        BlumState::new(self).solve()
    }
}

impl<M: SparseSquareMatrix + ?Sized> Blum for M {}
