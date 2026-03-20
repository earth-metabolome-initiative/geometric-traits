//! Modular product of two undirected graphs.
//!
//! Given graphs G₁ and G₂ and a set of vertex pairs P ⊆ V(G₁) × V(G₂),
//! the **modular product** has vertex set P and an edge between pairs
//! (u₁, u₂) and (v₁, v₂) iff u₁ ≠ v₁, u₂ ≠ v₂, and:
//!
//! - u₁v₁ ∈ E(G₁) **and** u₂v₂ ∈ E(G₂), **or**
//! - u₁v₁ ∉ E(G₁) **and** u₂v₂ ∉ E(G₂).
//!
//! This is the foundation of Barrow & Burstall (1976) subgraph
//! isomorphism and the RASCAL algorithm for MCES (Raymond et al. 2002).
//!
//! # Complexity
//! O(|P|²) time, O(|P|²/64) space (bit-packed adjacency).
//!
//! # Example
//! ```
//! use geometric_traits::{impls::BitSquareMatrix, prelude::*};
//!
//! // K3 (triangle) vs P3 (path 0-1-2)
//! let k3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
//! let p3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
//!
//! let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
//! let mp = k3.modular_product(&p3, &pairs);
//!
//! // Result is a 9×9 symmetric BitSquareMatrix with no self-loops.
//! assert_eq!(mp.order(), 9);
//! for i in 0..9 {
//!     assert!(!mp.has_entry(i, i));
//!     for j in 0..9 {
//!         assert_eq!(mp.has_entry(i, j), mp.has_entry(j, i));
//!     }
//! }
//! ```

use crate::{impls::BitSquareMatrix, traits::SparseSquareMatrix};

/// Trait for computing the modular product of two graphs.
///
/// The modular product adjacency matrix is built from `self` (G₁) and
/// a second graph `other` (G₂), over a caller-supplied set of vertex pairs.
pub trait ModularProduct: SparseSquareMatrix {
    /// Computes the modular product adjacency matrix.
    ///
    /// `other` is the second graph's adjacency matrix.
    /// `vertex_pairs` lists the (u₁, u₂) pairs forming the vertex set of the
    /// product; the caller controls which pairs are included (e.g. filtering
    /// by atom type).
    ///
    /// Returns a [`BitSquareMatrix`] of order `vertex_pairs.len()`.
    #[must_use]
    fn modular_product<M: SparseSquareMatrix>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix;
}

impl<G: SparseSquareMatrix> ModularProduct for G {
    #[inline]
    fn modular_product<M: SparseSquareMatrix>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix {
        let p = vertex_pairs.len();
        let mut matrix = BitSquareMatrix::new(p);
        for (a, &(u1, u2)) in vertex_pairs.iter().enumerate() {
            for (b, &(v1, v2)) in vertex_pairs.iter().enumerate().skip(a + 1) {
                if u1 != v1 && u2 != v2 && self.has_entry(u1, v1) == other.has_entry(u2, v2) {
                    matrix.set_symmetric(a, b);
                }
            }
        }
        matrix
    }
}
