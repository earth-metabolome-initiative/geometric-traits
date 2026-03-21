//! Modular product of two undirected graphs (unlabeled and labeled variants).
//!
//! Given graphs G₁ and G₂ and a set of vertex pairs P ⊆ V(G₁) × V(G₂), the
//! **modular product** has vertex set P and an edge between pairs (u₁, u₂) and
//! (v₁, v₂) iff u₁ ≠ v₁, u₂ ≠ v₂, and the adjacency or label relationship is
//! compatible.
//!
//! This module provides:
//!
//! - an unlabeled modular product based on adjacency equality
//! - a labeled modular product with a custom `comparator` over `Option<Value>`
//!
//! For labeled graphs, a custom `comparator` lambda controls edge compatibility
//! by receiving `Option<Value>` from each graph. This enables strict equality,
//! floating-point tolerance, or any user-defined comparison.
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
//! assert_eq!(mp.order(), 9);
//! ```

use num_traits::AsPrimitive;

use crate::{
    impls::BitSquareMatrix,
    traits::{SparseSquareMatrix, SparseValuedMatrix2D, ValuedMatrix},
};

/// Private helper: builds the modular product adjacency matrix from
/// pre-collected vertex pairs and an edge-compatibility predicate.
///
/// Two vertex pairs `(u1, u2)` and `(v1, v2)` are connected iff:
/// - `u1 ≠ v1` (injectivity in G₁)
/// - `u2 ≠ v2` (injectivity in G₂)
/// - `edge_compat(u1, v1, u2, v2)` returns true
#[inline]
fn modular_product_core<I1, I2, F>(vertex_pairs: &[(I1, I2)], edge_compat: F) -> BitSquareMatrix
where
    I1: PartialEq + Copy,
    I2: PartialEq + Copy,
    F: Fn(I1, I1, I2, I2) -> bool,
{
    let p = vertex_pairs.len();
    let mut matrix = BitSquareMatrix::new(p);
    for (a, &(u1, u2)) in vertex_pairs.iter().enumerate() {
        for (b, &(v1, v2)) in vertex_pairs.iter().enumerate().skip(a + 1) {
            if u1 != v1 && u2 != v2 && edge_compat(u1, v1, u2, v2) {
                matrix.set_symmetric(a, b);
            }
        }
    }
    matrix
}

/// Trait for computing the modular product of two graphs.
///
/// Provides unlabeled and labeled modular product methods over a
/// caller-supplied set of vertex pairs.
pub trait ModularProduct: SparseSquareMatrix {
    /// Computes the unlabeled modular product from pre-built vertex pairs.
    ///
    /// Edge condition: `self.has_entry(u1, v1) == other.has_entry(u2, v2)`.
    ///
    /// Returns a [`BitSquareMatrix`] of order `vertex_pairs.len()`.
    #[must_use]
    fn modular_product<M: SparseSquareMatrix>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix;

    /// Computes the labeled modular product from pre-built vertex pairs.
    ///
    /// `comparator(val1, val2)` controls edge compatibility, where `val1` and
    /// `val2` are the `Option<Value>` from `sparse_value_at` on each graph.
    ///
    /// Returns a [`BitSquareMatrix`] of order `vertex_pairs.len()`.
    #[must_use]
    fn labeled_modular_product<M, C>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
        comparator: C,
    ) -> BitSquareMatrix
    where
        Self: SparseValuedMatrix2D,
        M: SparseSquareMatrix + SparseValuedMatrix2D,
        C: Fn(Option<<Self as ValuedMatrix>::Value>, Option<<M as ValuedMatrix>::Value>) -> bool;
}

impl<G: SparseSquareMatrix> ModularProduct for G {
    #[inline]
    fn modular_product<M: SparseSquareMatrix>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix {
        debug_assert!(
            vertex_pairs
                .iter()
                .all(|&(i, j)| i.as_() < self.order().as_() && j.as_() < other.order().as_()),
            "vertex_pairs contains out-of-bounds indices"
        );
        modular_product_core(vertex_pairs, |u1, v1, u2, v2| {
            self.has_entry(u1, v1) == other.has_entry(u2, v2)
        })
    }

    #[inline]
    fn labeled_modular_product<M, C>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
        comparator: C,
    ) -> BitSquareMatrix
    where
        Self: SparseValuedMatrix2D,
        M: SparseSquareMatrix + SparseValuedMatrix2D,
        C: Fn(Option<<Self as ValuedMatrix>::Value>, Option<<M as ValuedMatrix>::Value>) -> bool,
    {
        debug_assert!(
            vertex_pairs
                .iter()
                .all(|&(i, j)| i.as_() < self.order().as_() && j.as_() < other.order().as_()),
            "vertex_pairs contains out-of-bounds indices"
        );
        modular_product_core(vertex_pairs, |u1, v1, u2, v2| {
            comparator(self.sparse_value_at(u1, v1), other.sparse_value_at(u2, v2))
        })
    }
}
