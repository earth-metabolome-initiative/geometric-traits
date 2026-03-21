//! Labeled modular product of two valued graphs.
//!
//! Given graphs G₁ and G₂ whose edges carry labels (values), and a set of
//! vertex pairs P ⊆ V(G₁) × V(G₂), the **labeled modular product** has
//! vertex set P and an edge between pairs (u₁, u₂) and (v₁, v₂) iff
//! u₁ ≠ v₁, u₂ ≠ v₂, and the label of u₁v₁ in G₁ equals the label of
//! u₂v₂ in G₂ (using `Option` semantics: `None == None` for non-edges,
//! `Some(a) == Some(a)` for matching labels).
//!
//! This is the labeled variant of [`ModularProduct`](super::ModularProduct),
//! needed for chemistry (RASCAL) and any domain with typed edges.
//!
//! # Complexity
//! O(|P|² · row_scan) time, O(|P|²/64) space (bit-packed adjacency).

use super::super::SparseValuedMatrix2D;
use crate::{impls::BitSquareMatrix, traits::SparseSquareMatrix};

/// Trait for computing the labeled modular product of two valued graphs.
pub trait LabeledModularProduct: SparseSquareMatrix + SparseValuedMatrix2D {
    /// Computes the labeled modular product adjacency matrix.
    ///
    /// `other` is the second graph's adjacency matrix (with compatible value
    /// type). `vertex_pairs` lists the (u₁, u₂) pairs forming the vertex
    /// set of the product.
    ///
    /// Returns a [`BitSquareMatrix`] of order `vertex_pairs.len()`.
    #[must_use]
    fn labeled_modular_product<M>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix
    where
        M: SparseSquareMatrix + SparseValuedMatrix2D<Value = Self::Value>,
        Self::Value: PartialEq,
        M::Index: PartialEq;
}

impl<G: SparseSquareMatrix + SparseValuedMatrix2D> LabeledModularProduct for G {
    #[inline]
    fn labeled_modular_product<M>(
        &self,
        other: &M,
        vertex_pairs: &[(Self::Index, M::Index)],
    ) -> BitSquareMatrix
    where
        M: SparseSquareMatrix + SparseValuedMatrix2D<Value = Self::Value>,
        Self::Value: PartialEq,
        M::Index: PartialEq,
    {
        let p = vertex_pairs.len();
        let mut matrix = BitSquareMatrix::new(p);
        for (a, &(u1, u2)) in vertex_pairs.iter().enumerate() {
            for (b, &(v1, v2)) in vertex_pairs.iter().enumerate().skip(a + 1) {
                if u1 != v1 && u2 != v2 {
                    let label1 = self.sparse_value_at(u1, v1);
                    let label2 = other.sparse_value_at(u2, v2);
                    if label1 == label2 {
                        matrix.set_symmetric(a, b);
                    }
                }
            }
        }
        matrix
    }
}
