//! Modular product of two undirected graphs (unlabeled and labeled variants).
//!
//! Given graphs G₁ and G₂, the **modular product** has vertex set
//! P ⊆ V(G₁) × V(G₂) and an edge between pairs (u₁, u₂) and (v₁, v₂) iff
//! u₁ ≠ v₁, u₂ ≠ v₂, and the adjacency/label relationship is compatible.
//!
//! Two API styles are provided:
//!
//! - **Lambda-based** (`_filtered` methods): the method iterates V₁ × V₂
//!   internally, calling a `pair_filter` lambda to decide which pairs to
//!   include. Returns a [`ModularProductResult`] containing both the product
//!   matrix and the collected vertex pairs.
//!
//! - **Reference-based**: the caller provides pre-built `vertex_pairs`. Returns
//!   only a [`BitSquareMatrix`]. This is the original API.
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
//! // Reference-based: caller provides pairs.
//! let pairs: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
//! let mp = k3.modular_product(&p3, &pairs);
//! assert_eq!(mp.order(), 9);
//!
//! // Lambda-based: method iterates internally.
//! let result = k3.modular_product_filtered(&p3, |_, _| true);
//! assert_eq!(result.matrix().order(), 9);
//! assert_eq!(result.vertex_pairs().len(), 9);
//! ```

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::{
    impls::BitSquareMatrix,
    traits::{PositiveInteger, SparseSquareMatrix, SparseValuedMatrix2D, ValuedMatrix},
};

/// Result of a lambda-based modular product computation.
///
/// Contains both the product adjacency matrix and the vertex pairs that
/// form the vertex set of the product (i.e., the pairs that passed the
/// `pair_filter`).
pub struct ModularProductResult<I1, I2> {
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(I1, I2)>,
}

impl<I1: Copy, I2: Copy> ModularProductResult<I1, I2> {
    /// Returns a reference to the product adjacency matrix.
    #[inline]
    #[must_use]
    pub fn matrix(&self) -> &BitSquareMatrix {
        &self.matrix
    }

    /// Returns the vertex pairs forming the product's vertex set.
    ///
    /// Index `k` in the [`BitSquareMatrix`] corresponds to `vertex_pairs()[k]`.
    #[inline]
    #[must_use]
    pub fn vertex_pairs(&self) -> &[(I1, I2)] {
        &self.vertex_pairs
    }

    /// Decomposes into the inner matrix and vertex pairs.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (BitSquareMatrix, Vec<(I1, I2)>) {
        (self.matrix, self.vertex_pairs)
    }
}

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

/// Collects vertex pairs by iterating V(G₁) × V(G₂) and filtering.
#[inline]
fn collect_pairs<I1, I2, PF>(order1: I1, order2: I2, mut pair_filter: PF) -> Vec<(I1, I2)>
where
    I1: PositiveInteger,
    I2: PositiveInteger,
    PF: FnMut(I1, I2) -> bool,
{
    let n1: usize = order1.as_();
    let n2: usize = order2.as_();
    let mut pairs = Vec::with_capacity(n1 * n2);
    let mut i = I1::ZERO;
    while i < order1 {
        let mut j = I2::ZERO;
        while j < order2 {
            if pair_filter(i, j) {
                pairs.push((i, j));
            }
            j += I2::ONE;
        }
        i += I1::ONE;
    }
    pairs
}

/// Trait for computing the modular product of two graphs.
///
/// Provides both lambda-based methods (iterate internally with a filter) and
/// reference-based methods (caller provides pre-built vertex pairs).
///
/// For labeled graphs, a custom `comparator` controls edge compatibility by
/// receiving `Option<Value>` pairs from both graphs.
pub trait ModularProduct: SparseSquareMatrix {
    /// Computes the unlabeled modular product, iterating V₁ × V₂ internally.
    ///
    /// `pair_filter(i, j)` is called for every `(i, j) ∈ V(G₁) × V(G₂)` to
    /// decide whether the pair enters the product. Use `|_, _| true` to include
    /// all pairs.
    ///
    /// Edge condition: `self.has_entry(u1, v1) == other.has_entry(u2, v2)`.
    #[must_use]
    fn modular_product_filtered<M, PF>(
        &self,
        other: &M,
        pair_filter: PF,
    ) -> ModularProductResult<Self::Index, M::Index>
    where
        M: SparseSquareMatrix,
        PF: FnMut(Self::Index, M::Index) -> bool;

    /// Computes the labeled modular product, iterating V₁ × V₂ internally.
    ///
    /// `pair_filter(i, j)` controls pair inclusion.
    /// `comparator(val1, val2)` controls edge compatibility, where `val1` and
    /// `val2` are the `Option<Value>` from `sparse_value_at` on each graph.
    ///
    /// Use `|a, b| a == b` for strict label equality, or a custom closure for
    /// tolerance-based comparison.
    #[must_use]
    fn labeled_modular_product_filtered<M, PF, C>(
        &self,
        other: &M,
        pair_filter: PF,
        comparator: C,
    ) -> ModularProductResult<Self::Index, M::Index>
    where
        Self: SparseValuedMatrix2D,
        M: SparseSquareMatrix + SparseValuedMatrix2D,
        PF: FnMut(Self::Index, M::Index) -> bool,
        C: Fn(Option<<Self as ValuedMatrix>::Value>, Option<<M as ValuedMatrix>::Value>) -> bool;

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
    fn modular_product_filtered<M, PF>(
        &self,
        other: &M,
        pair_filter: PF,
    ) -> ModularProductResult<Self::Index, M::Index>
    where
        M: SparseSquareMatrix,
        PF: FnMut(Self::Index, M::Index) -> bool,
    {
        let vertex_pairs = collect_pairs(self.order(), other.order(), pair_filter);
        let matrix = modular_product_core(&vertex_pairs, |u1, v1, u2, v2| {
            self.has_entry(u1, v1) == other.has_entry(u2, v2)
        });
        ModularProductResult { matrix, vertex_pairs }
    }

    #[inline]
    fn labeled_modular_product_filtered<M, PF, C>(
        &self,
        other: &M,
        pair_filter: PF,
        comparator: C,
    ) -> ModularProductResult<Self::Index, M::Index>
    where
        Self: SparseValuedMatrix2D,
        M: SparseSquareMatrix + SparseValuedMatrix2D,
        PF: FnMut(Self::Index, M::Index) -> bool,
        C: Fn(Option<<Self as ValuedMatrix>::Value>, Option<<M as ValuedMatrix>::Value>) -> bool,
    {
        let vertex_pairs = collect_pairs(self.order(), other.order(), pair_filter);
        let matrix = modular_product_core(&vertex_pairs, |u1, v1, u2, v2| {
            comparator(self.sparse_value_at(u1, v1), other.sparse_value_at(u2, v2))
        });
        ModularProductResult { matrix, vertex_pairs }
    }

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
