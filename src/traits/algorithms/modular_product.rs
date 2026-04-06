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
    traits::{
        BiMatrix2D, Edges, Graph, Matrix, Matrix2D, MonopartiteGraph, MonoplexGraph,
        PositiveInteger, RankSelectSparseMatrix, SizedRowsSparseMatrix2D, SizedSparseMatrix,
        SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, SparseSquareMatrix,
        SparseValuedMatrix2D, SquareMatrix, ValuedMatrix,
    },
};

/// Result of a lambda-based modular product computation.
///
/// Contains both the product adjacency matrix and the vertex pairs that
/// form the vertex set of the product (i.e., the pairs that passed the
/// `pair_filter`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModularProductResult<I1, I2> {
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(I1, I2)>,
}

/// Symmetric edge wrapper for a modular-product graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModularProductGraphEdges {
    matrix: BitSquareMatrix,
}

impl ModularProductGraphEdges {
    /// Creates a new modular-product edge set from a symmetric adjacency
    /// matrix.
    #[inline]
    #[must_use]
    pub const fn new(matrix: BitSquareMatrix) -> Self {
        Self { matrix }
    }

    /// Returns the underlying adjacency matrix.
    #[inline]
    #[must_use]
    pub const fn inner(&self) -> &BitSquareMatrix {
        &self.matrix
    }

    /// Decomposes into the inner adjacency matrix.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> BitSquareMatrix {
        self.matrix
    }
}

impl Matrix for ModularProductGraphEdges {
    type Coordinates = <BitSquareMatrix as Matrix>::Coordinates;

    #[inline]
    fn shape(&self) -> Vec<usize> {
        self.matrix.shape()
    }
}

impl Matrix2D for ModularProductGraphEdges {
    type RowIndex = usize;
    type ColumnIndex = usize;

    #[inline]
    fn number_of_rows(&self) -> usize {
        self.matrix.number_of_rows()
    }

    #[inline]
    fn number_of_columns(&self) -> usize {
        self.matrix.number_of_columns()
    }
}

impl crate::traits::SquareMatrix for ModularProductGraphEdges {
    type Index = usize;

    #[inline]
    fn order(&self) -> usize {
        self.matrix.order()
    }
}

impl SparseMatrix for ModularProductGraphEdges {
    type SparseIndex = usize;
    type SparseCoordinates<'a>
        = <BitSquareMatrix as SparseMatrix>::SparseCoordinates<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.matrix.sparse_coordinates()
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.matrix.last_sparse_coordinates()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
}

impl SizedSparseMatrix for ModularProductGraphEdges {
    #[inline]
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.matrix.number_of_defined_values()
    }
}

impl SparseSquareMatrix for ModularProductGraphEdges {
    #[inline]
    fn number_of_defined_diagonal_values(&self) -> usize {
        self.matrix.number_of_defined_diagonal_values()
    }
}

impl SparseMatrix2D for ModularProductGraphEdges {
    type SparseRow<'a>
        = <BitSquareMatrix as SparseMatrix2D>::SparseRow<'a>
    where
        Self: 'a;
    type SparseColumns<'a>
        = <BitSquareMatrix as SparseMatrix2D>::SparseColumns<'a>
    where
        Self: 'a;
    type SparseRows<'a>
        = <BitSquareMatrix as SparseMatrix2D>::SparseRows<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_row(&self, row: usize) -> Self::SparseRow<'_> {
        self.matrix.sparse_row(row)
    }

    #[inline]
    fn has_entry(&self, row: usize, column: usize) -> bool {
        self.matrix.has_entry(row, column)
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.matrix.sparse_columns()
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.matrix.sparse_rows()
    }
}

impl SizedRowsSparseMatrix2D for ModularProductGraphEdges {
    type SparseRowSizes<'a>
        = <BitSquareMatrix as SizedRowsSparseMatrix2D>::SparseRowSizes<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.matrix.sparse_row_sizes()
    }

    #[inline]
    fn number_of_defined_values_in_row(&self, row: usize) -> usize {
        self.matrix.number_of_defined_values_in_row(row)
    }
}

impl RankSelectSparseMatrix for ModularProductGraphEdges {
    #[inline]
    fn rank(&self, coordinates: &Self::Coordinates) -> Self::SparseIndex {
        SparseMatrix::sparse_coordinates(self)
            .position(|candidate| candidate == *coordinates)
            .unwrap_or_else(|| panic!("coordinates {coordinates:?} must refer to a defined edge"))
    }

    #[inline]
    fn select(&self, sparse_index: Self::SparseIndex) -> Self::Coordinates {
        SparseMatrix::sparse_coordinates(self)
            .nth(sparse_index)
            .unwrap_or_else(|| panic!("sparse index {sparse_index} must refer to a defined edge"))
    }
}

impl SizedSparseMatrix2D for ModularProductGraphEdges {
    #[inline]
    fn rank_row(&self, row: Self::RowIndex) -> Self::SparseIndex {
        self.sparse_row_sizes().take(row).sum()
    }

    #[inline]
    fn select_row(&self, sparse_index: Self::SparseIndex) -> Self::RowIndex {
        self.select(sparse_index).0
    }

    #[inline]
    fn select_column(&self, sparse_index: Self::SparseIndex) -> Self::ColumnIndex {
        self.select(sparse_index).1
    }
}

impl BiMatrix2D for ModularProductGraphEdges {
    type Matrix = Self;
    type TransposedMatrix = Self;

    #[inline]
    fn matrix(&self) -> &Self::Matrix {
        self
    }

    #[inline]
    fn transposed(&self) -> &Self::TransposedMatrix {
        self
    }
}

impl Edges for ModularProductGraphEdges {
    type Edge = (usize, usize);
    type SourceNodeId = usize;
    type DestinationNodeId = usize;
    type EdgeId = usize;
    type Matrix = Self;

    #[inline]
    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

/// First-class modular-product graph object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModularProductGraph<I1, I2> {
    edges: ModularProductGraphEdges,
    vertex_pairs: Vec<(I1, I2)>,
}

impl<I1, I2> ModularProductGraph<I1, I2> {
    /// Creates a modular-product graph from adjacency matrix and vertex pairs.
    ///
    /// # Panics
    ///
    /// Panics if the adjacency order does not match the number of vertex pairs.
    #[inline]
    #[must_use]
    pub fn new(matrix: BitSquareMatrix, vertex_pairs: Vec<(I1, I2)>) -> Self {
        assert_eq!(
            matrix.order(),
            vertex_pairs.len(),
            "modular product order must match the number of vertex pairs"
        );

        Self { edges: ModularProductGraphEdges::new(matrix), vertex_pairs }
    }

    /// Returns the underlying adjacency matrix.
    #[inline]
    #[must_use]
    pub const fn matrix(&self) -> &BitSquareMatrix {
        self.edges.inner()
    }

    /// Returns the product vertex pairs.
    #[inline]
    #[must_use]
    pub fn vertex_pairs(&self) -> &[(I1, I2)] {
        &self.vertex_pairs
    }

    /// Returns the product vertex pair for the given dense node id.
    #[inline]
    #[must_use]
    pub fn vertex_pair(&self, node_id: usize) -> Option<&(I1, I2)> {
        self.vertex_pairs.get(node_id)
    }

    /// Decomposes into adjacency matrix and vertex pairs.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (BitSquareMatrix, Vec<(I1, I2)>) {
        (self.edges.into_inner(), self.vertex_pairs)
    }
}

impl<I1, I2> From<ModularProductResult<I1, I2>> for ModularProductGraph<I1, I2> {
    #[inline]
    fn from(value: ModularProductResult<I1, I2>) -> Self {
        Self::new(value.matrix, value.vertex_pairs)
    }
}

impl<I1, I2> Graph for ModularProductGraph<I1, I2> {
    #[inline]
    fn has_nodes(&self) -> bool {
        !self.vertex_pairs.is_empty()
    }

    #[inline]
    fn has_edges(&self) -> bool {
        !self.edges.is_empty()
    }
}

impl<I1, I2> MonopartiteGraph for ModularProductGraph<I1, I2>
where
    I1: crate::traits::Symbol,
    I2: crate::traits::Symbol,
{
    type NodeId = usize;
    type NodeSymbol = (I1, I2);
    type Nodes = Vec<(I1, I2)>;

    #[inline]
    fn nodes_vocabulary(&self) -> &Self::Nodes {
        &self.vertex_pairs
    }
}

impl<I1, I2> MonoplexGraph for ModularProductGraph<I1, I2> {
    type Edge = (usize, usize);
    type Edges = ModularProductGraphEdges;

    #[inline]
    fn edges(&self) -> &Self::Edges {
        &self.edges
    }
}

impl<I1, I2> ModularProductResult<I1, I2> {
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

    /// Converts the result into a first-class modular-product graph.
    #[inline]
    #[must_use]
    pub fn into_graph(self) -> ModularProductGraph<I1, I2> {
        self.into()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        impls::{SquareCSR2D, ValuedCSR2D},
        traits::{
            BiMatrix2D, Edges, Graph, Matrix, Matrix2D, MatrixMut, MonopartiteGraph, MonoplexGraph,
            RankSelectSparseMatrix, SizedRowsSparseMatrix2D, SizedSparseMatrix,
            SizedSparseMatrix2D, SparseMatrix, SparseMatrix2D, SparseMatrixMut, SparseSquareMatrix,
            SquareMatrix, UndirectedMonopartiteMonoplexGraph,
        },
    };

    type TestValued = SquareCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;

    fn build_valued(n: usize, edges: &[(usize, usize, u8)]) -> TestValued {
        let mut valued: ValuedCSR2D<usize, usize, usize, u8> =
            SparseMatrixMut::with_sparse_shaped_capacity((n, n), edges.len() * 2);
        let mut all: Vec<(usize, usize, u8)> = Vec::with_capacity(edges.len() * 2);
        for &(row, column, value) in edges {
            all.push((row, column, value));
            if row != column {
                all.push((column, row, value));
            }
        }
        all.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        for (row, column, value) in all {
            MatrixMut::add(&mut valued, (row, column, value)).unwrap();
        }
        SquareCSR2D::from_parts(valued, 0)
    }

    #[test]
    fn test_modular_product_graph_edges_delegate_matrix_traits() {
        let matrix = BitSquareMatrix::from_symmetric_edges(3, [(0, 1), (1, 2)]);
        let edges = ModularProductGraphEdges::new(matrix.clone());

        assert_eq!(edges.inner(), &matrix);
        assert_eq!(edges.clone().into_inner(), matrix.clone());
        assert_eq!(Matrix::shape(&edges), Matrix::shape(&matrix));
        assert_eq!(Matrix2D::number_of_rows(&edges), Matrix2D::number_of_rows(&matrix));
        assert_eq!(Matrix2D::number_of_columns(&edges), Matrix2D::number_of_columns(&matrix));
        assert_eq!(SquareMatrix::order(&edges), SquareMatrix::order(&matrix));
        assert_eq!(
            SparseMatrix::sparse_coordinates(&edges).collect::<Vec<_>>(),
            SparseMatrix::sparse_coordinates(&matrix).collect::<Vec<_>>()
        );
        assert_eq!(
            SparseMatrix::last_sparse_coordinates(&edges),
            SparseMatrix::last_sparse_coordinates(&matrix)
        );
        assert_eq!(SparseMatrix::is_empty(&edges), SparseMatrix::is_empty(&matrix));
        assert_eq!(
            SizedSparseMatrix::number_of_defined_values(&edges),
            SizedSparseMatrix::number_of_defined_values(&matrix)
        );
        assert_eq!(
            SparseSquareMatrix::number_of_defined_diagonal_values(&edges),
            SparseSquareMatrix::number_of_defined_diagonal_values(&matrix)
        );
        assert_eq!(
            SparseMatrix2D::sparse_row(&edges, 1).collect::<Vec<_>>(),
            SparseMatrix2D::sparse_row(&matrix, 1).collect::<Vec<_>>()
        );
        assert_eq!(
            SparseMatrix2D::has_entry(&edges, 0, 1),
            SparseMatrix2D::has_entry(&matrix, 0, 1)
        );
        assert_eq!(
            SparseMatrix2D::sparse_columns(&edges).collect::<Vec<_>>(),
            SparseMatrix2D::sparse_columns(&matrix).collect::<Vec<_>>()
        );
        assert_eq!(
            SparseMatrix2D::sparse_rows(&edges).collect::<Vec<_>>(),
            SparseMatrix2D::sparse_rows(&matrix).collect::<Vec<_>>()
        );
        assert_eq!(
            SizedRowsSparseMatrix2D::sparse_row_sizes(&edges).collect::<Vec<_>>(),
            SizedRowsSparseMatrix2D::sparse_row_sizes(&matrix).collect::<Vec<_>>()
        );
        assert_eq!(
            SizedRowsSparseMatrix2D::number_of_defined_values_in_row(&edges, 1),
            SizedRowsSparseMatrix2D::number_of_defined_values_in_row(&matrix, 1)
        );

        let coordinates = SparseMatrix::sparse_coordinates(&edges).collect::<Vec<_>>();
        assert_eq!(RankSelectSparseMatrix::rank(&edges, &coordinates[1]), 1);
        assert_eq!(RankSelectSparseMatrix::select(&edges, 1), coordinates[1]);
        assert_eq!(
            SizedSparseMatrix2D::rank_row(&edges, 1),
            SizedRowsSparseMatrix2D::number_of_defined_values_in_row(&matrix, 0)
        );
        assert_eq!(SizedSparseMatrix2D::select_row(&edges, 1), coordinates[1].0);
        assert_eq!(SizedSparseMatrix2D::select_column(&edges, 1), coordinates[1].1);
        let edges_ptr = core::ptr::from_ref(&edges);
        assert!(core::ptr::eq(BiMatrix2D::matrix(&edges), edges_ptr));
        assert!(core::ptr::eq(BiMatrix2D::transposed(&edges), edges_ptr));
        assert!(core::ptr::eq(Edges::matrix(&edges), edges_ptr));
    }

    #[test]
    #[should_panic(expected = "coordinates (0, 0) must refer to a defined edge")]
    fn test_modular_product_graph_edges_rank_panics_on_missing_coordinates() {
        let edges =
            ModularProductGraphEdges::new(BitSquareMatrix::from_symmetric_edges(3, [(0, 1)]));
        let _ = RankSelectSparseMatrix::rank(&edges, &(0, 0));
    }

    #[test]
    #[should_panic(expected = "sparse index 2 must refer to a defined edge")]
    fn test_modular_product_graph_edges_select_panics_on_missing_sparse_index() {
        let edges =
            ModularProductGraphEdges::new(BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]));
        let _ = RankSelectSparseMatrix::select(&edges, 2);
    }

    #[test]
    fn test_modular_product_graph_empty_accessors() {
        let graph = ModularProductGraph::<u8, u8>::new(BitSquareMatrix::new(0), vec![]);

        assert!(!Graph::has_nodes(&graph));
        assert!(!Graph::has_edges(&graph));
        assert_eq!(graph.matrix().order(), 0);
        assert_eq!(graph.vertex_pairs(), &[]);
        assert_eq!(graph.vertex_pair(0), None);
        assert!(MonopartiteGraph::nodes_vocabulary(&graph).is_empty());
        assert!(SparseMatrix::is_empty(MonoplexGraph::edges(&graph)));
    }

    #[test]
    #[should_panic(expected = "modular product order must match the number of vertex pairs")]
    fn test_modular_product_graph_new_panics_on_shape_mismatch() {
        let _ = ModularProductGraph::new(BitSquareMatrix::new(2), vec![(0usize, 0usize)]);
    }

    #[test]
    fn test_modular_product_filtered_and_reference_apis_agree() {
        let left = BitSquareMatrix::from_symmetric_edges(3, [(0, 1), (1, 2)]);
        let right = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);

        let filtered = left.modular_product_filtered(&right, |left, right| left != 2 || right != 0);
        let reference = left.modular_product(&right, filtered.vertex_pairs());

        assert_eq!(reference, *filtered.matrix());

        let expected_matrix = filtered.matrix().clone();
        let expected_pairs = filtered.vertex_pairs().to_vec();
        let (matrix, pairs) = filtered.into_parts();
        assert_eq!(matrix, expected_matrix);
        assert_eq!(pairs, expected_pairs);
    }

    #[test]
    fn test_labeled_modular_product_filtered_and_reference_apis_agree() {
        let left = build_valued(3, &[(0, 1, 1), (1, 2, 2)]);
        let right = build_valued(3, &[(0, 1, 3), (1, 2, 5)]);

        let filtered = left.labeled_modular_product_filtered(
            &right,
            |left, right| left == right,
            |left, right| {
                match (left, right) {
                    (None, None) => true,
                    (Some(left), Some(right)) => left % 2 == right % 2,
                    _ => false,
                }
            },
        );
        let reference =
            left.labeled_modular_product(&right, filtered.vertex_pairs(), |left, right| {
                match (left, right) {
                    (None, None) => true,
                    (Some(left), Some(right)) => left % 2 == right % 2,
                    _ => false,
                }
            });

        assert_eq!(reference, *filtered.matrix());
        assert_eq!(filtered.vertex_pairs(), &[(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_modular_product_result_into_graph_preserves_pairs_and_adjacency() {
        let left = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);
        let right = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);

        let graph = left.modular_product_filtered(&right, |_, _| true).into_graph();

        assert_eq!(graph.number_of_nodes(), 4);
        assert_eq!(graph.number_of_edges(), 4);
        assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(graph.nodes().collect::<Vec<_>>(), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(graph.vertex_pair(2), Some(&(1, 0)));
        assert_eq!(graph.neighbors(0).collect::<Vec<_>>(), vec![3]);
        assert_eq!(graph.neighbors(1).collect::<Vec<_>>(), vec![2]);
        assert_eq!(graph.neighbors(2).collect::<Vec<_>>(), vec![1]);
        assert_eq!(graph.neighbors(3).collect::<Vec<_>>(), vec![0]);
        assert_eq!(graph.matrix().order(), 4);
    }

    #[test]
    fn test_modular_product_graph_roundtrips_parts() {
        let left = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);
        let right = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);

        let product = left.modular_product_filtered(&right, |_, _| true);
        let graph = product.clone().into_graph();
        let (matrix, vertex_pairs) = graph.into_parts();

        assert_eq!(matrix, *product.matrix());
        assert_eq!(vertex_pairs, product.vertex_pairs().to_vec());
    }
}
