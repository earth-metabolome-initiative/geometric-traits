//! Traits for monopartite undirected graphs.
//!
//! These graphs have the following properties:
//!
//! * All nodes are of the same type.
//! * All edges are bidirectional.

use super::{MonopartiteEdges, MonopartiteGraph, MonoplexMonopartiteGraph, TransposedEdges};
use crate::traits::{
    SizedRowsSparseMatrix2D, SizedSparseBiMatrix2D, SparseMatrix2D, SparseSymmetricMatrix2D,
    TransposedMonoplexGraph,
};

/// Trait defining the properties of a directed graph.
pub trait UndirectedMonopartiteEdges:
    MonopartiteEdges<
        MonopartiteMatrix = <Self as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix,
    > + TransposedEdges<BiMatrix = <Self as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix>
{
    /// Neighbors of a node.
    type SymmetricSquaredMatrix: SparseSymmetricMatrix2D<Index = Self::NodeId>
        + SizedSparseBiMatrix2D;

    /// Returns the neighbors of the node with the given identifier.
    #[inline]
    fn neighbors(
        &self,
        node: Self::NodeId,
    ) -> <Self::SymmetricSquaredMatrix as SparseMatrix2D>::SparseRow<'_> {
        self.matrix().sparse_row(node)
    }

    /// Returns the degree of the node with the given identifier.
    #[inline]
    fn degree(&self, node: Self::NodeId) -> Self::NodeId {
        debug_assert_eq!(
            self.matrix().number_of_defined_values_in_row(node),
            self.matrix().number_of_defined_values_in_column(node)
        );
        self.matrix().number_of_defined_values_in_row(node)
    }

    /// Returns the iterator over the degrees of the nodes in the graph.
    #[inline]
    fn degrees(
        &self,
    ) -> <Self::SymmetricSquaredMatrix as SizedRowsSparseMatrix2D>::SparseRowSizes<'_> {
        self.matrix().sparse_row_sizes()
    }
}

impl<E> UndirectedMonopartiteEdges for E
where
    E: TransposedEdges + MonopartiteEdges<MonopartiteMatrix = E::BiMatrix>,
    E::BiMatrix: SparseSymmetricMatrix2D<Index = E::NodeId, SparseIndex = E::EdgeId>
        + SizedSparseBiMatrix2D<
            SizedSparseMatrix = <E::BiMatrix as SparseSymmetricMatrix2D>::SymmetricSparseMatrix,
            SizedSparseTransposedMatrix = <E::BiMatrix as SparseSymmetricMatrix2D>::SymmetricSparseMatrix,
        >,
{
    type SymmetricSquaredMatrix = E::BiMatrix;
}

/// Trait defining the properties of monopartite undirected graphs.
///
/// This trait combines `MonoplexMonopartiteGraph` and
/// `TransposedMonoplexGraph`, requiring that the edges are undirected
/// (symmetric).
pub trait UndirectedMonopartiteMonoplexGraph:
    MonoplexMonopartiteGraph<MonoplexMonopartiteEdges = Self::UndirectedMonopartiteEdges>
    + TransposedMonoplexGraph<TransposedEdges = Self::UndirectedMonopartiteEdges>
{
    /// The undirected edges of the graph, constrained to have matching node
    /// identifiers.
    type UndirectedMonopartiteEdges: UndirectedMonopartiteEdges<
        NodeId = <Self as MonopartiteGraph>::NodeId,
    >;

    /// Returns the neighbors of the node with the given identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D, UpperTriangularCSR2D},
    ///     naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
    ///     GenericUndirectedMonopartiteEdgesBuilder::<
    ///         _,
    ///         UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    ///         SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    ///     >::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// let neighbors: Vec<usize> = graph.neighbors(1).collect();
    /// assert_eq!(neighbors, vec![0, 2]);
    /// ```
    #[inline]
    fn neighbors(
        &self,
        node: Self::NodeId,
    ) -> <<Self::UndirectedMonopartiteEdges as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix as SparseMatrix2D>::SparseRow<'_>{
        self.edges().neighbors(node)
    }

    /// Returns the degree of the node with the given identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D, UpperTriangularCSR2D},
    ///     naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
    ///     GenericUndirectedMonopartiteEdgesBuilder::<
    ///         _,
    ///         UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    ///         SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    ///     >::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// assert_eq!(graph.degree(1), 2);
    /// ```
    #[inline]
    fn degree(&self, node: Self::NodeId) -> Self::NodeId {
        self.edges().degree(node)
    }

    /// Returns the iterator over the degrees of the nodes in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D, UpperTriangularCSR2D},
    ///     naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
    ///     GenericUndirectedMonopartiteEdgesBuilder::<
    ///         _,
    ///         UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    ///         SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    ///     >::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// let degrees: Vec<usize> = graph.degrees().collect();
    /// assert_eq!(degrees, vec![1, 2, 1]);
    /// ```
    #[inline]
    fn degrees(&self) -> <<Self::UndirectedMonopartiteEdges as UndirectedMonopartiteEdges>::SymmetricSquaredMatrix as SizedRowsSparseMatrix2D>::SparseRowSizes<'_>{
        self.edges().degrees()
    }
}

impl<G> UndirectedMonopartiteMonoplexGraph for G
where
    G: MonoplexMonopartiteGraph
        + TransposedMonoplexGraph<TransposedEdges = G::MonoplexMonopartiteEdges>,
    G::MonoplexMonopartiteEdges: UndirectedMonopartiteEdges,
{
    type UndirectedMonopartiteEdges = G::MonoplexMonopartiteEdges;
}
