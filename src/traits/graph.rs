//! Module defining a bipartite graph.

/// Trait for a bipartite graph.
pub trait Graph {
    /// Returns whether the graph has any nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: DiGraph<usize> = DiGraph::from((nodes, edges));
    ///
    /// assert!(graph.has_nodes());
    /// ```
    fn has_nodes(&self) -> bool;

    /// Returns whether the graph has any edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: DiGraph<usize> = DiGraph::from((nodes, edges));
    ///
    /// assert!(graph.has_edges());
    /// ```
    fn has_edges(&self) -> bool;
}
