//! Submodule defining a trait characterizing monoplex bipartite graphs.
//!
//! These graphs are characterized by the fact that:
//!
//! * They are bipartite, i.e., they have two types of nodes.
//! * They are monoplex, i.e., they have only one type of edges.

use super::{BipartiteGraph, Edges, MonoplexGraph};

/// Trait defining the properties of a monoplex bipartite graph.
pub trait MonoplexBipartiteGraph:
    MonoplexGraph<Edges = <Self as MonoplexBipartiteGraph>::MonoplexBipartiteEdges> + BipartiteGraph
{
    /// The edges of the graph.
    type MonoplexBipartiteEdges: Edges<
            SourceNodeId = <Self as BipartiteGraph>::LeftNodeId,
            DestinationNodeId = <Self as BipartiteGraph>::RightNodeId,
        >;

    /// Returns a DOT representation for the Monoplex Bipartite Graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::CSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::naive_structs::GenericEdgesBuilder;
    /// use geometric_traits::naive_structs::named_types::BiGraph;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let left_nodes: Vec<usize> = vec![0, 1];
    /// let right_nodes: Vec<usize> = vec![0, 1];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 0)];
    /// let left_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(left_nodes.len())
    ///     .symbols(left_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let right_nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(right_nodes.len())
    ///     .symbols(right_nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<usize, usize, usize> = GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((left_nodes.len(), right_nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: BiGraph<usize, usize> = BiGraph::try_from((left_nodes, right_nodes, edges)).unwrap();
    ///
    /// let dot = graph.to_mb_dot();
    /// assert!(dot.contains("L0 -> R1;"));
    /// assert!(dot.contains("L1 -> R0;"));
    /// ```
    fn to_mb_dot(&self) -> String {
        use std::fmt::Write;
        let mut dot = String::new();
        writeln!(dot, "  graph {{").unwrap();

        for left_node_id in self.left_node_ids() {
            writeln!(dot, "  L{left_node_id} [color=red];").unwrap();
        }

        for right_node_id in self.right_node_ids() {
            writeln!(dot, "  R{right_node_id} [color=blue];").unwrap();
        }

        for (src, dst) in self.edges().sparse_coordinates() {
            writeln!(dot, "  L{src} -> R{dst};").unwrap();
        }

        writeln!(dot, "  }}").unwrap();
        dot
    }
}

impl<G> MonoplexBipartiteGraph for G
where
    G: MonoplexGraph,
    G: BipartiteGraph,
    G::Edges: Edges<
            SourceNodeId = <G as BipartiteGraph>::LeftNodeId,
            DestinationNodeId = <G as BipartiteGraph>::RightNodeId,
        >,
{
    type MonoplexBipartiteEdges = G::Edges;
}
