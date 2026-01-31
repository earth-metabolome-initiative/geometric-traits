//! Submodule defining a trait characterizing monoplex monopartite graphs.
//!
//! These graphs are characterized by the fact that:
//!
//! * They are monopartite, i.e., they have only one type of nodes.
//! * They are monoplex, i.e., they have only one type of edges.

use super::{MonopartiteEdges, MonopartiteGraph, MonoplexGraph};
#[cfg(feature = "alloc")]
use crate::traits::IntoUsize;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Trait defining the properties of monoplex monopartite graphs.
pub trait MonoplexMonopartiteGraph: MonoplexGraph<Edges = <Self as MonoplexMonopartiteGraph>::MonoplexMonopartiteEdges>
    + MonopartiteGraph<NodeId = <<Self as MonoplexMonopartiteGraph>::MonoplexMonopartiteEdges as MonopartiteEdges>::NodeId>
{
    /// The type of edges in the graph.
    type MonoplexMonopartiteEdges: MonopartiteEdges;

    /// Returns whether the graph has self-loops.
    fn has_self_loops(&self) -> bool {
        self.edges().has_self_loops()
    }

    /// Returns the number of self-loops in the graph.
    fn number_of_self_loops(&self) -> Self::NodeId {
        self.edges().number_of_self_loops()
    }

    /// Returns whether the current graph labelling follows a
    /// topological order, which means that for every directed edge (u, v),
    /// u comes before v in the ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::impls::SortedVec;
    /// use geometric_traits::impls::SquareCSR2D;
    /// use geometric_traits::prelude::*;
    /// use geometric_traits::traits::{EdgesBuilder, VocabularyBuilder};
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
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
    /// assert!(graph.is_topologically_sorted());
    /// ```
    fn is_topologically_sorted(&self) -> bool {
        self.sparse_coordinates().all(|(src, dst)| src < dst)
    }

    /// Returns the set of unique paths from the provided source node.
    /// 
    /// # Arguments
    /// 
    /// * `source` - The identifier of the source node.
    ///
    /// # Examples
    ///
    /// ```rust
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
    /// let paths = graph.unique_paths_from(0);
    /// assert_eq!(paths.len(), 2);
    /// ```
    #[cfg(feature = "alloc")]
    fn unique_paths_from(
        &self,
        source: Self::NodeId,
    ) -> Vec<Vec<Self::NodeId>> {
        let mut growing_paths = vec![vec![source]];
        let mut growing_paths_tmp = Vec::new();
        let mut paths = Vec::new();

        while !growing_paths.is_empty() {
            for growing_path in &growing_paths {
                let last_node = growing_path[growing_path.len() - 1];
                let mut found_successors = false;
                for successor in self.successors(last_node) {
                    growing_paths_tmp.push({
                        let mut new_path = growing_path.clone();
                        new_path.push(successor);
                        new_path
                    });
                    found_successors = true;
                }
                if !found_successors {
                    paths.push(growing_path.clone());
                }
            }
            core::mem::swap(&mut growing_paths, &mut growing_paths_tmp);
            growing_paths_tmp.clear();
        }

        paths
    }

    /// Returns the set of nodes reachable from the given source node.
    ///
    /// # Arguments
    ///
    /// * `source` - The identifier of the source node.
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
    /// let successors = graph.successors_set(0);
    /// assert_eq!(successors, vec![1, 2]);
    /// ```
    #[cfg(feature = "alloc")]
    fn successors_set(&self, source: Self::NodeId) -> Vec<Self::NodeId> {
        let mut visited_nodes = vec![false; self.number_of_nodes().into_usize()];

        let mut frontier = vec![source];
        let mut temporary_frontier = Vec::new();
        visited_nodes[source.into_usize()] = true;
        let mut reachable_nodes = Vec::new();

        while !frontier.is_empty() {
            for node in frontier.drain(..) {
                for successor in self.successors(node) {
                    if !visited_nodes[successor.into_usize()] {
                        visited_nodes[successor.into_usize()] = true;
                        temporary_frontier.push(successor);
                    }
                }
            }
            reachable_nodes.extend(temporary_frontier.iter().copied());
            core::mem::swap(&mut frontier, &mut temporary_frontier);
            temporary_frontier.clear();
        }

        reachable_nodes.sort_unstable();

        reachable_nodes
    }

    /// Returns whether the provided source node can reach the destination node.
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
    /// assert!(graph.has_path(0, 2));
    /// assert!(!graph.has_path(2, 0));
    /// ```
    #[cfg(feature = "alloc")]
    fn has_path(
        &self,
        source: Self::NodeId,
        destination: Self::NodeId,
    ) -> bool {
        let mut visited_nodes = vec![false; self.number_of_nodes().into_usize()];

        let mut frontier = vec![source];
        let mut temporary_frontier = Vec::new();
        visited_nodes[source.into_usize()] = true;

        while !frontier.is_empty() {
            for node in frontier.drain(..) {
                for successor in self.successors(node) {
                    if successor == destination {
                        return true;
                    }
                    if !visited_nodes[successor.into_usize()] {
                        visited_nodes[successor.into_usize()] = true;
                        temporary_frontier.push(successor);
                    }
                }
            }
            core::mem::swap(&mut frontier, &mut temporary_frontier);
            temporary_frontier.clear();
        }

        false
    }

    /// Returns whether there exist a path from a provided source node to a target node,
    /// possing through the provided node.
    /// 
    /// # Arguments
    /// 
    /// * `source` - The identifier of the source node.
    /// * `destination` - The identifier of the destination node.
    /// * `passing_through` - The identifier of the node that must be passed through.
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
    /// assert!(graph.is_reachable_through(0, 2, 1));
    /// ```
    #[cfg(feature = "alloc")]
    fn is_reachable_through(
        &self,
        source: Self::NodeId,
        destination: Self::NodeId,
        passing_through: Self::NodeId,
    ) -> bool {
        self.has_path(source, passing_through)
            && self.has_path(passing_through, destination)
    }
}

impl<G> MonoplexMonopartiteGraph for G
where
    G: MonopartiteGraph + MonoplexGraph,
    G::Edges: MonopartiteEdges<NodeId = G::NodeId>,
{
    type MonoplexMonopartiteEdges = G::Edges;
}
