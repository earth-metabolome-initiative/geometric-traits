//! Submodule providing the `CycleDetection` trait and its blanket
//! implementation.
use alloc::vec::Vec;

use crate::traits::{IntoUsize, MonoplexMonopartiteGraph};

/// Struct to support cycle detection in a graph.
struct CycleDetector<'graph, G: MonoplexMonopartiteGraph + ?Sized> {
    /// The graph to be analyzed.
    graph: &'graph G,
    /// A vector to keep track of visited nodes.
    visited: Vec<bool>,
    /// A vector to keep track of the recursion stack.
    stack: Vec<G::NodeId>,
}

impl<'graph, G: MonoplexMonopartiteGraph + ?Sized> From<&'graph G> for CycleDetector<'graph, G> {
    fn from(graph: &'graph G) -> Self {
        CycleDetector {
            graph,
            visited: vec![false; graph.number_of_nodes().into_usize()],
            stack: Vec::new(),
        }
    }
}

impl<G: MonoplexMonopartiteGraph + ?Sized> CycleDetector<'_, G> {
    /// Recursive function to detect cycles in the graph.
    fn dfs(&mut self, node: G::NodeId) -> bool {
        if !self.visited[node.into_usize()] {
            self.visited[node.into_usize()] = true;
            self.stack.push(node);

            for successor_node_id in self.graph.successors(node) {
                if !self.visited[successor_node_id.into_usize()] && self.dfs(successor_node_id)
                    || self.stack.contains(&successor_node_id)
                {
                    return true;
                }
            }
        }

        self.stack.pop();
        false
    }
}

/// Trait providing the `has_cycle` method, which returns true if the graph
/// contains a cycle, and false otherwise. The cycle detection algorithm is
/// based on a depth-first search (DFS) approach.
pub trait CycleDetection: MonoplexMonopartiteGraph {
    /// Returns true if the graph contains a cycle, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
    /// let edges: Vec<(usize, usize)> = vec![(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)];
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
    /// assert!(!graph.has_cycle());
    /// ```
    fn has_cycle(&self) -> bool {
        let mut cycle_detector = CycleDetector::from(self);
        for node in self.node_ids() {
            if cycle_detector.dfs(node) {
                return true;
            }
        }
        false
    }
}

impl<G: MonoplexMonopartiteGraph> CycleDetection for G {}
