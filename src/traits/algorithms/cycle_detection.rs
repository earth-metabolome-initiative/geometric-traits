//! Submodule providing the `CycleDetection` trait and its blanket
//! implementation.
use alloc::vec::Vec;

use bitvec::vec::BitVec;

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

/// Trait providing the `has_cycle` method, which returns true if the graph
/// contains a cycle, and false otherwise. The cycle detection algorithm is
/// based on a depth-first search (DFS) approach.
pub trait CycleDetection: MonoplexMonopartiteGraph {
    /// Returns true if the graph contains a cycle, false otherwise.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space. Node membership in the current DFS path
    /// is tracked with a `Vec<bool>` for O(1) per lookup instead of the
    /// O(n) `Vec::contains` that the naive stack approach would require.
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
        let n = self.number_of_nodes().as_();
        let mut visited: BitVec = BitVec::repeat(false, n);
        let mut on_stack: BitVec = BitVec::repeat(false, n);

        // DFS stack entries: (node, collected successors, next-successor index)
        let mut dfs_stack: Vec<(Self::NodeId, Vec<Self::NodeId>, usize)> = Vec::new();

        for start in self.node_ids() {
            if visited[start.as_()] {
                continue;
            }
            visited.set(start.as_(), true);
            on_stack.set(start.as_(), true);
            dfs_stack.push((start, self.successors(start).collect(), 0));

            loop {
                if dfs_stack.is_empty() {
                    break;
                }
                let top_idx = dfs_stack.last().unwrap().2;
                let top_len = dfs_stack.last().unwrap().1.len();

                if top_idx < top_len {
                    let successor = dfs_stack.last().unwrap().1[top_idx];
                    dfs_stack.last_mut().unwrap().2 += 1;

                    if on_stack[successor.as_()] {
                        return true;
                    }
                    if !visited[successor.as_()] {
                        visited.set(successor.as_(), true);
                        on_stack.set(successor.as_(), true);
                        dfs_stack.push((successor, self.successors(successor).collect(), 0));
                    }
                } else {
                    let (node, _, _) = dfs_stack.pop().unwrap();
                    on_stack.set(node.as_(), false);
                }
            }
        }

        false
    }
}

impl<G: MonoplexMonopartiteGraph> CycleDetection for G {}
