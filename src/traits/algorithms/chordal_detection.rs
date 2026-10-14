//! Submodule providing the `ChordalDetection` trait and its blanket
//! implementation for undirected graphs.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::UndirectedMonopartiteMonoplexGraph;

fn maximum_cardinality_search_ordering_and_positions<G>(graph: &G) -> (Vec<G::NodeId>, Vec<usize>)
where
    G: ?Sized + UndirectedMonopartiteMonoplexGraph,
{
    let number_of_nodes = graph.number_of_nodes().as_();
    let mut labels = vec![0usize; number_of_nodes];
    let mut numbered = vec![false; number_of_nodes];
    let mut positions = vec![usize::MAX; number_of_nodes];
    let mut buckets: Vec<Vec<G::NodeId>> = vec![Vec::new(); number_of_nodes + 1];
    let mut ordering = Vec::with_capacity(number_of_nodes);
    let mut max_label = 0usize;

    for node in graph.node_ids() {
        buckets[0].push(node);
    }

    for reverse_position in 0..number_of_nodes {
        let vertex = loop {
            while buckets[max_label].is_empty() && max_label > 0 {
                max_label -= 1;
            }

            let candidate = buckets[max_label].pop().expect("MCS bucket queue unexpectedly empty");
            let candidate_index = candidate.as_();

            if !numbered[candidate_index] && labels[candidate_index] == max_label {
                break candidate;
            }
        };

        let vertex_index = vertex.as_();
        let position = number_of_nodes - reverse_position - 1;
        numbered[vertex_index] = true;
        positions[vertex_index] = position;
        ordering.push(vertex);

        for neighbor in graph.neighbors(vertex) {
            let neighbor_index = neighbor.as_();
            if numbered[neighbor_index] {
                continue;
            }

            labels[neighbor_index] += 1;
            let new_label = labels[neighbor_index];
            buckets[new_label].push(neighbor);
            if new_label > max_label {
                max_label = new_label;
            }
        }
    }

    ordering.reverse();
    (ordering, positions)
}

fn is_perfect_elimination_ordering_with_positions<G>(
    graph: &G,
    ordering: &[G::NodeId],
    positions: &[usize],
) -> bool
where
    G: ?Sized + UndirectedMonopartiteMonoplexGraph,
{
    for &node in ordering {
        let node_position = positions[node.as_()];
        let mut parent_position = usize::MAX;
        let mut parent = node;

        for neighbor in graph.neighbors(node) {
            let neighbor_position = positions[neighbor.as_()];
            if neighbor_position > node_position && neighbor_position < parent_position {
                parent_position = neighbor_position;
                parent = neighbor;
            }
        }

        if parent_position == usize::MAX {
            continue;
        }

        for neighbor in graph.neighbors(node) {
            let neighbor_position = positions[neighbor.as_()];
            if neighbor_position > node_position
                && neighbor != parent
                && !graph.has_successor(parent, neighbor)
            {
                return false;
            }
        }
    }

    true
}

/// Trait providing chordality recognition utilities for undirected graphs.
///
/// The implementation uses Maximum Cardinality Search (MCS) to build a
/// candidate perfect elimination ordering (PEO), then verifies that the
/// ordering satisfies the clique condition on later neighbors.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::{SortedVec, SymmetricCSR2D},
///     prelude::*,
///     traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
/// };
///
/// let nodes: Vec<usize> = vec![0, 1, 2, 3];
/// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (1, 3), (2, 3)];
/// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
///     .expected_number_of_symbols(nodes.len())
///     .symbols(nodes.into_iter().enumerate())
///     .build()
///     .unwrap();
/// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
///     .expected_number_of_edges(edges.len())
///     .expected_shape(nodes.len())
///     .edges(edges.into_iter())
///     .build()
///     .unwrap();
/// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
///
/// assert!(graph.is_chordal());
/// assert!(graph.perfect_elimination_ordering().is_some());
/// ```
pub trait ChordalDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns the vertex ordering produced by Maximum Cardinality Search.
    ///
    /// The returned ordering is arranged from earliest to latest in the
    /// candidate elimination order, so it is a perfect elimination ordering
    /// exactly when the graph is chordal.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V + E) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (1, 3)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// assert!(graph.is_perfect_elimination_ordering(&graph.maximum_cardinality_search_ordering()));
    /// ```
    fn maximum_cardinality_search_ordering(&self) -> Vec<Self::NodeId> {
        maximum_cardinality_search_ordering_and_positions(self).0
    }

    /// Returns whether the provided ordering is a valid perfect elimination
    /// ordering for the graph.
    ///
    /// The slice must contain every node exactly once and in elimination order
    /// from earliest to latest.
    ///
    /// # Complexity
    ///
    /// O(V + E log V) time in the worst case and O(V) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// assert!(graph.is_perfect_elimination_ordering(&[0, 1, 2]));
    /// assert!(!graph.is_perfect_elimination_ordering(&[1, 0, 2]));
    /// ```
    fn is_perfect_elimination_ordering(&self, ordering: &[Self::NodeId]) -> bool {
        if self.has_self_loops() {
            return false;
        }

        let number_of_nodes = self.number_of_nodes().as_();
        if ordering.len() != number_of_nodes {
            return false;
        }

        let mut positions = vec![usize::MAX; number_of_nodes];

        for (position, &node) in ordering.iter().enumerate() {
            let node_index = node.as_();
            if node_index >= number_of_nodes || positions[node_index] != usize::MAX {
                return false;
            }
            positions[node_index] = position;
        }

        is_perfect_elimination_ordering_with_positions(self, ordering, &positions)
    }

    /// Returns a perfect elimination ordering when the graph is chordal.
    ///
    /// Returns `None` for non-chordal graphs or graphs with self-loops.
    ///
    /// # Complexity
    ///
    /// O(V + E log V) time in the worst case and O(V + E) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (1, 3), (2, 3)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// assert!(graph.perfect_elimination_ordering().is_some());
    /// ```
    #[inline]
    fn perfect_elimination_ordering(&self) -> Option<Vec<Self::NodeId>> {
        if self.has_self_loops() {
            return None;
        }

        let (ordering, positions) = maximum_cardinality_search_ordering_and_positions(self);
        is_perfect_elimination_ordering_with_positions(self, &ordering, &positions)
            .then_some(ordering)
    }

    /// Returns whether the graph is chordal.
    ///
    /// Chordal graphs are exactly the undirected graphs that admit a perfect
    /// elimination ordering.
    ///
    /// # Complexity
    ///
    /// O(V + E log V) time in the worst case and O(V + E) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{ChordalDetection, EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 3), (1, 2), (2, 3)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// assert!(!graph.is_chordal());
    /// ```
    #[inline]
    fn is_chordal(&self) -> bool {
        if self.has_self_loops() {
            return false;
        }

        let (ordering, positions) = maximum_cardinality_search_ordering_and_positions(self);
        is_perfect_elimination_ordering_with_positions(self, &ordering, &positions)
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> ChordalDetection for G {}
