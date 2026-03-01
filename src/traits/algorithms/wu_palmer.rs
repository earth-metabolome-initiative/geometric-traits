//! Implementation for the 'Wu Palmer' trait based on the algorithm
//! implementation
use alloc::vec::Vec;
use core::f64;

use crate::traits::{
    Edges, Kahn, KahnError, MonoplexMonopartiteGraph, ScalarSimilarity,
    algorithms::root_nodes::RootNodes,
};

/// Struct for the Wu-Palmer similarity trait
#[derive(Debug)]
pub struct WuPalmerResult<'graph, G: ?Sized + MonoplexMonopartiteGraph> {
    /// the graph to be analyzed
    graph: &'graph G,
    /// Root nodes of the graph
    root_nodes: Vec<G::NodeId>,
}

/// Trait providing the `Wu-Palmer` depth based similarity
///
/// # Reference
/// The implementation of the Wu-Palmer based similarity score as described in [Verb Semantics and Lexical Selection](https://arxiv.org/pdf/cmp-lg/9406033)
pub trait WuPalmer: MonoplexMonopartiteGraph {
    /// The method for applying the Wu-Palmer algorithm
    ///
    /// # Errors
    /// - If the graph is not a dag
    ///
    /// # Complexity
    ///
    /// `wu_palmer` preparation is O(V + E). Each call to `similarity` is
    /// O(V + E) per root node.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, ScalarSimilarity, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
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
    /// let wu_palmer = graph.wu_palmer().unwrap();
    /// assert!(wu_palmer.similarity(&0, &0) > 0.99);
    /// assert!(wu_palmer.similarity(&0, &1) < 0.99);
    /// ```
    fn wu_palmer(&self) -> Result<WuPalmerResult<'_, Self>, KahnError> {
        // Check whether the graph is a DAG (characterize by having no cycles)
        let _topological_ordering = self.edges().matrix().kahn()?;
        let root_nodes = self.root_nodes();
        Ok(WuPalmerResult { graph: self, root_nodes })
    }
}

impl<G> ScalarSimilarity<G::NodeId, G::NodeId> for WuPalmerResult<'_, G>
where
    G: MonoplexMonopartiteGraph,
{
    type Similarity = f64;
    #[allow(clippy::cast_precision_loss)]
    fn similarity(&self, left: &G::NodeId, right: &G::NodeId) -> Self::Similarity {
        if left == right {
            return 1.0;
        }
        let mut max_similarity = 0.0;
        for root_node in &self.root_nodes {
            let (Some(n1), Some(n2), n3) =
                // Use root depth = 1 to match the canonical Wu-Palmer score:
                // 2 * depth(LCS) / (depth(left) + depth(right)).
                wu_palmer_depth(self.graph, 1, *root_node, *left, *right)
            else {
                continue;
            };
            let mut denominator = n1 as f64 + n2 as f64;
            if denominator < f64::EPSILON {
                denominator = f64::EPSILON;
            }
            let similarity = (2.0 * n3 as f64) / denominator;
            if similarity > max_similarity {
                max_similarity = similarity;
            }
        }
        max_similarity
    }
}

/// Stack frame for the iterative post-order DFS used in `wu_palmer_depth`.
struct WuPalmerFrame<NodeId: Copy> {
    depth: usize,
    successors: Vec<NodeId>,
    idx: usize,
    n1: Option<usize>,
    n2: Option<usize>,
    /// Depth of the deepest common ancestor found in this subtree.
    n3: usize,
}

/// Iterative post-order DFS that computes the Wu-Palmer depth triple for the
/// subtree rooted at `initial_node`.
///
/// Returns `(n1, n2, n3)` where:
/// - `n1` = minimum depth of `left` in this subtree (`None` if not found)
/// - `n2` = minimum depth of `right` in this subtree (`None` if not found)
/// - `n3` = depth of the deepest common ancestor node that has both `left` and
///   `right` in its subtree (equals `initial_depth` when neither or only one
///   target is reachable)
fn wu_palmer_depth<G>(
    graph: &G,
    initial_depth: usize,
    initial_node: G::NodeId,
    left: G::NodeId,
    right: G::NodeId,
) -> (Option<usize>, Option<usize>, usize)
where
    G: MonoplexMonopartiteGraph + ?Sized,
{
    let init_n1 = if initial_node == left { Some(initial_depth) } else { None };
    let init_n2 = if initial_node == right { Some(initial_depth) } else { None };
    let init_successors: Vec<G::NodeId> = graph.successors(initial_node).collect();
    let mut stack = vec![WuPalmerFrame {
        depth: initial_depth,
        successors: init_successors,
        idx: 0,
        n1: init_n1,
        n2: init_n2,
        n3: initial_depth,
    }];

    loop {
        let (should_push, should_pop) = {
            let frame = stack.last_mut().expect("stack is non-empty inside loop");
            if frame.idx < frame.successors.len() {
                let successor = frame.successors[frame.idx];
                let new_depth = frame.depth + 1;
                frame.idx += 1;
                (Some((successor, new_depth)), false)
            } else {
                (None, true)
            }
        };

        if let Some((successor, new_depth)) = should_push {
            let sn1 = if successor == left { Some(new_depth) } else { None };
            let sn2 = if successor == right { Some(new_depth) } else { None };
            let child_successors: Vec<G::NodeId> = graph.successors(successor).collect();
            stack.push(WuPalmerFrame {
                depth: new_depth,
                successors: child_successors,
                idx: 0,
                n1: sn1,
                n2: sn2,
                n3: new_depth,
            });
        } else if should_pop {
            let child = stack.pop().expect("stack is non-empty when popping");
            let (child_n1, child_n2, child_n3) = (child.n1, child.n2, child.n3);

            if let Some(parent) = stack.last_mut() {
                match (child_n1, child_n2) {
                    (Some(rec_n1), None) => {
                        parent.n1 = Some(
                            parent.n1.map_or(rec_n1, |n1| if n1 < rec_n1 { n1 } else { rec_n1 }),
                        );
                    }
                    (None, Some(rec_n2)) => {
                        parent.n2 = Some(
                            parent.n2.map_or(rec_n2, |n2| if n2 < rec_n2 { n2 } else { rec_n2 }),
                        );
                    }
                    (None, None) => {}
                    // Both Some: the child subtree contains a common ancestor.
                    (rec_n1, rec_n2) => {
                        if parent.n3 < child_n3 {
                            parent.n3 = child_n3;
                            parent.n1 = rec_n1;
                            parent.n2 = rec_n2;
                        }
                    }
                }
            } else {
                // Stack exhausted: root frame result is the final answer.
                return (child_n1, child_n2, child_n3);
            }
        }
    }
}

impl<G> WuPalmer for G where G: MonoplexMonopartiteGraph {}
