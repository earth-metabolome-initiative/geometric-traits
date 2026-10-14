//! Submodule providing `Resnik` trait based on the algorithm implementation.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::{
    prelude::information_content::InformationContentError,
    traits::{
        InformationContent, MonoplexMonopartiteGraph, ScalarSimilarity,
        information_content::InformationContentResult,
    },
};

/// Struct to provide methods to compute Resnik Similarity Score
#[derive(Debug, PartialEq)]
pub struct ResnikResult<'graph, G: ?Sized + MonoplexMonopartiteGraph>(
    InformationContentResult<'graph, G>,
);

impl<'graph, G: ?Sized + MonoplexMonopartiteGraph> AsRef<InformationContentResult<'graph, G>>
    for ResnikResult<'graph, G>
{
    #[inline]
    fn as_ref(&self) -> &InformationContentResult<'graph, G> {
        &self.0
    }
}

impl<'graph, G: ?Sized + MonoplexMonopartiteGraph> From<InformationContentResult<'graph, G>>
    for ResnikResult<'graph, G>
{
    #[inline]
    fn from(value: InformationContentResult<'graph, G>) -> Self {
        Self(value)
    }
}

impl<G> ScalarSimilarity<G::NodeId, G::NodeId> for ResnikResult<'_, G>
where
    G: MonoplexMonopartiteGraph,
{
    type Similarity = f64;
    #[inline]
    fn similarity(&self, left: &G::NodeId, right: &G::NodeId) -> Self::Similarity {
        let mut max_score = 0.0;
        for root_node in self.as_ref().root_nodes() {
            if let InformationContentSearch::Both(score) =
                information_content_search(self.as_ref(), left, right, root_node)
                && max_score < score
            {
                max_score = score;
            }
        }
        max_score
    }
}

/// Combine two `InformationContentSearch` states at a given parent node.
///
/// Returns the updated state for the parent after observing `child_result`
/// in one of its subtrees. `parent_node_ic` is the IC of the parent node,
/// used when the parent becomes the lowest common ancestor.
fn combine_ic_states(
    parent_state: InformationContentSearch,
    child_result: InformationContentSearch,
    parent_node_ic: f64,
) -> InformationContentSearch {
    match (parent_state, child_result) {
        (InformationContentSearch::Left, InformationContentSearch::Right)
        | (InformationContentSearch::Right, InformationContentSearch::Left) => {
            InformationContentSearch::Both(parent_node_ic)
        }
        (InformationContentSearch::NotFound, other) => other,
        (InformationContentSearch::Both(a), InformationContentSearch::Both(b)) => {
            InformationContentSearch::Both(if a > b { a } else { b })
        }
        (_, InformationContentSearch::Both(other)) => InformationContentSearch::Both(other),
        // All remaining cases are no-ops: Left+Left, Right+Right,
        // Both+Left, Both+Right, Left/Right/Both + NotFound.
        (state, _) => state,
    }
}

/// DFS stack frame used by [`information_content_search`].
struct IcSearchFrame<NodeId: Copy> {
    node: NodeId,
    successors: Vec<NodeId>,
    idx: usize,
    state: InformationContentSearch,
}

/// Iterative post-order DFS with memoisation to compute the
/// `InformationContentSearch` result for the subtree rooted at `root_node`.
///
/// # Complexity
///
/// O(V + E) time and O(V) space per query pair after the iterative rewrite
/// with a per-call memo table indexed by node id.
fn information_content_search<G>(
    information_content_result: &InformationContentResult<'_, G>,
    left: &G::NodeId,
    right: &G::NodeId,
    root_node: &G::NodeId,
) -> InformationContentSearch
where
    G: MonoplexMonopartiteGraph + ?Sized,
{
    let n = information_content_result.graph().number_of_nodes().as_();
    // Memo table: once a subtree result is known, store it here to avoid
    // reprocessing diamond-shaped subgraphs.
    let mut memo: Vec<Option<InformationContentSearch>> = vec![None; n];

    let graph = information_content_result.graph();
    let root = *root_node;
    let root_state = if root == *left {
        InformationContentSearch::Left
    } else if root == *right {
        InformationContentSearch::Right
    } else {
        InformationContentSearch::NotFound
    };
    let root_successors: Vec<G::NodeId> = graph.successors(root).collect();
    let mut stack =
        vec![IcSearchFrame { node: root, successors: root_successors, idx: 0, state: root_state }];

    loop {
        // Determine what to do next without holding a long-lived mutable
        // borrow on `stack`.
        let (should_push, should_pop) = {
            let frame = stack.last_mut().expect("stack is non-empty inside loop");
            if frame.idx < frame.successors.len() {
                let successor = frame.successors[frame.idx];
                frame.idx += 1;
                if let Some(child_result) = memo[successor.as_()] {
                    // Already memoised: fold directly into parent state.
                    let parent_ic = information_content_result[frame.node];
                    frame.state = combine_ic_states(frame.state, child_result, parent_ic);
                    (None, false)
                } else {
                    (Some(successor), false)
                }
            } else {
                (None, true)
            }
        };

        if let Some(successor) = should_push {
            let child_state = if successor == *left {
                InformationContentSearch::Left
            } else if successor == *right {
                InformationContentSearch::Right
            } else {
                InformationContentSearch::NotFound
            };
            let child_successors: Vec<G::NodeId> = graph.successors(successor).collect();
            stack.push(IcSearchFrame {
                node: successor,
                successors: child_successors,
                idx: 0,
                state: child_state,
            });
        } else if should_pop {
            let frame = stack.pop().expect("stack is non-empty when popping");
            let node_state = frame.state;
            memo[frame.node.as_()] = Some(node_state);
            if let Some(parent) = stack.last_mut() {
                let parent_ic = information_content_result[parent.node];
                parent.state = combine_ic_states(parent.state, node_state, parent_ic);
            } else {
                // Root frame popped: this is the final result.
                return node_state;
            }
        }
        // else: memo hit was applied in-place; no push/pop needed.
    }
}

/// Enum for tracking IC Search possibilities
#[derive(Clone, Copy)]
enum InformationContentSearch {
    NotFound,
    Left,
    Right,
    Both(f64),
}

/// Trait providing the `Resnik` method
///
/// # Reference
/// The implementation of the Resnik score as described in [Using Information Content to Evaluate Semantic Similarity in a Taxonomy](https://arxiv.org/pdf/cmp-lg/9511007)
pub trait Resnik: MonoplexMonopartiteGraph {
    /// The method for applying the Resnik
    ///
    /// # Arguments
    /// - `occurrences`: the number of times each node has been observed
    ///
    /// # Errors
    /// - If the graph is not a dag
    /// - If the occurrences are not equal
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
    /// let resnik = graph.resnik(&[1, 1, 1]).unwrap();
    /// assert!(resnik.similarity(&0, &0) >= 0.0);
    /// ```
    #[inline]
    fn resnik(
        &self,
        occurrences: &[usize],
    ) -> Result<ResnikResult<'_, Self>, InformationContentError> {
        // Compute IC using information content method
        Ok(self.information_content(occurrences)?.into())
    }
}

impl<G> Resnik for G where G: MonoplexMonopartiteGraph {}
