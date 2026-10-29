//! Implementation for the 'Wu Palmer' trait based on the algorithm
//! implementation
use alloc::vec::Vec;

use num_traits::AsPrimitive;

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
    /// Every node, each one listed after all of its successors
    reverse_topological_order: Vec<G::NodeId>,
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
    /// O(V + E).
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
    #[inline]
    fn wu_palmer(&self) -> Result<WuPalmerResult<'_, Self>, KahnError> {
        // `kahn` rejects cycles and maps each matrix row to its rank.
        let ranks = self.edges().matrix().kahn()?;
        let mut by_rank: Vec<Option<Self::NodeId>> = alloc::vec![None; ranks.len()];
        for node in self.node_ids() {
            by_rank[ranks[node.as_()].as_()] = Some(node);
        }
        let reverse_topological_order = by_rank.into_iter().rev().flatten().collect();
        Ok(WuPalmerResult { graph: self, root_nodes: self.root_nodes(), reverse_topological_order })
    }
}

/// Wu-Palmer depths of the subtree below one node, counted from that node at
/// depth zero.
#[derive(Clone, Copy, Default)]
struct SubtreeDepths {
    /// Deepest depth of `left`, when reachable
    left: Option<usize>,
    /// Deepest depth of `right`, when reachable
    right: Option<usize>,
    /// Depth of the deepest node reaching both `left` and `right`
    common: usize,
}

impl SubtreeDepths {
    /// Folds in the depths of a successor, which sits one level deeper.
    fn absorb(&mut self, successor: Self) {
        match (successor.left, successor.right) {
            // Keeping the deepest depth across paths keeps 2 * common <= left +
            // right.
            (Some(left), None) => self.left = Some(self.left.map_or(left + 1, |d| d.max(left + 1))),
            (None, Some(right)) => {
                self.right = Some(self.right.map_or(right + 1, |d| d.max(right + 1)));
            }
            (None, None) => {}
            (Some(left), Some(right)) => {
                if self.common < successor.common + 1 {
                    self.common = successor.common + 1;
                    self.left = Some(left + 1);
                    self.right = Some(right + 1);
                }
            }
        }
    }
}

impl<G> ScalarSimilarity<G::NodeId, G::NodeId> for WuPalmerResult<'_, G>
where
    G: MonoplexMonopartiteGraph,
{
    type Similarity = f64;
    #[inline]
    fn similarity(&self, left: &G::NodeId, right: &G::NodeId) -> Self::Similarity {
        if left == right {
            return 1.0;
        }
        // Depths are relative to each node, so one pass serves every root.
        let mut depths =
            alloc::vec![SubtreeDepths::default(); self.reverse_topological_order.len()];
        for &node in &self.reverse_topological_order {
            let mut node_depths = SubtreeDepths {
                left: (node == *left).then_some(0),
                right: (node == *right).then_some(0),
                common: 0,
            };
            for successor in self.graph.successors(node) {
                node_depths.absorb(depths[successor.as_()]);
            }
            depths[node.as_()] = node_depths;
        }

        let mut max_similarity = 0.0;
        for root_node in &self.root_nodes {
            let SubtreeDepths { left: Some(left), right: Some(right), common } =
                depths[root_node.as_()]
            else {
                continue;
            };
            // Roots sit at depth one, matching the canonical Wu-Palmer score
            // 2 * depth(LCS) / (depth(left) + depth(right)).
            let similarity = 2.0 * depth_as_f64(common + 1) / depth_as_f64(left + right + 2);
            if similarity > max_similarity {
                max_similarity = similarity;
            }
        }
        max_similarity
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a depth is bounded by the node count, which stays below 2^53"
)]
fn depth_as_f64(depth: usize) -> f64 {
    debug_assert!(
        u64::try_from(depth).is_ok_and(|depth| depth < 1 << f64::MANTISSA_DIGITS),
        "depth {depth} exceeds the exact f64 range"
    );
    depth as f64
}

impl<G> WuPalmer for G where G: MonoplexMonopartiteGraph {}
