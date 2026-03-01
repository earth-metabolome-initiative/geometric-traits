//! Submodule defining a Transposed Monoplex Graph.
//!
//! A transposed monoplex graph is a graph where the edges are of a single type
//! and it is possible to efficiently access the predecessors of a node.

use super::TransposedEdges;
use crate::traits::{
    MonopartiteEdges, MonopartiteGraph, MonoplexMonopartiteGraph, TransposedMonoplexGraph,
};

/// Trait defining a transposed monoplex monopartite graph.
///
/// This trait combines `TransposedMonoplexGraph` and
/// `MonoplexMonopartiteGraph`, requiring that the edges satisfy both
/// `TransposedEdges` and `MonopartiteEdges` with unified matrix types.
pub trait TransposedMonoplexMonopartiteGraph:
    TransposedMonoplexGraph<TransposedEdges = Self::TransposedMonoplexMonopartiteEdges>
    + MonoplexMonopartiteGraph<MonoplexMonopartiteEdges = Self::TransposedMonoplexMonopartiteEdges>
{
    /// The type of edges in the transposed monoplex monopartite graph.
    type TransposedMonoplexMonopartiteEdges: TransposedEdges
        + MonopartiteEdges<
            NodeId = <Self as MonopartiteGraph>::NodeId,
            MonopartiteMatrix = <Self::TransposedMonoplexMonopartiteEdges as TransposedEdges>::BiMatrix,
        >;

    /// Returns whether the provided node is a singleton, i.e., it has no
    /// incoming or outgoing edges.
    ///
    /// # Arguments
    ///
    /// * `node` - The identifier of the node to check.
    ///
    /// # Returns
    ///
    /// `true` if the node has no incoming or outgoing edges, `false` otherwise.
    #[inline]
    fn is_singleton(&self, node: Self::NodeId) -> bool {
        !self.has_successors(node) && !self.has_predecessors(node)
    }
}

impl<G> TransposedMonoplexMonopartiteGraph for G
where
    G: TransposedMonoplexGraph<TransposedEdges = G::MonoplexMonopartiteEdges>
        + MonoplexMonopartiteGraph,
    G::MonoplexMonopartiteEdges: TransposedEdges<
        BiMatrix = <G::MonoplexMonopartiteEdges as MonopartiteEdges>::MonopartiteMatrix,
    >,
{
    type TransposedMonoplexMonopartiteEdges = G::MonoplexMonopartiteEdges;
}
