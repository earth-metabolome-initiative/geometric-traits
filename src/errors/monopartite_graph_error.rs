//! Errors that may occur when working with monopartite graphs.

use super::nodes::NodeError;
use crate::traits::MonopartiteGraph;

#[cfg(feature = "alloc")]
pub mod algorithms;
#[cfg(feature = "alloc")]
pub use algorithms::MonopartiteAlgorithmError;
pub mod illegal_graph_states;
pub use illegal_graph_states::IllegalMonopartiteGraphState;

/// Errors that may occur when working with graphs.
#[derive(Debug, thiserror::Error)]
pub enum MonopartiteError<G: MonopartiteGraph + ?Sized> {
    /// Error relative to graphs.
    #[error(transparent)]
    IllegalGraphState(#[from] IllegalMonopartiteGraphState<G>),
    /// Error relative to nodes.
    #[error(transparent)]
    NodeError(NodeError<G::Nodes>),
    /// Error relative to algorithms.
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    AlgorithmError(#[from] MonopartiteAlgorithmError),
}
