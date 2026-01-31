//! Errors that may occur when working with bipartite graphs.

use super::nodes::NodeError;
use crate::traits::BipartiteGraph;

#[cfg(feature = "alloc")]
pub mod algorithms;
#[cfg(feature = "alloc")]
pub use algorithms::BipartiteAlgorithmError;
pub mod illegal_graph_states;
pub use illegal_graph_states::IllegalBipartiteGraphState;

/// Errors that may occur when working with graphs.
#[derive(Debug, thiserror::Error)]
pub enum BipartiteError<G: BipartiteGraph + ?Sized> {
    /// Error relative to graphs.
    #[error(transparent)]
    IllegalGraphState(#[from] IllegalBipartiteGraphState<G>),
    /// Error relative to left nodes partition.
    #[error(transparent)]
    LeftNodeError(NodeError<G::LeftNodes>),
    /// Error relative to right nodes partition.
    #[error(transparent)]
    RightNodeError(NodeError<G::RightNodes>),
    /// Error relative to algorithms.
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    AlgorithmError(#[from] BipartiteAlgorithmError),
}
