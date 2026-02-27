//! Errors raised in algorithms defined for
//! [`crate::traits::MonopartiteGraph`]s.

use crate::traits::{LouvainError, connected_components::ConnectedComponentsError};

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Errors that may occur when executing algorithms on a
/// [`crate::traits::MonopartiteGraph`].
pub enum MonopartiteAlgorithmError {
    /// Error raised while computing connected components.
    #[error("{0}")]
    ConnectedComponentsError(ConnectedComponentsError),
    /// Error raised while computing Louvain communities.
    #[error("{0}")]
    LouvainError(LouvainError),
}
