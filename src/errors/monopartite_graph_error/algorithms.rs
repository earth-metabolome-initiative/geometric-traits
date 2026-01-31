#![cfg(feature = "alloc")]
//! Errors raised in algorithms defined for [`MonopartiteGraph`]s.

use crate::traits::connected_components::ConnectedComponentsError;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Errors that may occur when executing algorithms on a [`MonopartiteGraph`].
pub enum MonopartiteAlgorithmError {
    /// Error raised while computing connected components.
    #[error("{0}")]
    ConnectedComponentsError(ConnectedComponentsError),
}
