//! Errors raised in algorithms defined for
//! [`crate::traits::MonopartiteGraph`]s.

use crate::traits::{
    BiconnectedComponentsError, MinimumCycleBasisError, ModularityError,
    connected_components::ConnectedComponentsError,
};

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Errors that may occur when executing algorithms on a
/// [`crate::traits::MonopartiteGraph`].
pub enum MonopartiteAlgorithmError {
    /// Error raised while computing biconnected components.
    #[error("{0}")]
    BiconnectedComponentsError(BiconnectedComponentsError),
    /// Error raised while computing connected components.
    #[error("{0}")]
    ConnectedComponentsError(ConnectedComponentsError),
    /// Error raised while computing a minimum cycle basis.
    #[error("{0}")]
    MinimumCycleBasisError(MinimumCycleBasisError),
    /// Error raised while computing modularity-based communities.
    #[error("{0}")]
    ModularityError(ModularityError),
}
