//! Errors raised in algorithms defined for
//! [`crate::traits::MonopartiteGraph`]s.

use crate::traits::{
    BiconnectedComponentsError, ModularityError, OuterplanarityError, PlanarityError,
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
    /// Error raised while testing planarity.
    #[error("{0}")]
    PlanarityError(PlanarityError),
    /// Error raised while testing outerplanarity.
    #[error("{0}")]
    OuterplanarityError(OuterplanarityError),
    /// Error raised while computing modularity-based communities.
    #[error("{0}")]
    ModularityError(ModularityError),
}
