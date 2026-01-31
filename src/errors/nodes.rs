//! Submodule defining common errors relative to nodes.

use crate::traits::Vocabulary;

/// Error enumeration relative to nodes.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum NodeError<V: Vocabulary> {
    /// The node does not exist.
    #[error("The node with id {0:?} does not exist.")]
    UnknownNodeId(V::SourceSymbol),
    /// The node symbol does not exist.
    #[error("The node with symbol {0:?} does not exist.")]
    UnknownNodeSymbol(V::DestinationSymbol),
}
