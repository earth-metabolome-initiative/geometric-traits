#![cfg(feature = "alloc")]
//! Errors raised in algorithms defined for [`BipartiteGraph`]s.

use crate::traits::LAPJVError;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Errors that may occur when executing algorithms on a [`BipartiteGraph`].
pub enum BipartiteAlgorithmError {
    /// Error raised while executing the `LAPMOD` algorithm.
    #[error("{0}")]
    LAPMOD(LAPJVError),
}
