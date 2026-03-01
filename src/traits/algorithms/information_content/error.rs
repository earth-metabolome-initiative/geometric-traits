//! Submodule providing `Information Content` Errors for working with IC based
//! Algorithms
use crate::traits::KahnError;

/// Information Content Enum for Errors that may occur during IC calculation
/// process
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum InformationContentError {
    /// Error for when a graph is not a DAG / contains a cycle
    #[error("The graph is not a DAG")]
    NotDag,
    /// Error for unexpected occurrence size
    #[error("Received an occurrence vector with {found} entries but expected {expected} entries")]
    UnequalOccurrenceSize {
        /// The expected size for the occurrence
        expected: usize,
        /// The actual size found for the occurrence
        found: usize,
    },
    /// Sink Node found with 0 occurrence count
    #[error("Found a Sink Node with a 0 Occurrence")]
    SinkNodeZeroOccurrence,
}

impl From<KahnError> for InformationContentError {
    #[inline]
    fn from(value: KahnError) -> Self {
        match value {
            KahnError::Cycle => Self::NotDag,
        }
    }
}
