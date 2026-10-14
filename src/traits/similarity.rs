//! Trait for calculating similarity between two items.

/// Trait for calculating similarity between two items.
pub trait ScalarSimilarity<L, R> {
    /// The type of the similarity score.
    type Similarity;

    /// Calculate the similarity between two items.
    fn similarity(&self, left: &L, right: &R) -> Self::Similarity;
}
