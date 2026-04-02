//! Generic per-edge context memberships.
//!
//! This helper stores zero or more context signatures for each edge using a
//! CSR-style layout. It is intended for precomputed constraints such as
//! cycle/ring memberships, but it is not specific to chemistry.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use num_traits::AsPrimitive;

use crate::traits::PositiveInteger;

/// CSR-style per-edge context memberships.
///
/// Each row corresponds to one original graph edge and contains zero or more
/// context signatures attached to that edge.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeContexts<Signature, SparseIndex = usize> {
    offsets: Vec<SparseIndex>,
    signatures: Vec<Signature>,
}

impl<Signature, SparseIndex> Default for EdgeContexts<Signature, SparseIndex>
where
    SparseIndex: PositiveInteger,
{
    #[inline]
    fn default() -> Self {
        Self { offsets: vec![SparseIndex::ZERO], signatures: Vec::new() }
    }
}

impl<Signature, SparseIndex> EdgeContexts<Signature, SparseIndex> {
    /// Builds edge contexts from grouped per-edge rows.
    ///
    /// Each input row becomes the context list for one edge.
    pub fn from_rows<Rows, Row>(rows: Rows) -> Self
    where
        Rows: IntoIterator<Item = Row>,
        Row: IntoIterator<Item = Signature>,
        SparseIndex: PositiveInteger,
        <SparseIndex as TryFrom<usize>>::Error: Debug,
    {
        let mut contexts = Self::default();
        for row in rows {
            contexts.signatures.extend(row);
            contexts.offsets.push(
                SparseIndex::try_from_usize(contexts.signatures.len())
                    .expect("edge context signature count must fit into SparseIndex"),
            );
        }
        contexts
    }

    /// Returns the number of rows / edges stored in this helper.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns `true` when there are no edge rows.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the context slice for the given edge index.
    #[inline]
    #[must_use]
    pub fn contexts_of(&self, edge_index: usize) -> &[Signature]
    where
        SparseIndex: AsPrimitive<usize>,
    {
        assert!(
            edge_index < self.len(),
            "edge context row index {edge_index} out of bounds for {} rows",
            self.len()
        );
        let start = self.offsets[edge_index].as_();
        let end = self.offsets[edge_index + 1].as_();
        &self.signatures[start..end]
    }

    /// Returns `true` if two edge rows are compatible under the default
    /// completion semantics.
    ///
    /// Compatibility rules:
    /// - both rows empty => compatible
    /// - exactly one row empty => incompatible
    /// - both non-empty => compatible iff the two rows share at least one
    ///   signature
    #[must_use]
    pub fn compatible_with(&self, edge_index: usize, other: &Self, other_edge_index: usize) -> bool
    where
        Signature: PartialEq,
        SparseIndex: AsPrimitive<usize>,
    {
        let left = self.contexts_of(edge_index);
        let right = other.contexts_of(other_edge_index);

        if left.is_empty() || right.is_empty() {
            return left.is_empty() && right.is_empty();
        }

        left.iter().any(|signature| right.contains(signature))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::EdgeContexts;

    #[test]
    fn test_edge_contexts_rows_are_sliced_correctly() {
        let contexts = EdgeContexts::<u8>::from_rows(vec![vec![1, 2], Vec::new(), vec![3]]);

        assert_eq!(contexts.len(), 3);
        assert_eq!(contexts.contexts_of(0), &[1, 2]);
        assert_eq!(contexts.contexts_of(1), &[] as &[u8]);
        assert_eq!(contexts.contexts_of(2), &[3]);
    }

    #[test]
    fn test_edge_contexts_compatibility_rules() {
        let both_empty = EdgeContexts::<u8>::from_rows(vec![Vec::new()]);
        let left_only = EdgeContexts::<u8>::from_rows(vec![vec![1]]);
        let right_only = EdgeContexts::<u8>::from_rows(vec![vec![2]]);
        let overlap = EdgeContexts::<u8>::from_rows(vec![vec![1, 3]]);

        assert!(both_empty.compatible_with(0, &both_empty, 0));
        assert!(!left_only.compatible_with(0, &both_empty, 0));
        assert!(!both_empty.compatible_with(0, &left_only, 0));
        assert!(!left_only.compatible_with(0, &right_only, 0));
        assert!(left_only.compatible_with(0, &overlap, 0));
    }
}
