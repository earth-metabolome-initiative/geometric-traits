use core::fmt::Debug;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Error type for sorted data structures.
pub enum SortedError<V> {
    /// The entry is not sorted.
    #[error("Unsorted entry: {0:?}")]
    UnsortedEntry(V),
}
