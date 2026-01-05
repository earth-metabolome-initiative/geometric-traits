use core::fmt::Debug;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error type for sorted data structures.
pub enum SortedError<V> {
    /// The entry is not sorted.
    UnsortedEntry(V),
}

impl<V: Debug> core::fmt::Display for SortedError<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SortedError::UnsortedEntry(v) => write!(f, "Unsorted entry: {v:?}"),
        }
    }
}

impl<V: Debug> core::error::Error for SortedError<V> {}
