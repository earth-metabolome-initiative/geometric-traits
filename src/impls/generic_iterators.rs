//! Submodule providing the implementation of iterators to be used with the
//! algebraic structures.

pub mod implicit_valued_sparse_iterator;
pub use implicit_valued_sparse_iterator::ImplicitValuedSparseIterator;
/// Submodule providing the implementation of the intersection iterator.
pub mod intersection;
pub use intersection::{Intersection, SortedIterator};
