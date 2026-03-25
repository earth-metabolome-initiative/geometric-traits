//! Submodule defining the [`ValuedMatrix2D`] trait.

use super::{Matrix2D, ValuedMatrix};

mod dense_valued_matrix2d;
mod sparse_valued_matrix2d;
mod sparse_valued_matrix2d_mut;
mod sparse_valued_matrix2d_ref;
pub use dense_valued_matrix2d::*;
pub use sparse_valued_matrix2d::*;
pub use sparse_valued_matrix2d_mut::*;
pub use sparse_valued_matrix2d_ref::*;

/// Trait defining a bi-dimensional matrix.
pub trait ValuedMatrix2D: Matrix2D + ValuedMatrix {}

impl<M: ValuedMatrix2D> ValuedMatrix2D for &M {}
