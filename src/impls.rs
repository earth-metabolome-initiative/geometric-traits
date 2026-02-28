//! Implementations of traits for standard library types.

#[cfg(feature = "std")]
mod hashmap;

#[cfg(feature = "alloc")]
pub mod vec;

#[cfg(feature = "alloc")]
pub mod sorted_vec;
#[cfg(feature = "alloc")]
pub use sorted_vec::SortedVec;

mod array;
mod implicit_numeric_vocabularies;
mod slice;
mod sorted_array;
mod tuple;

#[cfg(feature = "alloc")]
mod ragged_vec;
#[cfg(feature = "alloc")]
pub mod ranged_csr;

#[cfg(feature = "alloc")]
mod csr2d_edges;
#[cfg(feature = "alloc")]
mod generic_bimatrix;
#[cfg(feature = "alloc")]
pub mod generic_implicit_valued_matrix2d;
#[cfg(feature = "alloc")]
mod squared_csr2d;
#[cfg(feature = "alloc")]
mod symmetric_csr;
#[cfg(feature = "alloc")]
mod upper_triangular_csr;
#[cfg(feature = "alloc")]
mod valued_csr2d;

pub mod csr;
pub use csr::*;
pub mod error;
pub use error::*;
#[cfg(feature = "alloc")]
pub mod generic_bimatrix2d;
pub mod generic_iterators;
#[cfg(feature = "alloc")]
pub mod generic_matrix2d_with_padded_diagonal;
#[cfg(feature = "alloc")]
pub mod lower_bounded_sparse_square_matrix;
#[cfg(feature = "alloc")]
pub mod compact_matrix;
#[cfg(feature = "alloc")]
pub mod padded_matrix2d;
#[cfg(feature = "alloc")]
pub mod ragged_vector;
#[cfg(feature = "alloc")]
pub mod subset_sparse_square_matrix;
#[cfg(feature = "alloc")]
pub mod valued_matrix;
mod vector;

#[cfg(feature = "alloc")]
pub use compact_matrix::{CompactMatrix, compactify};
#[cfg(feature = "alloc")]
pub use generic_bimatrix2d::GenericBiMatrix2D;
#[cfg(feature = "alloc")]
pub use generic_implicit_valued_matrix2d::GenericImplicitValuedMatrix2D;
pub use generic_iterators::*;
#[cfg(feature = "alloc")]
pub use generic_matrix2d_with_padded_diagonal::GenericMatrix2DWithPaddedDiagonal;
#[cfg(feature = "alloc")]
pub use lower_bounded_sparse_square_matrix::LowerBoundedSquareMatrix;
#[cfg(feature = "alloc")]
pub use padded_matrix2d::*;
#[cfg(feature = "alloc")]
pub use ragged_vector::RaggedVector;
#[cfg(feature = "alloc")]
pub use ranged_csr::*;
#[cfg(feature = "alloc")]
pub use subset_sparse_square_matrix::SubsetSquareMatrix;
#[cfg(feature = "alloc")]
pub use valued_matrix::*;
#[cfg(feature = "alloc")]
pub use vec::VecMatrix2D;
