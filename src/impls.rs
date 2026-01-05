//! Implementations of traits for standard library types.

#[cfg(feature = "std")]
mod hashmap;

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod vec;

pub mod sorted_vec;
pub use sorted_vec::SortedVec;

mod array;
mod implicit_numeric_vocabularies;
mod slice;
mod sorted_array;
mod tuple;

#[cfg(any(feature = "std", feature = "alloc"))]
mod ragged_vec;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod ranged_csr;

#[cfg(any(feature = "std", feature = "alloc"))]
mod csr2d_edges;
#[cfg(any(feature = "std", feature = "alloc"))]
mod generic_bimatrix;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod generic_implicit_valued_matrix2d;
#[cfg(any(feature = "std", feature = "alloc"))]
mod squared_csr2d;
#[cfg(any(feature = "std", feature = "alloc"))]
mod symmetric_csr;
#[cfg(any(feature = "std", feature = "alloc"))]
mod upper_triangular_csr;
#[cfg(any(feature = "std", feature = "alloc"))]
mod valued_csr2d;

pub mod csr;
pub use csr::*;
pub mod error;
pub use error::*;
pub mod generic_bimatrix2d;
pub mod generic_iterators;
pub mod generic_matrix2d_with_padded_diagonal;
pub mod lower_bounded_sparse_square_matrix;
pub mod padded_matrix2d;
pub mod ragged_vector;
pub mod subset_sparse_square_matrix;
pub mod valued_matrix;
mod vector;

pub use generic_bimatrix2d::GenericBiMatrix2D;
pub use generic_implicit_valued_matrix2d::GenericImplicitValuedMatrix2D;
pub use generic_iterators::*;
pub use generic_matrix2d_with_padded_diagonal::GenericMatrix2DWithPaddedDiagonal;
pub use lower_bounded_sparse_square_matrix::LowerBoundedSquareMatrix;
pub use padded_matrix2d::*;
pub use ragged_vector::RaggedVector;
pub use ranged_csr::*;
pub use subset_sparse_square_matrix::SubsetSquareMatrix;
pub use valued_matrix::*;
pub use vec::VecMatrix2D;
