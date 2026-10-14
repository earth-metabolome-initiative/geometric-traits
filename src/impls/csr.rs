//! Submodule providing a definition of a CSR matrix.

#[cfg(feature = "alloc")]
pub mod csr2d;
#[cfg(feature = "alloc")]
pub use csr2d::CSR2D;
#[cfg(feature = "alloc")]
pub mod upper_triangular_csr2d;
#[cfg(feature = "alloc")]
pub use upper_triangular_csr2d::UpperTriangularCSR2D;
#[cfg(feature = "alloc")]
pub mod square_csr2d;
#[cfg(feature = "alloc")]
pub use square_csr2d::SquareCSR2D;
pub mod csr2d_view;
pub use csr2d_view::CSR2DView;
pub mod csr2d_sized_rows;
pub use csr2d_sized_rows::CSR2DSizedRows;
pub mod csr2d_rows;
pub use csr2d_rows::CSR2DRows;
pub mod csr2d_columns;
pub use csr2d_columns::CSR2DColumns;
#[cfg(feature = "alloc")]
pub mod symmetric_csr2d;
#[cfg(feature = "alloc")]
pub use symmetric_csr2d::SymmetricCSR2D;
pub mod csr2d_row_sizes;
pub use csr2d_row_sizes::CSR2DSizedRowsizes;
pub mod csr2d_empty_rows_indices;
pub use csr2d_empty_rows_indices::CSR2DEmptyRowIndices;
pub mod csr2d_non_empty_rows_indices;
pub use csr2d_non_empty_rows_indices::CSR2DNonEmptyRowIndices;
mod csr2d_values;
pub use csr2d_values::M2DValues;
#[cfg(all(feature = "arbitrary", feature = "alloc"))]
mod arbitrary_impl;
