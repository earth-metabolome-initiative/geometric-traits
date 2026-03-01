//! Submodule providing a padded matrix, which fills all of the values not
//! defined in the underlying sparse matrix with the value provided by the Map
//! function.
use alloc::{string::String, vec::Vec};
use core::fmt::Debug;

use num_traits::{AsPrimitive, Bounded};

use super::{MutabilityError, square_padding_utils::validate_padded_square_capacity};
use crate::traits::{Matrix2D, SparseMatrix2D, SparseValuedMatrix2D, ValuedMatrix};

mod imputed_row_values;
mod matrix;
pub mod padded_coordinates;
mod sparse_matrix;
mod valued_matrix;

/// A padded matrix that fills all of the values not defined in the
/// underlying sparse matrix with the value provided by the Map function.
pub struct PaddedMatrix2D<M, Map> {
    /// The underlying sparse matrix.
    matrix: M,
    /// The function to map the values not defined in the underlying sparse
    /// matrix.
    map: Map,
}

impl<M, Map> Debug for PaddedMatrix2D<M, Map>
where
    M: SparseMatrix2D,
    Self: SparseValuedMatrix2D + Matrix2D<RowIndex = M::RowIndex, ColumnIndex = M::ColumnIndex>,
    <Self as ValuedMatrix>::Value: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rows: Vec<Vec<String>> = self
            .row_indices()
            .map(|row_index| {
                self.sparse_row(row_index)
                    .zip(self.sparse_row_values(row_index))
                    .map(|(column_index, value)| {
                        if self.is_imputed((row_index, column_index)) {
                            format!("I({value:?})")
                        } else {
                            format!("{value:?}")
                        }
                    })
                    .collect()
            })
            .collect();

        rows.fmt(f)
    }
}

impl<M: SparseMatrix2D, Map> PaddedMatrix2D<M, Map>
where
    M: SparseMatrix2D,
    M::RowIndex: AsPrimitive<usize> + Bounded,
    M::ColumnIndex: AsPrimitive<usize> + Bounded,
{
    /// Creates a new padded matrix with the given underlying sparse matrix and
    /// map function.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The underlying sparse matrix.
    /// * `map` - The function to map the values not defined in the underlying
    ///   sparse matrix.
    ///
    /// # Errors
    ///
    /// * `MutabilityError::MaxedOutColumnIndex` - The number of columns in the
    ///   matrix exceeds the maximum column index.
    /// * `MutabilityError::MaxedOutRowIndex` - The number of rows in the matrix
    ///   exceeds the maximum row index.
    #[inline]
    pub fn new(matrix: M, map: Map) -> Result<Self, MutabilityError<M>> {
        validate_padded_square_capacity(&matrix)?;

        Ok(Self { matrix, map })
    }

    #[inline]
    /// Returns whether the value at the provided coordinates is imputed or
    /// not.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The coordinates to check.
    pub fn is_imputed(&self, (row_index, column_index): (M::RowIndex, M::ColumnIndex)) -> bool {
        if row_index >= self.matrix.number_of_rows()
            || column_index >= self.matrix.number_of_columns()
        {
            return true;
        }

        self.matrix.sparse_row(row_index).all(|column| column != column_index)
    }
}
