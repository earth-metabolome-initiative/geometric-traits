//! Submodule implementing the `Matrix` trait for the `PaddedMatrix` struct.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::PaddedMatrix2D;
use crate::impls::square_padding_utils::padded_square_size;
use crate::traits::{Matrix, Matrix2D, TryFromUsize};

impl<M, Map> Matrix for PaddedMatrix2D<M, Map>
where
    M: Matrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type Coordinates = M::Coordinates;

    #[inline]
    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows().as_(), self.number_of_columns().as_()]
    }
}

impl<M, Map> Matrix2D for PaddedMatrix2D<M, Map>
where
    M: Matrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type RowIndex = M::RowIndex;
    type ColumnIndex = M::ColumnIndex;

    #[inline]
    fn number_of_columns(&self) -> Self::ColumnIndex {
        let max = padded_square_size(&self.matrix);
        let Ok(number_of_columns) = Self::ColumnIndex::try_from_usize(max) else {
            panic!("The number of columns {max} is too large to be represented as a ColumnIndex")
        };
        number_of_columns
    }

    #[inline]
    fn number_of_rows(&self) -> Self::RowIndex {
        let max = padded_square_size(&self.matrix);
        let Ok(number_of_rows) = Self::RowIndex::try_from_usize(max) else {
            panic!("The number of rows {max} is too large to be represented as a RowIndex")
        };
        number_of_rows
    }
}
