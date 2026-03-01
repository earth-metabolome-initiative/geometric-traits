//! Shared helpers for square-padding wrappers.
#![cfg(feature = "alloc")]

use num_traits::{AsPrimitive, Bounded};

use super::MutabilityError;
use crate::traits::Matrix2D;

#[inline]
pub(crate) fn padded_square_size<M>(matrix: &M) -> usize
where
    M: Matrix2D,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
    let number_of_columns: usize = matrix.number_of_columns().as_();
    let number_of_rows: usize = matrix.number_of_rows().as_();
    number_of_columns.max(number_of_rows)
}

#[inline]
pub(crate) fn validate_padded_square_capacity<M>(matrix: &M) -> Result<(), MutabilityError<M>>
where
    M: Matrix2D,
    M::RowIndex: AsPrimitive<usize> + Bounded,
    M::ColumnIndex: AsPrimitive<usize> + Bounded,
{
    let number_of_columns: usize = matrix.number_of_columns().as_();
    let number_of_rows: usize = matrix.number_of_rows().as_();

    if number_of_columns > M::RowIndex::max_value().as_() {
        return Err(MutabilityError::<M>::MaxedOutColumnIndex);
    }
    if number_of_rows > M::ColumnIndex::max_value().as_() {
        return Err(MutabilityError::<M>::MaxedOutRowIndex);
    }

    Ok(())
}
