//! Submodule providing traits for reference access to sparse valued matrix
//! values.

use super::ValuedMatrix2D;
use crate::traits::{SparseValuedMatrix2D, SparseValuedMatrixRef, ValuedMatrix};

/// Trait defining reference access to values in a sparse bi-dimensional matrix.
///
/// Unlike [`SparseValuedMatrix2D`] which returns cloned values, this trait
/// provides direct references to the stored values, avoiding copies for
/// non-`Copy` types.
pub trait SparseValuedMatrix2DRef:
    SparseValuedMatrix2D + ValuedMatrix2D + SparseValuedMatrixRef
{
    /// Iterator over references to the values of a row.
    type SparseRowValuesRef<'a>: Iterator<Item = &'a <Self as ValuedMatrix>::Value>
        + DoubleEndedIterator<Item = &'a <Self as ValuedMatrix>::Value>
    where
        Self: 'a,
        <Self as ValuedMatrix>::Value: 'a;

    /// Returns an iterator over references to the values of a row.
    ///
    /// # Arguments
    ///
    /// * `row`: The row.
    fn sparse_row_values_ref(&self, row: Self::RowIndex) -> Self::SparseRowValuesRef<'_>;

    /// Returns a reference to the value at the given row and column, if
    /// present.
    #[inline]
    fn sparse_value_at_ref(
        &self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
    ) -> Option<&<Self as ValuedMatrix>::Value>
    where
        Self::ColumnIndex: PartialEq,
    {
        self.sparse_row(row)
            .zip(self.sparse_row_values_ref(row))
            .find(|&(col, _)| col == column)
            .map(|(_, val)| val)
    }

    /// Returns an iterator over the entries (column index and value reference)
    /// of a row.
    #[inline]
    fn sparse_row_entries(
        &self,
        row: Self::RowIndex,
    ) -> impl Iterator<Item = (Self::ColumnIndex, &<Self as ValuedMatrix>::Value)> {
        self.sparse_row(row).zip(self.sparse_row_values_ref(row))
    }
}

impl<M: SparseValuedMatrix2DRef> SparseValuedMatrix2DRef for &M {
    type SparseRowValuesRef<'a>
        = M::SparseRowValuesRef<'a>
    where
        Self: 'a,
        M::Value: 'a;

    #[inline]
    fn sparse_row_values_ref(&self, row: Self::RowIndex) -> Self::SparseRowValuesRef<'_> {
        (*self).sparse_row_values_ref(row)
    }

    #[inline]
    fn sparse_value_at_ref(
        &self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
    ) -> Option<&<Self as ValuedMatrix>::Value>
    where
        Self::ColumnIndex: PartialEq,
    {
        (*self).sparse_value_at_ref(row, column)
    }
}
