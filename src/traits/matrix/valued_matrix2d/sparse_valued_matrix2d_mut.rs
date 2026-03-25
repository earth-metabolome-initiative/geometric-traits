//! Submodule providing traits for mutable access to sparse valued matrix
//! values.

use crate::traits::{SparseValuedMatrix2DRef, SparseValuedMatrixMut, ValuedMatrix};

/// Trait defining mutable access to values in a sparse bi-dimensional matrix.
///
/// This trait allows modifying stored values in-place without changing the
/// sparse structure (row offsets and column indices) of the matrix.
pub trait SparseValuedMatrix2DMut:
    SparseValuedMatrix2DRef + SparseValuedMatrixMut
{
    /// Iterator over mutable references to the values of a row.
    type SparseRowValuesMut<'a>: Iterator<Item = &'a mut <Self as ValuedMatrix>::Value>
        + DoubleEndedIterator<Item = &'a mut <Self as ValuedMatrix>::Value>
    where
        Self: 'a,
        <Self as ValuedMatrix>::Value: 'a;

    /// Returns an iterator over mutable references to the values of a row.
    ///
    /// # Arguments
    ///
    /// * `row`: The row.
    fn sparse_row_values_mut(&mut self, row: Self::RowIndex) -> Self::SparseRowValuesMut<'_>;

    /// Returns a mutable reference to the value at the given row and column,
    /// if present.
    ///
    /// # Arguments
    ///
    /// * `row`: The row.
    /// * `column`: The column.
    fn sparse_value_at_mut(
        &mut self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
    ) -> Option<&mut <Self as ValuedMatrix>::Value>
    where
        Self::ColumnIndex: PartialEq;

    /// Replaces the value at the given row and column, returning the old
    /// value, or `None` if the entry does not exist.
    #[inline]
    fn replace_value(
        &mut self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
        value: <Self as ValuedMatrix>::Value,
    ) -> Option<<Self as ValuedMatrix>::Value>
    where
        Self::ColumnIndex: PartialEq,
    {
        self.sparse_value_at_mut(row, column)
            .map(|v| core::mem::replace(v, value))
    }

    /// Applies a function to the value at the given row and column.
    /// Returns `true` if the entry existed and the function was applied.
    #[inline]
    fn update_value(
        &mut self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
        f: impl FnOnce(&mut <Self as ValuedMatrix>::Value),
    ) -> bool
    where
        Self::ColumnIndex: PartialEq,
    {
        if let Some(v) = self.sparse_value_at_mut(row, column) {
            f(v);
            true
        } else {
            false
        }
    }

    /// Returns an iterator over the entries (column index and mutable value
    /// reference) of a row.
    ///
    /// This is a required method because the default implementation would
    /// create conflicting borrows between `sparse_row` and
    /// `sparse_row_values_mut`. Concrete types implement this via split
    /// borrows.
    fn sparse_row_entries_mut(
        &mut self,
        row: Self::RowIndex,
    ) -> impl Iterator<Item = (Self::ColumnIndex, &mut <Self as ValuedMatrix>::Value)>;
}
