//! Submodule providing a 2D matrix which pads the missing elements over the
//! diagonal, meaning it will not change the values of existing elements but
//! will add new elements where missing. If the underlying matrix is
//! rectangular, new rows and columns will be added to make it square.
use alloc::vec::Vec;

use num_traits::{AsPrimitive, Bounded, One, Zero};

use crate::traits::{
    EmptyRows, Matrix, Matrix2D, SparseMatrix, SparseMatrix2D, SparseValuedMatrix,
    SparseValuedMatrix2D, TryFromUsize, ValuedMatrix, ValuedMatrix2D,
};
mod sparse_row_with_padded_diagonal;
use sparse_row_with_padded_diagonal::SparseRowWithPaddedDiagonal;
mod sparse_rows_with_padded_diagonal;
use sparse_rows_with_padded_diagonal::SparseRowsWithPaddedDiagonal;
mod sparse_row_values_with_padded_diagonal;
use multi_ranged::{SimpleRange, Step};
use sparse_row_values_with_padded_diagonal::SparseRowValuesWithPaddedDiagonal;

use super::{
    CSR2DColumns, CSR2DView, M2DValues, MutabilityError,
    square_padding_utils::{padded_square_size, validate_padded_square_capacity},
};

#[cfg(feature = "arbitrary")]
mod arbitrary_impl;

/// A 2D matrix which pads the missing elements over the diagonal.
pub struct GenericMatrix2DWithPaddedDiagonal<M, Map> {
    /// The underlying matrix.
    matrix: M,
    /// The map function defining the values of the new elements.
    map: Map,
}

impl<M, Map> GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M::RowIndex: AsPrimitive<usize> + Bounded,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize + Bounded,
    M: SparseMatrix2D,
{
    /// Creates a new `GenericMatrix2DWithPaddedDiagonal` with the given matrix
    /// and map function.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The underlying matrix.
    /// * `map` - The map function defining the values of the new elements.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of rows or columns exceeds the maximum
    ///   allowed size for the given row and column index types.
    #[inline]
    pub fn new(matrix: M, map: Map) -> Result<Self, MutabilityError<M>> {
        validate_padded_square_capacity(&matrix)?;

        Ok(Self { matrix, map })
    }

    /// Returns a reference to the underlying matrix.
    #[inline]
    pub fn matrix(&self) -> &M {
        &self.matrix
    }

    /// Returns whether the diagonal of the provided row is imputed or not.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the sparse row.
    ///
    /// # Panics
    ///
    /// * If the row index is out of bounds.
    #[inline]
    pub fn is_diagonal_imputed(&self, row: M::RowIndex) -> bool {
        if row >= self.matrix.number_of_rows() {
            return true;
        }

        let row_as_column = M::ColumnIndex::try_from_usize(row.as_())
            .map_err(|_| MutabilityError::<M>::MaxedOutColumnIndex)
            .unwrap();

        self.matrix.sparse_row(row).all(|column| column != row_as_column)
    }
}

impl<M, Map> Matrix for GenericMatrix2DWithPaddedDiagonal<M, Map>
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

impl<M, Map> Matrix2D for GenericMatrix2DWithPaddedDiagonal<M, Map>
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

impl<M, Map> SparseMatrix for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: SparseMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type SparseIndex = M::SparseIndex;
    type SparseCoordinates<'a>
        = CSR2DView<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        CSR2DView::from(self)
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        // Since the diagonal is padded, the last coordinates are the last
        // row and column of the matrix, unless the matrix is empty.
        if self.is_empty() {
            return None;
        }
        Some((
            self.number_of_rows() - M::RowIndex::one(),
            self.number_of_columns() - M::ColumnIndex::one(),
        ))
    }

    #[inline]
    fn is_empty(&self) -> bool {
        // The matrix is solely empty when it has no rows and no columns.
        self.number_of_rows() == M::RowIndex::zero()
            && self.number_of_columns() == M::ColumnIndex::zero()
    }
}

impl<M, Map> SparseMatrix2D for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: SparseMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type SparseRow<'a>
        = SparseRowWithPaddedDiagonal<'a, M>
    where
        Self: 'a;
    type SparseColumns<'a>
        = CSR2DColumns<'a, Self>
    where
        Self: 'a;

    type SparseRows<'a>
        = SparseRowsWithPaddedDiagonal<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        SparseRowWithPaddedDiagonal::new(&self.matrix, row).unwrap()
    }

    #[inline]
    fn has_entry(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> bool {
        self.sparse_row(row).any(|col| col == column)
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        SparseRowsWithPaddedDiagonal::from(self)
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        CSR2DColumns::from(self)
    }
}

impl<M, Map> EmptyRows for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: EmptyRows,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize + Step,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type EmptyRowIndices<'a>
        = core::iter::Empty<Self::RowIndex>
    where
        Self: 'a;
    type NonEmptyRowIndices<'a>
        = SimpleRange<Self::RowIndex>
    where
        Self: 'a;
    #[inline]
    fn empty_row_indices(&self) -> Self::EmptyRowIndices<'_> {
        // Since we are artificially always adding rows and columns, we
        // will never have empty rows.
        core::iter::empty()
    }

    #[inline]
    fn non_empty_row_indices(&self) -> Self::NonEmptyRowIndices<'_> {
        // Since we are artificially always adding rows and columns, we
        // will always have non-empty rows.
        SimpleRange::try_from((Self::RowIndex::zero(), self.number_of_rows())).unwrap()
    }

    #[inline]
    fn number_of_empty_rows(&self) -> Self::RowIndex {
        // Since we are artificially always adding rows and columns, we
        // will never have empty rows.
        Self::RowIndex::zero()
    }

    #[inline]
    fn number_of_non_empty_rows(&self) -> Self::RowIndex {
        // Since we are artificially always adding rows and columns, we
        // will always have non-empty rows.
        self.number_of_rows()
    }
}

impl<M, Map> ValuedMatrix for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: ValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    type Value = M::Value;
}

impl<M, Map> ValuedMatrix2D for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: ValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
}

impl<M, Map> SparseValuedMatrix for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: SparseValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
    Map: Fn(M::RowIndex) -> M::Value,
    M::Value: Clone,
{
    type SparseValues<'a>
        = M2DValues<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_values(&self) -> Self::SparseValues<'_> {
        M2DValues::from(self)
    }
}

impl<M, Map> SparseValuedMatrix2D for GenericMatrix2DWithPaddedDiagonal<M, Map>
where
    M: SparseValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
    Map: Fn(M::RowIndex) -> M::Value,
    M::Value: Clone,
{
    type SparseRowValues<'a>
        = SparseRowValuesWithPaddedDiagonal<'a, M, &'a Map>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        SparseRowValuesWithPaddedDiagonal::new(&self.matrix, row, &self.map).unwrap()
    }
}
