//! Submodule providing the `ValuedCsr2D` type, a 2D CSR matrix which stores
//! values in addition to the row and column indices.
#[cfg(feature = "mem_dbg")]
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

use multi_ranged::Step;
use num_traits::{AsPrimitive, One, Zero};

use super::{CSR2D, MutabilityError};
use crate::traits::{
    EmptyRows, Matrix, Matrix2D, Matrix2DRef, MatrixMut, PositiveInteger, RankSelectSparseMatrix,
    SizedRowsSparseMatrix2D, SizedSparseMatrix, SizedSparseMatrix2D, SizedSparseValuedMatrix,
    SizedSparseValuedMatrixMut, SizedSparseValuedMatrixRef, SparseMatrix, SparseMatrix2D,
    SparseMatrixMut, SparseValuedMatrix, SparseValuedMatrix2D, SparseValuedMatrix2DMut,
    SparseValuedMatrix2DRef, SparseValuedMatrixMut, SparseValuedMatrixRef, TryFromUsize,
    ValuedMatrix, ValuedMatrix2D,
};

#[cfg(feature = "arbitrary")]
mod arbitrary_impl;

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, PartialEq, Eq, Hash)]
/// A 2D CSR matrix which stores values in addition to the row and column
/// indices.
pub struct ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value> {
    csr: CSR2D<SparseIndex, RowIndex, ColumnIndex>,
    values: Vec<Value>,
}

/// Errors raised when constructing a [`ValuedCSR2D`] from pre-built parts.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ValuedCsrPartsError {
    /// The provided values vector does not match the CSR structure.
    #[error("Values length mismatch: expected {expected}, got {actual}")]
    ValuesLengthMismatch {
        /// Number of values required by the CSR structure.
        expected: usize,
        /// Number of values provided by the caller.
        actual: usize,
    },
}

impl<SparseIndex: AsPrimitive<usize>, RowIndex, ColumnIndex, Value>
    ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SizedSparseMatrix<SparseIndex = SparseIndex>,
{
    /// Creates a new valued CSR matrix from a CSR structure and its stored
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`ValuedCsrPartsError::ValuesLengthMismatch`] when the number of
    /// provided values does not match the number of sparse entries stored in
    /// the CSR structure.
    #[inline]
    pub fn from_parts(
        csr: CSR2D<SparseIndex, RowIndex, ColumnIndex>,
        values: Vec<Value>,
    ) -> Result<Self, ValuedCsrPartsError> {
        let expected = csr.number_of_defined_values().as_();
        let actual = values.len();

        if actual != expected {
            return Err(ValuedCsrPartsError::ValuesLengthMismatch { expected, actual });
        }

        Ok(Self { csr, values })
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value>
    ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
{
    /// Returns the underlying CSR structure and raw values storage.
    #[inline]
    pub fn into_parts(self) -> (CSR2D<SparseIndex, RowIndex, ColumnIndex>, Vec<Value>) {
        (self.csr, self.values)
    }

    /// Returns the raw values slice in CSR storage order.
    #[inline]
    pub fn values_ref(&self) -> &[Value] {
        &self.values
    }

    /// Returns the raw mutable values slice in CSR storage order.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [Value] {
        &mut self.values
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    Value,
> ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>:
        Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    /// Returns the values slice stored for a sparse row.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::prelude::*;
    ///
    /// let mut matrix: ValuedCSR2D<usize, usize, usize, i32> =
    ///     SparseMatrixMut::with_sparse_shape((3, 8));
    /// MatrixMut::add(&mut matrix, (1, 2, 20)).unwrap();
    /// MatrixMut::add(&mut matrix, (1, 4, 40)).unwrap();
    /// MatrixMut::add(&mut matrix, (2, 7, 70)).unwrap();
    ///
    /// assert_eq!(matrix.sparse_row_values_slice(0), &[]);
    /// assert_eq!(matrix.sparse_row_values_slice(1), &[20, 40]);
    /// assert_eq!(matrix.sparse_row_values_slice(2), &[70]);
    /// ```
    #[inline]
    pub fn sparse_row_values_slice(&self, row: RowIndex) -> &[Value] {
        let range = self.csr.sparse_row_sparse_index_range(row);
        &self.values[range.start.as_()..range.end.as_()]
    }

    /// Returns matching column and value slices stored for a sparse row.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::prelude::*;
    ///
    /// let mut matrix: ValuedCSR2D<usize, usize, usize, i32> =
    ///     SparseMatrixMut::with_sparse_shape((3, 8));
    /// MatrixMut::add(&mut matrix, (1, 2, 20)).unwrap();
    /// MatrixMut::add(&mut matrix, (1, 4, 40)).unwrap();
    /// MatrixMut::add(&mut matrix, (1, 7, 70)).unwrap();
    ///
    /// let (columns, values) = matrix.sparse_row_entries_slice(1);
    ///
    /// assert_eq!(columns, &[2, 4, 7]);
    /// assert_eq!(values, &[20, 40, 70]);
    /// assert_eq!(matrix.sparse_row_entries_slice(0), (&[][..], &[][..]));
    /// ```
    #[inline]
    pub fn sparse_row_entries_slice(&self, row: RowIndex) -> (&[ColumnIndex], &[Value]) {
        (self.csr.sparse_row_slice(row), self.sparse_row_values_slice(row))
    }
}

impl<SparseIndex: AsPrimitive<usize>, RowIndex, ColumnIndex>
    CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: SizedSparseMatrix<SparseIndex = SparseIndex>,
{
    /// Attaches values to the existing CSR topology in storage order.
    ///
    /// # Errors
    ///
    /// Returns [`ValuedCsrPartsError::ValuesLengthMismatch`] when the provided
    /// values do not match the number of sparse coordinates stored in `self`.
    #[inline]
    pub fn with_values<Value>(
        self,
        values: Vec<Value>,
    ) -> Result<ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>, ValuedCsrPartsError> {
        ValuedCSR2D::from_parts(self, values)
    }
}

impl<
    SparseIndex: PositiveInteger + TryFromUsize + AsPrimitive<usize>,
    RowIndex: Step + TryFromUsize + PositiveInteger + AsPrimitive<usize> + Debug,
    ColumnIndex: Step + TryFromUsize + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex> + Debug,
    Value,
    const ROWS: usize,
    const COLS: usize,
> TryFrom<[[Value; COLS]; ROWS]> for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
{
    type Error = MutabilityError<Self>;

    #[inline]
    fn try_from(value: [[Value; COLS]; ROWS]) -> Result<Self, Self::Error> {
        let mut valued_csr: ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value> =
            ValuedCSR2D::with_sparse_shaped_capacity(
                (
                    RowIndex::try_from_usize(ROWS)
                        .map_err(|_| MutabilityError::<Self>::MaxedOutRowIndex)?,
                    ColumnIndex::try_from_usize(COLS)
                        .map_err(|_| MutabilityError::<Self>::MaxedOutColumnIndex)?,
                ),
                SparseIndex::try_from_usize(ROWS * COLS)
                    .map_err(|_| MutabilityError::<Self>::MaxedOutSparseIndex)?,
            );
        for (row, row_values) in valued_csr.row_indices().zip(value) {
            for (column, value) in valued_csr.column_indices().zip(row_values) {
                valued_csr.add((row, column, value)).expect("Failed to add value to ValuedCSR2D");
            }
        }

        Ok(valued_csr)
    }
}

impl<SparseIndex: Debug, RowIndex: Debug, ColumnIndex: Debug, Value: Debug> Debug
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
{
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ValuedCSR2D").field("csr", &self.csr).field("values", &self.values).finish()
    }
}

impl<SparseIndex: Zero, RowIndex: Zero, ColumnIndex: Zero, Value> Default
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
{
    #[inline]
    fn default() -> Self {
        Self { csr: CSR2D::default(), values: Vec::default() }
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> Matrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: Matrix2D,
{
    type Coordinates = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as Matrix>::Coordinates;

    #[inline]
    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows().as_(), self.number_of_columns().as_()]
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> Matrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: Matrix2D,
{
    type ColumnIndex = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as Matrix2D>::ColumnIndex;
    type RowIndex = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as Matrix2D>::RowIndex;

    #[inline]
    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.csr.number_of_columns()
    }

    #[inline]
    fn number_of_rows(&self) -> Self::RowIndex {
        self.csr.number_of_rows()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> Matrix2DRef
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: Matrix2DRef,
{
    #[inline]
    fn number_of_columns_ref(&self) -> &Self::ColumnIndex {
        self.csr.number_of_columns_ref()
    }

    #[inline]
    fn number_of_rows_ref(&self) -> &Self::RowIndex {
        self.csr.number_of_rows_ref()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseMatrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SparseMatrix2D,
{
    type SparseRow<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrix2D>::SparseRow<'a>
    where
        Self: 'a;
    type SparseColumns<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrix2D>::SparseColumns<'a>
    where
        Self: 'a;
    type SparseRows<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrix2D>::SparseRows<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.csr.sparse_rows()
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.csr.sparse_columns()
    }

    #[inline]
    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        self.csr.sparse_row(row)
    }

    #[inline]
    fn has_entry(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> bool {
        self.csr.has_entry(row, column)
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> EmptyRows
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: EmptyRows,
{
    type EmptyRowIndices<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as EmptyRows>::EmptyRowIndices<'a>
    where
        Self: 'a;
    type NonEmptyRowIndices<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as EmptyRows>::NonEmptyRowIndices<'a>
    where
        Self: 'a;

    #[inline]
    fn empty_row_indices(&self) -> Self::EmptyRowIndices<'_> {
        self.csr.empty_row_indices()
    }

    #[inline]
    fn non_empty_row_indices(&self) -> Self::NonEmptyRowIndices<'_> {
        self.csr.non_empty_row_indices()
    }

    #[inline]
    fn number_of_empty_rows(&self) -> Self::RowIndex {
        self.csr.number_of_empty_rows()
    }

    #[inline]
    fn number_of_non_empty_rows(&self) -> Self::RowIndex {
        self.csr.number_of_non_empty_rows()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedRowsSparseMatrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SizedRowsSparseMatrix2D<
            RowIndex = RowIndex,
            ColumnIndex = ColumnIndex,
            SparseIndex = SparseIndex,
        >,
{
    type SparseRowSizes<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SizedRowsSparseMatrix2D>::SparseRowSizes<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.csr.sparse_row_sizes()
    }

    #[inline]
    fn number_of_defined_values_in_row(&self, row: Self::RowIndex) -> Self::ColumnIndex {
        self.csr.number_of_defined_values_in_row(row)
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedSparseMatrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SizedSparseMatrix2D<
            RowIndex = RowIndex,
            ColumnIndex = ColumnIndex,
            SparseIndex = SparseIndex,
        >,
{
    #[inline]
    fn rank_row(&self, row: Self::RowIndex) -> Self::SparseIndex {
        self.csr.rank_row(row)
    }

    #[inline]
    fn select_row(&self, sparse_index: Self::SparseIndex) -> Self::RowIndex {
        self.csr.select_row(sparse_index)
    }

    #[inline]
    fn select_column(&self, sparse_index: Self::SparseIndex) -> Self::ColumnIndex {
        self.csr.select_column(sparse_index)
    }

    #[inline]
    fn try_rank(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> Option<Self::SparseIndex>
    where
        Self::ColumnIndex: PartialEq,
    {
        self.csr.try_rank(row, column)
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SparseMatrix2D,
{
    type SparseIndex = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrix>::SparseIndex;
    type SparseCoordinates<'a>
        = <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrix>::SparseCoordinates<'a>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.csr.sparse_coordinates()
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.csr.last_sparse_coordinates()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.csr.is_empty()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedSparseMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SizedSparseMatrix2D,
{
    #[inline]
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.csr.number_of_defined_values()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> RankSelectSparseMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: RankSelectSparseMatrix<SparseIndex = SparseIndex>,
{
    #[inline]
    fn rank(&self, coordinates: &Self::Coordinates) -> Self::SparseIndex {
        self.csr.rank(coordinates)
    }

    #[inline]
    fn select(&self, sparse_index: Self::SparseIndex) -> Self::Coordinates {
        self.csr.select(sparse_index)
    }
}

impl<SparseIndex: Zero, RowIndex: Zero + Debug, ColumnIndex: Zero + Debug, Value> MatrixMut
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: MatrixMut<
            Entry = (RowIndex, ColumnIndex),
            Error = MutabilityError<CSR2D<SparseIndex, RowIndex, ColumnIndex>>,
        > + Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type Entry = (RowIndex, ColumnIndex, Value);
    type Error = MutabilityError<Self>;

    #[inline]
    fn add(&mut self, (row, column, value): Self::Entry) -> Result<(), Self::Error> {
        self.csr.add((row, column))?;
        self.values.push(value);
        Ok(())
    }

    #[inline]
    fn increase_shape(&mut self, shape: Self::Coordinates) -> Result<(), Self::Error> {
        self.csr.increase_shape(shape)?;
        Ok(())
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    Value,
> SparseMatrixMut for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SparseMatrixMut
        + Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>
        + MatrixMut<
            Entry = (RowIndex, ColumnIndex),
            Error = MutabilityError<CSR2D<SparseIndex, RowIndex, ColumnIndex>>,
        >,
{
    type MinimalShape =
        <CSR2D<SparseIndex, RowIndex, ColumnIndex> as SparseMatrixMut>::MinimalShape;

    #[inline]
    fn with_sparse_capacity(number_of_values: Self::SparseIndex) -> Self {
        Self {
            csr: CSR2D::with_sparse_capacity(number_of_values),
            values: Vec::with_capacity(number_of_values.as_()),
        }
    }

    #[inline]
    fn with_sparse_shape(shape: Self::MinimalShape) -> Self {
        Self { csr: CSR2D::with_sparse_shape(shape), values: Vec::new() }
    }

    #[inline]
    fn with_sparse_shaped_capacity(
        shape: Self::MinimalShape,
        number_of_values: Self::SparseIndex,
    ) -> Self {
        Self {
            csr: CSR2D::with_sparse_shaped_capacity(shape, number_of_values),
            values: Vec::with_capacity(number_of_values.as_()),
        }
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> ValuedMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: Matrix,
{
    type Value = Value;
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> ValuedMatrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: Matrix2D,
{
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SparseMatrix + ValuedMatrix<Value = Value>,
    Self::Value: Clone,
{
    type SparseValues<'a>
        = core::iter::Cloned<core::slice::Iter<'a, Self::Value>>
    where
        Self: 'a;

    #[inline]
    fn sparse_values(&self) -> Self::SparseValues<'_> {
        self.values.iter().cloned()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedSparseValuedMatrix
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SparseValuedMatrix<Value = Value>,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
    Self::Value: Clone,
{
    #[inline]
    fn select_value(&self, sparse_index: Self::SparseIndex) -> Self::Value {
        self.values[sparse_index.as_()].clone()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrix2D
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SizedSparseMatrix2D + SparseValuedMatrix<Value = Value>,
    Self::Value: Clone,
{
    type SparseRowValues<'a>
        = core::iter::Cloned<core::slice::Iter<'a, Self::Value>>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        let start = self.rank_row(row).as_();
        let end = self.rank_row(row + Self::RowIndex::one()).as_();
        self.values[start..end].iter().cloned()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrixRef
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SparseValuedMatrix<Value = Value>,
{
    type SparseValuesRef<'a>
        = core::slice::Iter<'a, Value>
    where
        Self: 'a,
        Value: 'a;

    #[inline]
    fn sparse_values_ref(&self) -> Self::SparseValuesRef<'_> {
        self.values.iter()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedSparseValuedMatrixRef
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SizedSparseValuedMatrix<Value = Value> + SparseValuedMatrixRef<Value = Value>,
{
    #[inline]
    fn select_value_ref(&self, sparse_index: Self::SparseIndex) -> &Self::Value {
        &self.values[sparse_index.as_()]
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrix2DRef
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SizedSparseMatrix2D
        + SparseValuedMatrix2D<Value = Value>
        + SparseValuedMatrixRef<Value = Value>,
{
    type SparseRowValuesRef<'a>
        = core::slice::Iter<'a, Value>
    where
        Self: 'a,
        Value: 'a;

    #[inline]
    fn sparse_row_values_ref(&self, row: Self::RowIndex) -> Self::SparseRowValuesRef<'_> {
        let start = self.rank_row(row).as_();
        let end = self.rank_row(row + Self::RowIndex::one()).as_();
        self.values[start..end].iter()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrixMut
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SparseValuedMatrixRef<Value = Value>,
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SparseMatrix<Coordinates = Self::Coordinates>,
{
    type SparseValuesMut<'a>
        = core::slice::IterMut<'a, Value>
    where
        Self: 'a,
        Value: 'a;

    #[inline]
    fn sparse_values_mut(&mut self) -> Self::SparseValuesMut<'_> {
        self.values.iter_mut()
    }

    #[inline]
    fn sparse_entries_mut(
        &mut self,
    ) -> impl Iterator<Item = (Self::Coordinates, &mut Self::Value)> {
        core::iter::zip(self.csr.sparse_coordinates(), self.values.iter_mut())
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SizedSparseValuedMatrixMut
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SizedSparseValuedMatrixRef<Value = Value> + SparseValuedMatrixMut<Value = Value>,
{
    #[inline]
    fn select_value_mut(&mut self, sparse_index: Self::SparseIndex) -> &mut Self::Value {
        &mut self.values[sparse_index.as_()]
    }
}

impl<SparseIndex, RowIndex, ColumnIndex, Value> SparseValuedMatrix2DMut
    for ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>
where
    Self: SparseValuedMatrix2DRef<Value = Value, RowIndex = RowIndex, ColumnIndex = ColumnIndex>
        + SparseValuedMatrixMut<Value = Value>,
    SparseIndex: AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize>,
    ColumnIndex: PartialEq,
    CSR2D<SparseIndex, RowIndex, ColumnIndex>: SizedSparseMatrix2D<
            RowIndex = RowIndex,
            ColumnIndex = ColumnIndex,
            SparseIndex = SparseIndex,
        > + SparseMatrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type SparseRowValuesMut<'a>
        = core::slice::IterMut<'a, Value>
    where
        Self: 'a,
        Value: 'a;

    #[inline]
    fn sparse_row_values_mut(&mut self, row: Self::RowIndex) -> Self::SparseRowValuesMut<'_> {
        let start = self.csr.rank_row(row).as_();
        let end = self.csr.rank_row(row + RowIndex::one()).as_();
        self.values[start..end].iter_mut()
    }

    fn sparse_value_at_mut(
        &mut self,
        row: Self::RowIndex,
        column: Self::ColumnIndex,
    ) -> Option<&mut <Self as ValuedMatrix>::Value>
    where
        Self::ColumnIndex: PartialEq,
    {
        let sparse_index = self.csr.try_rank(row, column)?;
        Some(&mut self.values[sparse_index.as_()])
    }

    #[inline]
    fn sparse_row_entries_mut(
        &mut self,
        row: Self::RowIndex,
    ) -> impl Iterator<Item = (Self::ColumnIndex, &mut <Self as ValuedMatrix>::Value)> {
        let start = self.csr.rank_row(row).as_();
        let end = self.csr.rank_row(row + RowIndex::one()).as_();
        core::iter::zip(self.csr.sparse_row(row), self.values[start..end].iter_mut())
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::{string::String, vec::Vec};

    use super::*;

    type TestValuedCSR2D = ValuedCSR2D<usize, usize, usize, i32>;

    #[test]
    fn test_valued_csr2d_default() {
        let matrix: TestValuedCSR2D = ValuedCSR2D::default();
        assert_eq!(matrix.number_of_rows(), 0);
        assert_eq!(matrix.number_of_columns(), 0);
        assert_eq!(matrix.number_of_defined_values(), 0);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_valued_csr2d_with_sparse_capacity() {
        let matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_capacity(10);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_valued_csr2d_with_sparse_shape() {
        let matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 4));
        assert_eq!(matrix.number_of_rows(), 3);
        assert_eq!(matrix.number_of_columns(), 4);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_valued_csr2d_add_entries() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert!(matrix.add((0, 1, 10)).is_ok());
        assert!(matrix.add((1, 0, 20)).is_ok());
        assert!(matrix.add((1, 2, 30)).is_ok());
        assert_eq!(matrix.number_of_defined_values(), 3);
    }

    #[test]
    fn test_valued_csr2d_sparse_values() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        matrix.add((0, 1, 2)).unwrap();
        matrix.add((1, 0, 3)).unwrap();

        let values: Vec<i32> = matrix.sparse_values().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_valued_csr2d_sparse_row_values() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 1, 20)).unwrap();
        matrix.add((1, 2, 30)).unwrap();

        let row0_values: Vec<i32> = matrix.sparse_row_values(0).collect();
        assert_eq!(row0_values, vec![10, 20]);

        let row1_values: Vec<i32> = matrix.sparse_row_values(1).collect();
        assert_eq!(row1_values, vec![30]);
    }

    #[test]
    fn test_valued_csr2d_sparse_row_values_slice() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((4, 8));
        matrix.add((1, 2, 20)).unwrap();
        matrix.add((1, 4, 40)).unwrap();
        matrix.add((1, 7, 70)).unwrap();
        matrix.add((3, 5, 50)).unwrap();

        assert_eq!(matrix.sparse_row_values_slice(0), &[]);
        assert_eq!(matrix.sparse_row_values_slice(1), &[20, 40, 70]);
        assert_eq!(matrix.sparse_row_values_slice(2), &[]);
        assert_eq!(matrix.sparse_row_values_slice(3), &[50]);
    }

    #[test]
    fn test_valued_csr2d_sparse_row_entries_slice() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((4, 8));
        matrix.add((1, 2, 20)).unwrap();
        matrix.add((1, 4, 40)).unwrap();
        matrix.add((1, 7, 70)).unwrap();
        matrix.add((3, 5, 50)).unwrap();

        let (columns, values) = matrix.sparse_row_entries_slice(1);
        assert_eq!(columns, &[2, 4, 7]);
        assert_eq!(values, &[20, 40, 70]);

        let (columns, values) = matrix.sparse_row_entries_slice(2);
        assert_eq!(columns, &[]);
        assert_eq!(values, &[]);
    }

    #[test]
    fn test_valued_csr2d_sparse_row() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 2, 20)).unwrap();

        let row0_cols: Vec<usize> = matrix.sparse_row(0).collect();
        assert_eq!(row0_cols, vec![0, 2]);
    }

    #[test]
    fn test_valued_csr2d_has_entry() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 1, 10)).unwrap();

        assert!(!matrix.has_entry(0, 0));
        assert!(matrix.has_entry(0, 1));
        assert!(!matrix.has_entry(1, 0));
    }

    #[test]
    fn test_valued_csr2d_select_value() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 100)).unwrap();
        matrix.add((0, 1, 200)).unwrap();
        matrix.add((1, 1, 300)).unwrap();

        assert_eq!(matrix.select_value(0), 100);
        assert_eq!(matrix.select_value(1), 200);
        assert_eq!(matrix.select_value(2), 300);
    }

    #[test]
    fn test_valued_csr2d_shape() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 4));
        matrix.add((0, 0, 1)).unwrap();
        assert_eq!(matrix.shape(), vec![3, 4]);
    }

    #[test]
    fn test_valued_csr2d_increase_shape() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        assert!(matrix.increase_shape((4, 4)).is_ok());
        assert_eq!(matrix.number_of_rows(), 4);
        assert_eq!(matrix.number_of_columns(), 4);
    }

    #[test]
    fn test_valued_csr2d_debug() {
        let matrix: TestValuedCSR2D = ValuedCSR2D::default();
        let debug = alloc::format!("{matrix:?}");
        assert!(debug.contains("ValuedCSR2D"));
    }

    #[test]
    fn test_valued_csr2d_try_from_array() {
        let arr = [[1, 2], [3, 4]];
        let matrix: TestValuedCSR2D = ValuedCSR2D::try_from(arr).unwrap();
        assert_eq!(matrix.number_of_rows(), 2);
        assert_eq!(matrix.number_of_columns(), 2);
        let values: Vec<i32> = matrix.sparse_values().collect();
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_from_parts_round_trip() {
        let mut csr: CSR2D<usize, usize, usize> = SparseMatrixMut::with_sparse_shape((2, 3));
        csr.add((0, 1)).unwrap();
        csr.add((1, 2)).unwrap();

        let matrix = TestValuedCSR2D::from_parts(csr, vec![10, 20]).unwrap();
        assert_eq!(matrix.values_ref(), &[10, 20]);
        assert_eq!(matrix.sparse_coordinates().collect::<Vec<_>>(), vec![(0, 1), (1, 2)]);

        let (csr, values) = matrix.into_parts();
        assert_eq!(csr.sparse_coordinates().collect::<Vec<_>>(), vec![(0, 1), (1, 2)]);
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn test_from_parts_rejects_length_mismatch() {
        let mut csr: CSR2D<usize, usize, usize> = SparseMatrixMut::with_sparse_shape((2, 2));
        csr.add((0, 0)).unwrap();
        csr.add((1, 1)).unwrap();

        assert_eq!(
            TestValuedCSR2D::from_parts(csr, vec![10]).unwrap_err(),
            ValuedCsrPartsError::ValuesLengthMismatch { expected: 2, actual: 1 }
        );
    }

    #[test]
    fn test_values_ref_matches_sparse_storage_order() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 10)).unwrap();
        matrix.add((0, 2, 20)).unwrap();
        matrix.add((1, 0, 30)).unwrap();

        let entries: Vec<_> = matrix
            .sparse_coordinates()
            .zip(matrix.values_ref().iter())
            .map(|(coordinates, value)| (coordinates, *value))
            .collect();

        assert_eq!(entries, vec![((0, 1), 10), ((0, 2), 20), ((1, 0), 30)]);
    }

    #[test]
    fn test_values_mut_updates_selected_values() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 10)).unwrap();
        matrix.add((1, 2, 20)).unwrap();

        for value in matrix.values_mut() {
            *value += 5;
        }

        assert_eq!(matrix.select_value_ref(0), &15);
        assert_eq!(matrix.select_value_ref(1), &25);
    }

    #[test]
    fn test_values_ref_with_non_copy_type() {
        let mut csr: CSR2D<usize, usize, usize> = SparseMatrixMut::with_sparse_shape((1, 2));
        csr.add((0, 0)).unwrap();
        csr.add((0, 1)).unwrap();

        let matrix =
            ValuedCSR2D::from_parts(csr, vec![String::from("left"), String::from("right")])
                .unwrap();

        assert_eq!(
            matrix.values_ref().iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["left", "right"]
        );
    }

    #[test]
    fn test_sparse_values_ref() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        matrix.add((0, 1, 2)).unwrap();
        matrix.add((1, 0, 3)).unwrap();
        let values: Vec<&i32> = matrix.sparse_values_ref().collect();
        assert_eq!(values, vec![&1, &2, &3]);
    }

    #[test]
    fn test_select_value_ref() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 100)).unwrap();
        matrix.add((0, 1, 200)).unwrap();
        assert_eq!(matrix.select_value_ref(0), &100);
        assert_eq!(matrix.select_value_ref(1), &200);
    }

    #[test]
    fn test_sparse_row_values_ref() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 1, 20)).unwrap();
        matrix.add((1, 2, 30)).unwrap();
        let row0: Vec<&i32> = matrix.sparse_row_values_ref(0).collect();
        assert_eq!(row0, vec![&10, &20]);
        let row1: Vec<&i32> = matrix.sparse_row_values_ref(1).collect();
        assert_eq!(row1, vec![&30]);
    }

    #[test]
    fn test_sparse_value_at_ref() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 42)).unwrap();
        assert_eq!(matrix.sparse_value_at_ref(0, 1), Some(&42));
        assert_eq!(matrix.sparse_value_at_ref(0, 0), None);
        assert_eq!(matrix.sparse_value_at_ref(1, 0), None);
    }

    #[test]
    fn test_ref_on_empty_matrix() {
        let matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert_eq!(matrix.sparse_values_ref().count(), 0);
        assert_eq!(matrix.sparse_row_values_ref(0).count(), 0);
    }

    #[test]
    fn test_sparse_values_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        matrix.add((1, 1, 2)).unwrap();
        for v in matrix.sparse_values_mut() {
            *v *= 10;
        }
        let values: Vec<i32> = matrix.sparse_values().collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn test_select_value_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 5)).unwrap();
        *matrix.select_value_mut(0) = 50;
        assert_eq!(matrix.select_value(0), 50);
    }

    #[test]
    fn test_sparse_row_values_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 1, 20)).unwrap();
        matrix.add((1, 2, 30)).unwrap();
        for v in matrix.sparse_row_values_mut(0) {
            *v += 100;
        }
        let row0: Vec<i32> = matrix.sparse_row_values(0).collect();
        assert_eq!(row0, vec![110, 120]);
        let row1: Vec<i32> = matrix.sparse_row_values(1).collect();
        assert_eq!(row1, vec![30]);
    }

    #[test]
    fn test_sparse_value_at_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 42)).unwrap();
        if let Some(v) = matrix.sparse_value_at_mut(0, 1) {
            *v = 99;
        }
        assert_eq!(matrix.sparse_value_at_ref(0, 1), Some(&99));
        assert!(matrix.sparse_value_at_mut(0, 0).is_none());
    }

    #[test]
    fn test_mut_on_empty_row() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((1, 0, 7)).unwrap();
        assert_eq!(matrix.sparse_row_values_mut(0).count(), 0);
        for v in matrix.sparse_row_values_mut(1) {
            *v = 77;
        }
        assert_eq!(matrix.select_value(0), 77);
    }

    #[test]
    fn test_try_rank() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 10)).unwrap();
        matrix.add((1, 2, 20)).unwrap();
        assert_eq!(matrix.try_rank(0, 1), Some(0));
        assert_eq!(matrix.try_rank(1, 2), Some(1));
        assert_eq!(matrix.try_rank(0, 0), None);
        assert_eq!(matrix.try_rank(0, 2), None);
    }

    #[test]
    fn test_try_select_value() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 100)).unwrap();
        assert_eq!(matrix.try_select_value(0), Some(&100));
        assert_eq!(matrix.try_select_value(1), None);
    }

    #[test]
    fn test_try_select_value_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 5)).unwrap();
        if let Some(v) = matrix.try_select_value_mut(0) {
            *v = 50;
        }
        assert_eq!(matrix.select_value(0), 50);
        assert!(matrix.try_select_value_mut(1).is_none());
    }

    #[test]
    fn test_replace_value() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 1, 42)).unwrap();
        assert_eq!(matrix.replace_value(0, 1, 99), Some(42));
        assert_eq!(matrix.select_value(0), 99);
        assert_eq!(matrix.replace_value(0, 0, 10), None);
    }

    #[test]
    fn test_update_value() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 1, 10)).unwrap();
        assert!(matrix.update_value(0, 1, |v| *v *= 5));
        assert_eq!(matrix.select_value(0), 50);
        assert!(!matrix.update_value(0, 0, |v| *v *= 5));
    }

    #[test]
    fn test_update_value_at() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 7)).unwrap();
        matrix.update_value_at(0, |v| *v += 3);
        assert_eq!(matrix.select_value(0), 10);
    }

    #[test]
    fn test_sparse_entries() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 10)).unwrap();
        matrix.add((1, 2, 20)).unwrap();
        let entries: Vec<_> = matrix.sparse_entries().collect();
        assert_eq!(entries, vec![((0, 1), &10), ((1, 2), &20)]);
    }

    #[test]
    fn test_sparse_entries_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 1, 10)).unwrap();
        matrix.add((1, 2, 20)).unwrap();
        for (_, v) in matrix.sparse_entries_mut() {
            *v *= 10;
        }
        let values: Vec<i32> = matrix.sparse_values().collect();
        assert_eq!(values, vec![100, 200]);
    }

    #[test]
    fn test_sparse_row_entries() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 2, 20)).unwrap();
        matrix.add((1, 1, 30)).unwrap();
        let entries: Vec<_> = matrix.sparse_row_entries(0).collect();
        assert_eq!(entries, vec![(0, &10), (2, &20)]);
    }

    #[test]
    fn test_sparse_row_entries_mut() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 2, 20)).unwrap();
        matrix.add((1, 1, 30)).unwrap();
        for (_, v) in matrix.sparse_row_entries_mut(0) {
            *v += 100;
        }
        let row0: Vec<i32> = matrix.sparse_row_values(0).collect();
        assert_eq!(row0, vec![110, 120]);
        let row1: Vec<i32> = matrix.sparse_row_values(1).collect();
        assert_eq!(row1, vec![30]);
    }

    #[test]
    fn test_try_rank_empty_matrix() {
        let matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert_eq!(matrix.try_rank(0, 0), None);
        assert_eq!(matrix.try_rank(2, 2), None);
    }

    #[test]
    fn test_try_rank_multi_entry_row() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 5));
        matrix.add((0, 0, 1)).unwrap();
        matrix.add((0, 2, 2)).unwrap();
        matrix.add((0, 4, 3)).unwrap();
        matrix.add((1, 1, 4)).unwrap();
        matrix.add((1, 3, 5)).unwrap();
        assert_eq!(matrix.try_rank(0, 0), Some(0));
        assert_eq!(matrix.try_rank(0, 2), Some(1));
        assert_eq!(matrix.try_rank(0, 4), Some(2));
        assert_eq!(matrix.try_rank(0, 1), None);
        assert_eq!(matrix.try_rank(0, 3), None);
        assert_eq!(matrix.try_rank(1, 1), Some(3));
        assert_eq!(matrix.try_rank(1, 3), Some(4));
        assert_eq!(matrix.try_rank(1, 0), None);
    }

    #[test]
    #[should_panic]
    fn test_select_value_ref_panics_oob() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        let _ = matrix.select_value_ref(1);
    }

    #[test]
    #[should_panic]
    fn test_select_value_mut_panics_oob() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        let _ = matrix.select_value_mut(1);
    }

    #[test]
    #[should_panic]
    fn test_update_value_at_panics_oob() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        matrix.add((0, 0, 1)).unwrap();
        matrix.update_value_at(1, |v| *v += 1);
    }

    #[test]
    fn test_sparse_entries_empty_matrix() {
        let matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert_eq!(matrix.sparse_entries().count(), 0);
    }

    #[test]
    fn test_sparse_entries_mut_empty_matrix() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert_eq!(matrix.sparse_entries_mut().count(), 0);
    }

    #[test]
    fn test_sparse_row_entries_empty_row() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((1, 0, 10)).unwrap();
        assert_eq!(matrix.sparse_row_entries(0).count(), 0);
        assert_eq!(matrix.sparse_row_entries(1).count(), 1);
    }

    #[test]
    fn test_sparse_row_entries_mut_empty_row() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((1, 0, 10)).unwrap();
        assert_eq!(matrix.sparse_row_entries_mut(0).count(), 0);
    }

    #[test]
    fn test_replace_value_preserves_other_values() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 1, 20)).unwrap();
        matrix.add((1, 2, 30)).unwrap();
        matrix.replace_value(0, 1, 99);
        assert_eq!(matrix.select_value(0), 10);
        assert_eq!(matrix.select_value(1), 99);
        assert_eq!(matrix.select_value(2), 30);
    }

    #[test]
    fn test_double_ended_ref_iterators() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((2, 3));
        matrix.add((0, 0, 10)).unwrap();
        matrix.add((0, 1, 20)).unwrap();
        matrix.add((0, 2, 30)).unwrap();
        let rev_values: Vec<&i32> = matrix.sparse_values_ref().rev().collect();
        assert_eq!(rev_values, vec![&30, &20, &10]);
        let rev_row: Vec<&i32> = matrix.sparse_row_values_ref(0).rev().collect();
        assert_eq!(rev_row, vec![&30, &20, &10]);
    }

    #[test]
    fn test_double_ended_mut_iterators() {
        let mut matrix: TestValuedCSR2D = SparseMatrixMut::with_sparse_shape((1, 3));
        matrix.add((0, 0, 1)).unwrap();
        matrix.add((0, 1, 2)).unwrap();
        matrix.add((0, 2, 3)).unwrap();
        for (new_value, v) in [100, 200, 300].into_iter().zip(matrix.sparse_values_mut().rev()) {
            *v = new_value;
        }
        let values: Vec<i32> = matrix.sparse_values().collect();
        assert_eq!(values, vec![300, 200, 100]);
    }
}
