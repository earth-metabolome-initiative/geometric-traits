//! Submodule providing a definition of a CSR matrix.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{fmt::Debug, iter::repeat_n};

use multi_ranged::{MultiRanged, Step, errors::Error as RangedError};
use num_traits::{One, Zero};

use num_traits::AsPrimitive;

use crate::{
    impls::MutabilityError,
    prelude::*,
    traits::{PositiveInteger, TryFromUsize},
};

#[derive(Clone)]
/// A compressed sparse row matrix.
pub struct RangedCSR2D<SparseIndex, RowIndex, R: MultiRanged> {
    /// The number of elements in the matrix.
    pub(super) number_of_defined_values: SparseIndex,
    /// The number of columns.
    pub(super) number_of_columns: R::Step,
    /// The number of rows.
    pub(super) number_of_rows: RowIndex,
    /// The destination ranges.
    pub(super) ranges: Vec<R>,
    /// The number of non-empty rows.
    pub(super) number_of_non_empty_rows: RowIndex,
}

impl<SparseIndex: Debug, RowIndex: Debug, R: MultiRanged> Debug
    for RangedCSR2D<SparseIndex, RowIndex, R>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RangedCSR2D")
            .field("number_of_defined_values", &self.number_of_defined_values)
            .field("number_of_columns", &self.number_of_columns)
            .field("number_of_rows", &self.number_of_rows)
            .field("column_indices", &self.ranges)
            .field("number_of_non_empty_rows", &self.number_of_non_empty_rows)
            .finish()
    }
}

impl<SparseIndex: Zero, RowIndex: Zero, R: MultiRanged> Default
    for RangedCSR2D<SparseIndex, RowIndex, R>
{
    fn default() -> Self {
        Self {
            number_of_defined_values: SparseIndex::zero(),
            number_of_columns: R::Step::zero(),
            number_of_rows: RowIndex::zero(),
            ranges: Vec::new(),
            number_of_non_empty_rows: RowIndex::zero(),
        }
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> SparseMatrixMut for RangedCSR2D<SparseIndex, RowIndex, R>
where
    Self: SparseMatrix2D<RowIndex = RowIndex, ColumnIndex = R::Step, SparseIndex = SparseIndex>,
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <<R as MultiRanged>::Step as TryFrom<usize>>::Error: Debug,
{
    type MinimalShape = Self::Coordinates;

    fn with_sparse_capacity(number_of_values: Self::SparseIndex) -> Self {
        Self::with_sparse_shaped_capacity((RowIndex::zero(), R::Step::zero()), number_of_values)
    }

    fn with_sparse_shape(shape: Self::MinimalShape) -> Self {
        Self::with_sparse_shaped_capacity(shape, SparseIndex::zero())
    }

    fn with_sparse_shaped_capacity(
        (number_of_rows, number_of_columns): Self::MinimalShape,
        _number_of_values: Self::SparseIndex,
    ) -> Self {
        Self {
            number_of_defined_values: SparseIndex::zero(),
            number_of_columns,
            number_of_rows,
            ranges: Vec::with_capacity(number_of_rows.as_()),
            number_of_non_empty_rows: RowIndex::zero(),
        }
    }
}

impl<SparseIndex, RowIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize, R: MultiRanged> Matrix
    for RangedCSR2D<SparseIndex, RowIndex, R>
where
    R::Step: AsPrimitive<usize> + PositiveInteger,
{
    type Coordinates = (RowIndex, R::Step);

    #[inline]
    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows.as_(), self.number_of_columns.as_()]
    }
}

impl<SparseIndex, RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize, R: MultiRanged>
    Matrix2D for RangedCSR2D<SparseIndex, RowIndex, R>
where
    R::Step: AsPrimitive<usize> + PositiveInteger,
{
    type RowIndex = RowIndex;
    type ColumnIndex = R::Step;

    #[inline]
    fn number_of_rows(&self) -> Self::RowIndex {
        self.number_of_rows
    }

    #[inline]
    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.number_of_columns
    }
}

impl<SparseIndex, RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize, R: MultiRanged>
    Matrix2DRef for RangedCSR2D<SparseIndex, RowIndex, R>
where
    R::Step: AsPrimitive<usize> + PositiveInteger,
{
    #[inline]
    fn number_of_columns_ref(&self) -> &Self::ColumnIndex {
        &self.number_of_columns
    }

    #[inline]
    fn number_of_rows_ref(&self) -> &Self::RowIndex {
        &self.number_of_rows
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> SparseMatrix for RangedCSR2D<SparseIndex, RowIndex, R>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = R::Step>,
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <<R as MultiRanged>::Step as TryFrom<usize>>::Error: Debug,
{
    type SparseIndex = SparseIndex;
    type SparseCoordinates<'a>
        = crate::impls::CSR2DView<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.into()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        if self.is_empty() {
            return None;
        }
        let last_row_index = RowIndex::try_from_usize(self.ranges.len() - 1)
            .expect("The matrix is in a valid state.");
        let last_row_with_values = self.ranges.last().expect("The matrix should not be empty.");
        let last_column =
            last_row_with_values.clone().last().expect("The last row should not be empty.");
        Some((last_row_index, last_column))
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.number_of_defined_values == SparseIndex::zero()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> SizedSparseMatrix for RangedCSR2D<SparseIndex, RowIndex, R>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = R::Step>,
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <<R as MultiRanged>::Step as TryFrom<usize>>::Error: Debug,
{
    #[inline]
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.number_of_defined_values
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> SparseMatrix2D for RangedCSR2D<SparseIndex, RowIndex, R>
where
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <R::Step as TryFrom<usize>>::Error: Debug,
{
    type SparseRow<'a>
        = R
    where
        Self: 'a;
    type SparseColumns<'a>
        = crate::impls::CSR2DColumns<'a, Self>
    where
        Self: 'a;
    type SparseRows<'a>
        = crate::impls::CSR2DSizedRows<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        self.ranges[row.as_()].clone()
    }

    #[inline]
    fn has_entry(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> bool {
        self.sparse_row(row).contains(column)
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.into()
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.into()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> EmptyRows for RangedCSR2D<SparseIndex, RowIndex, R>
where
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <R::Step as TryFrom<usize>>::Error: Debug,
{
    type EmptyRowIndices<'a>
        = crate::impls::CSR2DEmptyRowIndices<'a, Self>
    where
        Self: 'a;
    type NonEmptyRowIndices<'a>
        = crate::impls::CSR2DNonEmptyRowIndices<'a, Self>
    where
        Self: 'a;
    #[inline]
    fn empty_row_indices(&self) -> Self::EmptyRowIndices<'_> {
        self.into()
    }

    #[inline]
    fn non_empty_row_indices(&self) -> Self::NonEmptyRowIndices<'_> {
        self.into()
    }

    #[inline]
    fn number_of_empty_rows(&self) -> Self::RowIndex {
        self.number_of_rows() - self.number_of_non_empty_rows()
    }

    #[inline]
    fn number_of_non_empty_rows(&self) -> Self::RowIndex {
        self.number_of_non_empty_rows
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> SizedRowsSparseMatrix2D for RangedCSR2D<SparseIndex, RowIndex, R>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = R::Step>,
    Self::ColumnIndex: TryFromUsize,
    R::Step: AsPrimitive<usize> + PositiveInteger,
    <R::Step as TryFrom<usize>>::Error: Debug,
    <RowIndex as TryFrom<usize>>::Error: Debug,
{
    type SparseRowSizes<'a>
        = crate::impls::CSR2DSizedRowsizes<'a, Self>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.into()
    }

    #[inline]
    fn number_of_defined_values_in_row(&self, row: Self::RowIndex) -> Self::ColumnIndex {
        Self::ColumnIndex::try_from_usize(self.ranges[row.as_()].len()).unwrap()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize>,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    R: MultiRanged,
> MatrixMut for RangedCSR2D<SparseIndex, RowIndex, R>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = R::Step>,
    R::Step: AsPrimitive<usize> + PositiveInteger + TryFromUsize,
    <R::Step as TryFrom<usize>>::Error: Debug,
    <RowIndex as TryFrom<usize>>::Error: Debug,
{
    type Entry = Self::Coordinates;
    type Error = MutabilityError<Self>;

    fn add(&mut self, (row, column): Self::Entry) -> Result<(), Self::Error> {
        if row.as_() >= self.ranges.len() {
            self.ranges.extend(repeat_n(R::default(), row.as_() - self.ranges.len() + 1));
        }

        let range = &mut self.ranges[row.as_()];

        if let Err(err) = range.insert(column) {
            match err {
                RangedError::DuplicateElement(_) => {
                    return Err(MutabilityError::DuplicatedEntry((row, column)));
                }
                RangedError::OutOfRange(_) | RangedError::NotSorted(_) => {
                    return Err(MutabilityError::UnorderedCoordinate((row, column)));
                }
                RangedError::NotDense => {
                    unreachable!("This error cannot occur in a CSR matrix.");
                }
            }
        }

        self.number_of_defined_values += SparseIndex::one();
        self.number_of_columns = self.number_of_columns.max(column + R::Step::one());
        self.number_of_rows = self.number_of_rows.max(row + RowIndex::one());

        if R::Step::try_from_usize(range.len()).unwrap() == R::Step::one() {
            self.number_of_non_empty_rows += RowIndex::one();
        }

        Ok(())
    }

    fn increase_shape(&mut self, shape: Self::Coordinates) -> Result<(), Self::Error> {
        if shape.0 < self.number_of_rows || shape.1 < self.number_of_columns {
            return Err(MutabilityError::IncompatibleShape);
        }

        self.ranges.extend(repeat_n(R::default(), shape.0.as_() - self.ranges.len()));

        self.number_of_rows = shape.0;
        self.number_of_columns = shape.1;

        Ok(())
    }
}

impl<SparseIndex: PositiveInteger + AsPrimitive<usize> + 'static, R1: MultiRanged, R2: MultiRanged>
    TransposableMatrix2D<RangedCSR2D<SparseIndex, R1::Step, R2>>
    for RangedCSR2D<SparseIndex, R2::Step, R1>
where
    Self: Matrix2D<RowIndex = R2::Step, ColumnIndex = R1::Step>,
    RangedCSR2D<SparseIndex, R1::Step, R2>: Matrix2D<RowIndex = R1::Step, ColumnIndex = R2::Step>,
    R1::Step: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    R2::Step: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    <<R1 as MultiRanged>::Step as TryFrom<usize>>::Error: Debug,
    <<R2 as MultiRanged>::Step as TryFrom<usize>>::Error: Debug,
{
    fn transpose(&self) -> RangedCSR2D<SparseIndex, R1::Step, R2> {
        // We initialize the transposed matrix.
        let mut transposed: RangedCSR2D<SparseIndex, R1::Step, R2> =
            RangedCSR2D::with_sparse_shaped_capacity(
                (self.number_of_columns, self.number_of_rows),
                self.number_of_defined_values,
            );

        // We iterate over the rows of the matrix.
        for (row, column) in crate::traits::SparseMatrix::sparse_coordinates(self) {
            crate::traits::GrowableEdges::add(&mut transposed, (column, row))
                .expect("The addition should not fail.");
        }

        transposed
    }
}
