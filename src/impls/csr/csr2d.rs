//! Submodule providing a definition of a CSR matrix.
use alloc::vec::Vec;
use core::{fmt::Debug, iter::repeat_n};

use multi_ranged::Step;
use num_traits::{AsPrimitive, Zero};

use crate::{
    impls::{
        CSR2DEmptyRowIndices, CSR2DNonEmptyRowIndices, CSR2DSizedRows, CSR2DSizedRowsizes,
        MutabilityError,
    },
    prelude::*,
    traits::{PositiveInteger, TryFromUsize},
};

#[derive(Clone, Eq, PartialEq, Hash)]
/// A compressed sparse row matrix.
pub struct CSR2D<SparseIndex, RowIndex, ColumnIndex> {
    /// The row pointers.
    pub(super) offsets: Vec<SparseIndex>,
    /// The number of columns.
    pub(super) number_of_columns: ColumnIndex,
    /// The number of rows.
    pub(super) number_of_rows: RowIndex,
    /// The column indices.
    pub(super) column_indices: Vec<ColumnIndex>,
    /// The number of non-empty rows.
    pub(super) number_of_non_empty_rows: RowIndex,
}

impl<SparseIndex: Debug, RowIndex: Debug, ColumnIndex: Debug> Debug
    for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CSR2D")
            .field("offsets", &self.offsets)
            .field("number_of_columns", &self.number_of_columns)
            .field("number_of_rows", &self.number_of_rows)
            .field("column_indices", &self.column_indices)
            .field("number_of_non_empty_rows", &self.number_of_non_empty_rows)
            .finish()
    }
}

impl<SparseIndex: Zero, RowIndex: Zero, ColumnIndex: Zero> Default
    for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    fn default() -> Self {
        Self {
            offsets: vec![SparseIndex::zero()],
            number_of_columns: ColumnIndex::zero(),
            number_of_rows: RowIndex::zero(),
            column_indices: Vec::new(),
            number_of_non_empty_rows: RowIndex::zero(),
        }
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SparseMatrixMut for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: SparseMatrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex, SparseIndex = SparseIndex>,
{
    type MinimalShape = Self::Coordinates;

    fn with_sparse_capacity(number_of_values: Self::SparseIndex) -> Self {
        Self::with_sparse_shaped_capacity((RowIndex::zero(), ColumnIndex::zero()), number_of_values)
    }

    fn with_sparse_shape((number_of_rows, number_of_columns): Self::MinimalShape) -> Self {
        Self::with_sparse_shaped_capacity((number_of_rows, number_of_columns), SparseIndex::zero())
    }

    fn with_sparse_shaped_capacity(
        (number_of_rows, number_of_columns): Self::MinimalShape,
        number_of_values: Self::SparseIndex,
    ) -> Self {
        let mut offsets = Vec::with_capacity(number_of_rows.as_() + 1);
        offsets.push(SparseIndex::zero());
        Self {
            offsets,
            number_of_columns,
            number_of_rows,
            column_indices: Vec::with_capacity(number_of_values.as_()),
            number_of_non_empty_rows: RowIndex::zero(),
        }
    }
}

impl<
    SparseIndex,
    RowIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: PositiveInteger + AsPrimitive<usize>,
> Matrix for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type Coordinates = (RowIndex, ColumnIndex);

    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows.as_(), self.number_of_columns.as_()]
    }
}

impl<
    SparseIndex,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize>,
> Matrix2D for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type RowIndex = RowIndex;
    type ColumnIndex = ColumnIndex;

    fn number_of_rows(&self) -> Self::RowIndex {
        debug_assert!(
            !self.offsets.is_empty(),
            "The offsets should always have at least one element."
        );
        debug_assert!(
            self.offsets.len() - 1 <= self.number_of_rows.as_(),
            "The matrix is in an illegal state where the number of rows {} is less than the number of rows in the offsets {}.",
            self.number_of_rows.as_(),
            self.offsets.len()
        );
        self.number_of_rows
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.number_of_columns
    }
}

impl<
    SparseIndex,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize>,
> Matrix2DRef for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    fn number_of_rows_ref(&self) -> &Self::RowIndex {
        &self.number_of_rows
    }

    fn number_of_columns_ref(&self) -> &Self::ColumnIndex {
        &self.number_of_columns
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SparseMatrix for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type SparseIndex = SparseIndex;
    type SparseCoordinates<'a>
        = super::CSR2DView<'a, Self>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.into()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        if self.is_empty() {
            return None;
        }
        let last_row = self
            .offsets
            .len()
            .checked_sub(2)
            .and_then(|x| RowIndex::try_from_usize(x).ok())
            .expect("The offsets should always have at least one element.");
        debug_assert!(
            self.number_of_defined_values_in_row(last_row) > ColumnIndex::zero(),
            "The last row stores in the offsets should always have at least one column, as all subsequent empty rows should be left implicit and represented by the `number_of_rows` field."
        );
        let last_column = self
            .column_indices
            .last()
            .copied()
            .expect("The column indices cannot be empty if the matrix is not empty.");
        Some((last_row, last_column))
    }

    fn is_empty(&self) -> bool {
        self.number_of_defined_values() == SparseIndex::zero()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SizedSparseMatrix for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.offsets.last().copied().unwrap_or(SparseIndex::zero())
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> RankSelectSparseMatrix for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    fn select(&self, sparse_index: Self::SparseIndex) -> Self::Coordinates {
        (self.select_row(sparse_index), self.select_column(sparse_index))
    }

    fn rank(&self, &(row_index, column_index): &Self::Coordinates) -> Self::SparseIndex {
        let start = self.rank_row(row_index);
        let end = self.rank_row(row_index + RowIndex::one());
        let Ok(relative_column_index) =
            self.column_indices[start.as_()..end.as_()].binary_search(&column_index)
        else {
            panic!("The column index {column_index} is not present in the row {row_index}.");
        };

        start + Self::SparseIndex::try_from_usize(relative_column_index)
            .unwrap_or_else(|_| {
                unreachable!(
                    "The Matrix is in an illegal state where a sparse index is greater than the number of defined values."
                )
            })
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SparseMatrix2D for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type SparseRow<'a>
        = core::iter::Copied<core::slice::Iter<'a, Self::ColumnIndex>>
    where
        Self: 'a;
    type SparseColumns<'a>
        = core::iter::Copied<core::slice::Iter<'a, Self::ColumnIndex>>
    where
        Self: 'a;
    type SparseRows<'a>
        = CSR2DSizedRows<'a, Self>
    where
        Self: 'a;

    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        let start = self.rank_row(row).as_();
        let end = self.rank_row(row + RowIndex::one()).as_();
        self.column_indices[start..end].iter().copied()
    }

    fn has_entry(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> bool {
        let start = self.rank_row(row).as_();
        let end = self.rank_row(row + RowIndex::one()).as_();
        self.column_indices[start..end].binary_search(&column).is_ok()
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.column_indices.iter().copied()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.into()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> EmptyRows for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type EmptyRowIndices<'a>
        = CSR2DEmptyRowIndices<'a, Self>
    where
        Self: 'a;
    type NonEmptyRowIndices<'a>
        = CSR2DNonEmptyRowIndices<'a, Self>
    where
        Self: 'a;
    fn number_of_non_empty_rows(&self) -> Self::RowIndex {
        self.number_of_non_empty_rows
    }

    fn number_of_empty_rows(&self) -> Self::RowIndex {
        self.number_of_rows() - self.number_of_non_empty_rows()
    }

    fn empty_row_indices(&self) -> Self::EmptyRowIndices<'_> {
        self.into()
    }

    fn non_empty_row_indices(&self) -> Self::NonEmptyRowIndices<'_> {
        self.into()
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SizedSparseMatrix2D for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    fn rank_row(&self, row: RowIndex) -> SparseIndex {
        if self.offsets.len() <= row.as_() && row <= self.number_of_rows() {
            return self.number_of_defined_values();
        }
        assert!(
            row <= self.number_of_rows(),
            "The matrix is in an illegal state where the row index {row} is greater than the number of rows {}, with number of columns {}, with offset size {}.",
            self.number_of_rows(),
            self.number_of_columns(),
            self.offsets.len()
        );
        self.offsets[row.as_()]
    }

    fn select_row(&self, sparse_index: Self::SparseIndex) -> Self::RowIndex {
        assert!(
            sparse_index < self.number_of_defined_values(),
            "The sparse index {sparse_index} is out of bounds for a matrix with {} defined values.",
            self.number_of_defined_values()
        );

        // Rows are half-open intervals in `offsets`: [offsets[r], offsets[r + 1]).
        // We therefore need the last row start <= sparse_index, i.e. upper_bound - 1.
        let row = self.offsets.partition_point(|&offset| offset <= sparse_index) - 1;
        Self::RowIndex::try_from_usize(row).unwrap_or_else(|_| {
            unreachable!(
                "The Matrix is in an illegal state where a sparse index is greater than the number of defined values."
            )
        })
    }

    fn select_column(&self, sparse_index: Self::SparseIndex) -> Self::ColumnIndex {
        self.column_indices[sparse_index.as_()]
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> SizedRowsSparseMatrix2D for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type SparseRowSizes<'a>
        = CSR2DSizedRowsizes<'a, Self>
    where
        Self: 'a;

    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.into()
    }

    fn number_of_defined_values_in_row(&self, row: Self::RowIndex) -> Self::ColumnIndex {
        if let Ok(out_degree) =
            (self.rank_row(row + RowIndex::one()) - self.rank_row(row)).try_into()
        {
            out_degree
        } else {
            unreachable!(
                "The Matrix is in an illegal state where a sparse row has a number of defined columns greater than the data type of the columns allows for."
            )
        }
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> MatrixMut for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type Entry = Self::Coordinates;
    type Error = crate::impls::MutabilityError<Self>;

    fn add(&mut self, (row, column): Self::Entry) -> Result<(), Self::Error> {
        if !self.is_empty() && row.as_() == self.offsets.len() - 2 {
            // We check that the provided column is not repeated.
            if self.sparse_row(row).last().is_some_and(|last| last == column) {
                return Err(MutabilityError::DuplicatedEntry((row, column)));
            }
            // We check that the provided column is provided in sorted order.
            if self.sparse_row(row).last().is_some_and(|last| last > column) {
                return Err(MutabilityError::UnorderedCoordinate((row, column)));
            }

            if column == ColumnIndex::max_value() {
                return Err(MutabilityError::MaxedOutColumnIndex);
            }

            if let Some(offset) = self.offsets.last_mut() {
                if *offset == SparseIndex::max_value() {
                    return Err(MutabilityError::MaxedOutSparseIndex);
                }
                *offset += SparseIndex::one();
            } else {
                unreachable!()
            }

            // If the row is the last row, we can add the entry at the end of the column
            // indices.
            self.column_indices.push(column);
            self.number_of_columns = self.number_of_columns.max(column + ColumnIndex::one());

            debug_assert_eq!(
                self.sparse_row(row).last(),
                Some(column),
                "The last column of the row {row} should be equal to the column {column}."
            );

            Ok(())
        } else if row.as_() >= self.offsets.len() - 1 {
            if self.number_of_non_empty_rows == RowIndex::max_value() {
                return Err(MutabilityError::MaxedOutRowIndex);
            }
            if column == ColumnIndex::max_value() {
                return Err(MutabilityError::MaxedOutColumnIndex);
            }
            if row == RowIndex::max_value() {
                return Err(MutabilityError::MaxedOutSparseIndex);
            }
            let last_offset = self.offsets.last().copied().unwrap_or(SparseIndex::zero());
            if last_offset == SparseIndex::max_value() {
                return Err(MutabilityError::MaxedOutSparseIndex);
            }
            // If the row is the next row, we can add the entry at the end of the column
            // indices.
            self.offsets.extend(repeat_n(
                self.number_of_defined_values(),
                (row.as_() + 1) - self.offsets.len(),
            ));
            self.number_of_non_empty_rows += RowIndex::one();
            self.column_indices.push(column);
            self.number_of_columns = self.number_of_columns.max(column + ColumnIndex::one());
            self.number_of_rows = self.number_of_rows.max(row + RowIndex::one());
            self.offsets.push(last_offset + SparseIndex::one());

            debug_assert_eq!(
                self.sparse_row(row).last(),
                Some(column),
                "The last column of the row {row} should be equal to the column {column}."
            );

            Ok(())
        } else {
            Err(MutabilityError::UnorderedCoordinate((row, column)))
        }
    }

    fn increase_shape(
        &mut self,
        (number_of_rows, number_of_columns): Self::Coordinates,
    ) -> Result<(), Self::Error> {
        if number_of_rows < self.number_of_rows() || number_of_columns < self.number_of_columns() {
            return Err(MutabilityError::IncompatibleShape);
        }
        self.number_of_rows = self.number_of_rows.max(number_of_rows);
        self.number_of_columns = self.number_of_columns.max(number_of_columns);
        Ok(())
    }
}

impl<
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFrom<SparseIndex>,
> TransposableMatrix2D<CSR2D<SparseIndex, ColumnIndex, RowIndex>>
    for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
    CSR2D<SparseIndex, ColumnIndex, RowIndex>:
        Matrix2D<RowIndex = ColumnIndex, ColumnIndex = RowIndex>,
{
    fn transpose(&self) -> CSR2D<SparseIndex, ColumnIndex, RowIndex> {
        // We initialize the transposed matrix.
        let mut transposed: CSR2D<SparseIndex, ColumnIndex, RowIndex> = CSR2D {
            offsets: vec![SparseIndex::zero(); self.number_of_columns().as_() + 1],
            number_of_columns: self.number_of_rows(),
            number_of_rows: self.number_of_columns(),
            column_indices: vec![RowIndex::zero(); self.number_of_defined_values().as_()],
            number_of_non_empty_rows: ColumnIndex::zero(),
        };

        // First, we proceed to compute the number of elements in each column.
        for column in self.column_indices.iter().copied() {
            transposed.offsets[column.as_() + 1] += SparseIndex::one();
        }

        // Then, we compute the prefix sum of the degrees to get the offsets.
        let mut prefix_sum = SparseIndex::zero();
        for (row_degree_index, offset) in transposed.offsets.iter_mut().enumerate() {
            // Before prefix summation, offsets[1..] store row degrees in the transposed matrix.
            if row_degree_index > 0 && *offset > SparseIndex::zero() {
                transposed.number_of_non_empty_rows += ColumnIndex::one();
            }
            prefix_sum += *offset;
            *offset = prefix_sum;
        }

        debug_assert!(
            transposed.number_of_non_empty_rows <= transposed.number_of_rows,
            "The transposed matrix has {} non-empty rows but only {} rows.",
            transposed.number_of_non_empty_rows,
            transposed.number_of_rows
        );

        // Finally, we fill the column indices.
        let mut degree = vec![SparseIndex::zero(); self.number_of_columns.as_()];
        for (row, column) in crate::traits::SparseMatrix::sparse_coordinates(self) {
            let current_degree: &mut SparseIndex = &mut degree[column.as_()];
            let index = *current_degree + transposed.offsets[column.as_()];
            transposed.column_indices[index.as_()] = row;
            *current_degree += SparseIndex::one();
        }

        transposed
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::traits::MatrixMut;

    type TestCSR2D = CSR2D<usize, usize, usize>;

    #[test]
    fn test_csr2d_default() {
        let csr: TestCSR2D = CSR2D::default();
        assert_eq!(csr.number_of_rows(), 0);
        assert_eq!(csr.number_of_columns(), 0);
        assert_eq!(csr.number_of_defined_values(), 0);
        assert!(csr.is_empty());
    }

    #[test]
    fn test_csr2d_with_sparse_shape() {
        let csr: TestCSR2D = SparseMatrixMut::with_sparse_shape((3, 4));
        assert_eq!(csr.number_of_rows(), 3);
        assert_eq!(csr.number_of_columns(), 4);
        assert!(csr.is_empty());
    }

    #[test]
    fn test_csr2d_with_sparse_capacity() {
        let csr: TestCSR2D = SparseMatrixMut::with_sparse_capacity(10);
        assert!(csr.is_empty());
    }

    #[test]
    fn test_csr2d_add_entries() {
        let mut csr: TestCSR2D = CSR2D::default();
        assert!(MatrixMut::add(&mut csr, (0, 0)).is_ok());
        assert!(MatrixMut::add(&mut csr, (0, 1)).is_ok());
        assert!(MatrixMut::add(&mut csr, (1, 2)).is_ok());
        assert_eq!(csr.number_of_defined_values(), 3);
        assert_eq!(csr.number_of_rows(), 2);
        assert_eq!(csr.number_of_columns(), 3);
    }

    #[test]
    fn test_csr2d_add_duplicate_error() {
        let mut csr: TestCSR2D = CSR2D::default();
        assert!(MatrixMut::add(&mut csr, (0, 1)).is_ok());
        assert!(MatrixMut::add(&mut csr, (0, 1)).is_err());
    }

    #[test]
    fn test_csr2d_add_unordered_error() {
        let mut csr: TestCSR2D = CSR2D::default();
        assert!(MatrixMut::add(&mut csr, (0, 2)).is_ok());
        assert!(MatrixMut::add(&mut csr, (0, 1)).is_err());
    }

    #[test]
    fn test_csr2d_sparse_row() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (0, 3)).unwrap();
        MatrixMut::add(&mut csr, (1, 2)).unwrap();

        let row0: Vec<usize> = csr.sparse_row(0).collect();
        assert_eq!(row0, vec![1, 3]);

        let row1: Vec<usize> = csr.sparse_row(1).collect();
        assert_eq!(row1, vec![2]);
    }

    #[test]
    fn test_csr2d_has_entry() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (0, 3)).unwrap();

        assert!(!csr.has_entry(0, 0));
        assert!(csr.has_entry(0, 1));
        assert!(!csr.has_entry(0, 2));
        assert!(csr.has_entry(0, 3));
    }

    #[test]
    fn test_csr2d_shape() {
        let mut csr: TestCSR2D = SparseMatrixMut::with_sparse_shape((3, 4));
        MatrixMut::add(&mut csr, (0, 0)).unwrap();
        assert_eq!(csr.shape(), vec![3, 4]);
    }

    #[test]
    fn test_csr2d_sparse_coordinates() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 0)).unwrap();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (1, 0)).unwrap();

        let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&csr).collect();
        assert_eq!(coords, vec![(0, 0), (0, 1), (1, 0)]);
    }

    #[test]
    fn test_csr2d_last_sparse_coordinates() {
        let mut csr: TestCSR2D = CSR2D::default();
        assert_eq!(csr.last_sparse_coordinates(), None);

        MatrixMut::add(&mut csr, (0, 0)).unwrap();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        assert_eq!(csr.last_sparse_coordinates(), Some((0, 1)));

        MatrixMut::add(&mut csr, (1, 2)).unwrap();
        assert_eq!(csr.last_sparse_coordinates(), Some((1, 2)));
    }

    #[test]
    fn test_csr2d_transpose() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (0, 2)).unwrap();
        MatrixMut::add(&mut csr, (1, 0)).unwrap();

        let transposed: TestCSR2D = csr.transpose();
        assert!(transposed.has_entry(1, 0));
        assert!(transposed.has_entry(2, 0));
        assert!(transposed.has_entry(0, 1));
    }

    #[test]
    fn test_csr2d_empty_rows() {
        let mut csr: TestCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (2, 0)).unwrap();

        assert_eq!(csr.number_of_non_empty_rows(), 2);
        assert_eq!(csr.number_of_empty_rows(), 1);
    }

    #[test]
    fn test_csr2d_increase_shape() {
        let mut csr: TestCSR2D = SparseMatrixMut::with_sparse_shape((2, 2));
        assert!(csr.increase_shape((4, 4)).is_ok());
        assert_eq!(csr.number_of_rows(), 4);
        assert_eq!(csr.number_of_columns(), 4);
    }

    #[test]
    fn test_csr2d_increase_shape_error() {
        let mut csr: TestCSR2D = SparseMatrixMut::with_sparse_shape((3, 3));
        assert!(csr.increase_shape((2, 3)).is_err());
        assert!(csr.increase_shape((3, 2)).is_err());
    }

    #[test]
    fn test_csr2d_rank_and_select() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 0)).unwrap();
        MatrixMut::add(&mut csr, (0, 1)).unwrap();
        MatrixMut::add(&mut csr, (1, 0)).unwrap();

        // rank returns the sparse index for a given (row, column)
        assert_eq!(csr.rank(&(0, 0)), 0);
        assert_eq!(csr.rank(&(0, 1)), 1);
        assert_eq!(csr.rank(&(1, 0)), 2);

        // select returns the (row, column) for a given sparse index
        assert_eq!(csr.select(0), (0, 0));
        assert_eq!(csr.select(2), (1, 0));
    }

    #[test]
    fn test_csr2d_debug() {
        let csr: TestCSR2D = CSR2D::default();
        let debug = alloc::format!("{csr:?}");
        assert!(debug.contains("CSR2D"));
    }

    #[test]
    fn test_csr2d_clone() {
        let mut csr: TestCSR2D = CSR2D::default();
        MatrixMut::add(&mut csr, (0, 0)).unwrap();
        let cloned = csr.clone();
        assert_eq!(csr, cloned);
    }
}
