//! Submodule providing an iterator over the indices of empty rows in a CSR
//! matrix.

use multi_ranged::SimpleRange;
use num_traits::Zero;

use crate::traits::{Matrix2D, SizedRowsSparseMatrix2D};

/// Iterator over the indices of empty rows in a CSR matrix.
pub struct CSR2DEmptyRowIndices<'a, CSR: SizedRowsSparseMatrix2D + 'a> {
    /// The iterator of the row indices and their sizes.
    row_sizes: (SimpleRange<CSR::RowIndex>, CSR::SparseRowSizes<'a>),
    /// The number of empty rows still to return.
    remaining_empty_rows: usize,
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> Iterator for CSR2DEmptyRowIndices<'a, CSR> {
    type Item = CSR::RowIndex;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(row_index), Some(row_size)) =
            (self.row_sizes.0.next(), self.row_sizes.1.next())
        {
            if row_size == <CSR as Matrix2D>::ColumnIndex::zero() {
                self.remaining_empty_rows -= 1;
                return Some(row_index);
            }
        }
        None
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> ExactSizeIterator for CSR2DEmptyRowIndices<'a, CSR> {
    fn len(&self) -> usize {
        self.remaining_empty_rows
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> DoubleEndedIterator for CSR2DEmptyRowIndices<'a, CSR> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let (Some(row_index), Some(row_size)) =
            (self.row_sizes.0.next_back(), self.row_sizes.1.next_back())
        {
            if row_size == <CSR as Matrix2D>::ColumnIndex::zero() {
                self.remaining_empty_rows -= 1;
                return Some(row_index);
            }
        }
        None
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D> From<&'a CSR> for CSR2DEmptyRowIndices<'a, CSR> {
    fn from(csr2d: &'a CSR) -> Self {
        let remaining_empty_rows = csr2d
            .row_indices()
            .filter(|&row| csr2d.number_of_defined_values_in_row(row) == CSR::ColumnIndex::zero())
            .count();

        CSR2DEmptyRowIndices {
            row_sizes: (csr2d.row_indices(), csr2d.sparse_row_sizes()),
            remaining_empty_rows,
        }
    }
}
