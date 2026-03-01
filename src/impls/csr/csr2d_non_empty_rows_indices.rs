//! Submodule providing an iterator over the indices of non-empty rows in a CSR
//! matrix.

use multi_ranged::SimpleRange;
use num_traits::Zero;

use crate::traits::SizedRowsSparseMatrix2D;

/// Iterator over the indices of non-empty rows in a CSR matrix.
pub struct CSR2DNonEmptyRowIndices<'a, CSR: SizedRowsSparseMatrix2D + 'a> {
    /// The iterator of the row indices and their sizes.
    row_sizes: (SimpleRange<CSR::RowIndex>, CSR::SparseRowSizes<'a>),
    /// The number of non-empty rows still to return.
    remaining_non_empty_rows: usize,
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> Iterator for CSR2DNonEmptyRowIndices<'a, CSR> {
    type Item = CSR::RowIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(row_index), Some(row_size)) =
            (self.row_sizes.0.next(), self.row_sizes.1.next())
        {
            if row_size > CSR::ColumnIndex::zero() {
                self.remaining_non_empty_rows -= 1;
                return Some(row_index);
            }
        }
        None
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> ExactSizeIterator for CSR2DNonEmptyRowIndices<'a, CSR> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining_non_empty_rows
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D + 'a> DoubleEndedIterator
    for CSR2DNonEmptyRowIndices<'a, CSR>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        while let (Some(row_index), Some(row_size)) =
            (self.row_sizes.0.next_back(), self.row_sizes.1.next_back())
        {
            if row_size > CSR::ColumnIndex::zero() {
                self.remaining_non_empty_rows -= 1;
                return Some(row_index);
            }
        }
        None
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D> From<&'a CSR> for CSR2DNonEmptyRowIndices<'a, CSR> {
    #[inline]
    fn from(csr2d: &'a CSR) -> Self {
        let remaining_non_empty_rows = csr2d
            .row_indices()
            .filter(|&row| csr2d.number_of_defined_values_in_row(row) > CSR::ColumnIndex::zero())
            .count();

        CSR2DNonEmptyRowIndices {
            row_sizes: (csr2d.row_indices(), csr2d.sparse_row_sizes()),
            remaining_non_empty_rows,
        }
    }
}
