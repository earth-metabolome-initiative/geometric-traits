//! Iterator of the sparse coordinates of the CSR2D matrix.

use num_traits::{AsPrimitive, One, Zero};

use crate::prelude::*;

/// Iterator of the sparse coordinates of the CSR2D matrix.
pub struct CSR2DSizedRowsizes<'a, CSR: SizedRowsSparseMatrix2D> {
    /// The CSR matrix.
    csr2d: &'a CSR,
    /// The row index.
    next_row: CSR::RowIndex,
    /// The end row index.
    back_row: CSR::RowIndex,
    /// Whether the iterator is exhausted.
    exhausted: bool,
}

impl<CSR: SizedRowsSparseMatrix2D> Iterator for CSR2DSizedRowsizes<'_, CSR> {
    type Item = CSR::ColumnIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted || self.next_row > self.back_row {
            self.exhausted = true;
            return None;
        }

        let out_degree = self.csr2d.number_of_defined_values_in_row(self.next_row);
        if self.next_row == self.back_row {
            self.exhausted = true;
        } else {
            self.next_row += CSR::RowIndex::one();
        }

        Some(out_degree)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.exhausted || self.next_row > self.back_row {
            0
        } else {
            (self.back_row + CSR::RowIndex::one() - self.next_row).as_()
        };
        (remaining, Some(remaining))
    }
}

impl<CSR: SizedRowsSparseMatrix2D> ExactSizeIterator for CSR2DSizedRowsizes<'_, CSR> {
    #[inline]
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

impl<CSR: SizedRowsSparseMatrix2D> DoubleEndedIterator for CSR2DSizedRowsizes<'_, CSR> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.exhausted || self.next_row > self.back_row {
            self.exhausted = true;
            return None;
        }

        let row = self.back_row;
        let out_degree = self.csr2d.number_of_defined_values_in_row(row);
        if self.next_row == self.back_row {
            self.exhausted = true;
        } else {
            self.back_row -= CSR::RowIndex::one();
        }

        Some(out_degree)
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D> From<&'a CSR> for CSR2DSizedRowsizes<'a, CSR> {
    #[inline]
    fn from(csr2d: &'a CSR) -> Self {
        let next_row = CSR::RowIndex::zero();
        let mut back_row = CSR::RowIndex::zero();
        let has_rows = next_row < csr2d.number_of_rows();
        if has_rows {
            back_row = csr2d.number_of_rows() - CSR::RowIndex::one();
        }
        Self { csr2d, next_row, back_row, exhausted: !has_rows }
    }
}
