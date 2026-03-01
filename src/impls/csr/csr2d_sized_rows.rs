//! Iterator of the sparse coordinates of the CSR2D matrix.

use core::{
    cmp::Ordering,
    iter::{RepeatN, repeat_n},
};

use num_traits::{AsPrimitive, One, Zero};

use crate::prelude::*;

/// Iterator of the sparse coordinates of the CSR2D matrix.
pub struct CSR2DSizedRows<'a, CSR: SizedRowsSparseMatrix2D> {
    /// The CSR matrix.
    csr2d: &'a CSR,
    /// The row index.
    next_row: CSR::RowIndex,
    /// The end row index.
    back_row: CSR::RowIndex,
    /// The row associated with the index at the beginning of the iteration.
    next: Option<RepeatN<CSR::RowIndex>>,
    /// The row associated with the index at the end of the iteration.
    back: Option<RepeatN<CSR::RowIndex>>,
}

impl<CSR: SizedRowsSparseMatrix2D> Iterator for CSR2DSizedRows<'_, CSR> {
    type Item = CSR::RowIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(row) = self.next.as_mut()?.next() {
                return Some(row);
            }

            match self.next_row.cmp(&self.back_row) {
                Ordering::Less => {
                    self.next_row += CSR::RowIndex::one();
                    if self.next_row == self.back_row {
                        self.next = self.back.take();
                    } else {
                        self.next = Some(repeat_n(
                            self.next_row,
                            self.csr2d.number_of_defined_values_in_row(self.next_row).as_(),
                        ));
                    }
                }
                Ordering::Equal | Ordering::Greater => {
                    return self.back.as_mut().and_then(Iterator::next);
                }
            }
        }
    }
}

impl<CSR: SizedSparseMatrix2D> ExactSizeIterator for CSR2DSizedRows<'_, CSR> {
    #[inline]
    fn len(&self) -> usize {
        if self.next.is_none() {
            return 0;
        }
        if self.back.is_none() {
            return self.next.as_ref().map_or(0, ExactSizeIterator::len);
        }

        // Entries in rows [next_row, back_row) minus already consumed from
        // next_row, plus remaining entries in back_row via `self.back`.
        let next_row_rank = self.csr2d.rank_row(self.next_row).as_();
        let already_observed_in_next_row =
            self.csr2d.number_of_defined_values_in_row(self.next_row).as_()
                - self.next.as_ref().map_or(0, ExactSizeIterator::len);
        let back_row_rank = self.csr2d.rank_row(self.back_row).as_();
        back_row_rank - next_row_rank - already_observed_in_next_row
            + self.back.as_ref().map_or(0, ExactSizeIterator::len)
    }
}

impl<CSR: SizedRowsSparseMatrix2D> DoubleEndedIterator for CSR2DSizedRows<'_, CSR> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(back) = self.back.as_mut() {
                if let Some(row) = back.next_back() {
                    return Some(row);
                }

                if self.back_row > self.next_row {
                    self.back_row -= CSR::RowIndex::one();
                    if self.back_row == self.next_row {
                        self.back = None;
                    } else {
                        self.back = Some(repeat_n(
                            self.back_row,
                            self.csr2d.number_of_defined_values_in_row(self.back_row).as_(),
                        ));
                    }
                } else {
                    self.back = None;
                }
            } else {
                return self.next.as_mut().and_then(DoubleEndedIterator::next_back);
            }
        }
    }
}

impl<'a, CSR: SizedRowsSparseMatrix2D> From<&'a CSR> for CSR2DSizedRows<'a, CSR> {
    #[inline]
    fn from(csr2d: &'a CSR) -> Self {
        let next_row = CSR::RowIndex::zero();
        let mut back_row = CSR::RowIndex::zero();
        let has_rows = next_row < csr2d.number_of_rows();
        let next = has_rows
            .then(|| repeat_n(next_row, csr2d.number_of_defined_values_in_row(next_row).as_()));
        let back = if has_rows {
            back_row = csr2d.number_of_rows() - CSR::RowIndex::one();
            (next_row < back_row)
                .then(|| repeat_n(back_row, csr2d.number_of_defined_values_in_row(back_row).as_()))
        } else {
            None
        };

        Self { csr2d, next_row, back_row, next, back }
    }
}
