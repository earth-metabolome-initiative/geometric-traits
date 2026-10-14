//! Iterator of the sparse coordinates of the CSR2D matrix.

use core::cmp::Ordering;

use num_traits::{AsPrimitive, One, Zero};

use crate::prelude::*;

/// Iterator of the sparse coordinates of the CSR2D matrix.
pub struct CSR2DView<'a, CSR: SparseMatrix2D> {
    /// The CSR matrix.
    csr2d: &'a CSR,
    /// The row index.
    next_row: CSR::RowIndex,
    /// The end row index.
    back_row: CSR::RowIndex,
    /// The row associated with the index at the beginning of the iteration.
    next: Option<CSR::SparseRow<'a>>,
    /// The row associated with the index at the end of the iteration.
    back: Option<CSR::SparseRow<'a>>,
}

impl<CSR: SparseMatrix2D> Iterator for CSR2DView<'_, CSR> {
    type Item = CSR::Coordinates;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(column_index) = self.next.as_mut()?.next() {
                return Some((self.next_row, column_index));
            }

            match self.next_row.cmp(&self.back_row) {
                Ordering::Less => {
                    self.next_row += CSR::RowIndex::one();
                    if self.next_row == self.back_row {
                        self.next = self.back.take();
                    } else {
                        self.next = Some(self.csr2d.sparse_row(self.next_row));
                    }
                }
                Ordering::Equal | Ordering::Greater => {
                    return self
                        .back
                        .as_mut()
                        .and_then(Iterator::next)
                        .map(|column_index| (self.back_row, column_index));
                }
            }
        }
    }
}

impl<'matrix, CSR: SizedSparseMatrix2D> ExactSizeIterator for CSR2DView<'matrix, CSR>
where
    CSR::SparseRow<'matrix>: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        if self.next.is_none() {
            return 0;
        }
        if self.back.is_none() {
            return self.next.as_ref().map_or(0, ExactSizeIterator::len);
        }

        let next_row_rank = self.csr2d.rank_row(self.next_row).as_();
        let already_observed_in_next_row =
            self.csr2d.number_of_defined_values_in_row(self.next_row).as_()
                - self.next.as_ref().map_or(0, ExactSizeIterator::len);
        let back_row_rank = self.csr2d.rank_row(self.back_row).as_();
        let still_to_be_observed_in_back_row = self.back.as_ref().map_or(0, ExactSizeIterator::len);
        back_row_rank + still_to_be_observed_in_back_row
            - next_row_rank
            - already_observed_in_next_row
    }
}

impl<CSR: SparseMatrix2D> DoubleEndedIterator for CSR2DView<'_, CSR> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(back) = self.back.as_mut() {
                if let Some(column_index) = back.next_back() {
                    return Some((self.back_row, column_index));
                }

                if self.back_row > self.next_row {
                    self.back_row -= CSR::RowIndex::one();
                    if self.back_row == self.next_row {
                        self.back = None;
                    } else {
                        self.back = Some(self.csr2d.sparse_row(self.back_row));
                    }
                } else {
                    self.back = None;
                }
            } else {
                return self
                    .next
                    .as_mut()
                    .and_then(DoubleEndedIterator::next_back)
                    .map(|column_index| (self.next_row, column_index));
            }
        }
    }
}

impl<'a, CSR: SparseMatrix2D> From<&'a CSR> for CSR2DView<'a, CSR> {
    #[inline]
    fn from(csr2d: &'a CSR) -> Self {
        let next_row = CSR::RowIndex::zero();
        let mut back_row = CSR::RowIndex::zero();
        let has_rows = next_row < csr2d.number_of_rows();
        let next = has_rows.then(|| csr2d.sparse_row(next_row));
        let back = if has_rows {
            back_row = csr2d.number_of_rows() - CSR::RowIndex::one();
            (next_row < back_row).then(|| csr2d.sparse_row(back_row))
        } else {
            None
        };
        Self { csr2d, next_row, back_row, next, back }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::traits::{MatrixMut, SparseMatrixMut};

    type TestCSR2D = CSR2D<usize, usize, usize>;

    fn sample_csr() -> TestCSR2D {
        let mut csr = TestCSR2D::with_sparse_shape((3, 4));
        MatrixMut::add(&mut csr, (0, 1)).expect("insert (0,1)");
        MatrixMut::add(&mut csr, (0, 3)).expect("insert (0,3)");
        MatrixMut::add(&mut csr, (2, 0)).expect("insert (2,0)");
        csr
    }

    #[test]
    fn test_view_forward_backward_and_len() {
        let csr = sample_csr();
        let mut view = CSR2DView::from(&csr);

        assert_eq!(view.len(), 3);
        assert_eq!(view.next(), Some((0, 1)));
        assert_eq!(view.next_back(), Some((2, 0)));
        assert_eq!(view.next(), Some((0, 3)));
        assert_eq!(view.next(), None);
        assert_eq!(view.next_back(), None);
    }

    #[test]
    fn test_view_single_row_back_falls_through_to_front_iter() {
        let mut csr = TestCSR2D::with_sparse_shape((1, 5));
        MatrixMut::add(&mut csr, (0, 2)).expect("insert (0,2)");
        MatrixMut::add(&mut csr, (0, 4)).expect("insert (0,4)");

        let mut view = CSR2DView::from(&csr);
        assert_eq!(view.len(), 2);
        assert_eq!(view.next_back(), Some((0, 4)));
        assert_eq!(view.len(), 1);
        assert_eq!(view.next(), Some((0, 2)));
        assert_eq!(view.next_back(), None);
    }

    #[test]
    fn test_view_manual_state_hits_back_none_branch() {
        let csr = TestCSR2D::with_sparse_shape((1, 1));
        let mut view = CSR2DView {
            csr2d: &csr,
            next_row: 1,
            back_row: 0,
            next: Some(csr.sparse_row(0)),
            back: Some(csr.sparse_row(0)),
        };

        assert_eq!(view.next_back(), None);
        assert!(view.back.is_none());
    }

    #[test]
    fn test_view_empty_matrix() {
        let csr = TestCSR2D::with_sparse_shape((0, 0));
        let mut view = CSR2DView::from(&csr);

        assert_eq!(view.len(), 0);
        assert_eq!(view.next(), None);
        assert_eq!(view.next_back(), None);
    }
}
