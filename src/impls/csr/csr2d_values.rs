//! Iterator of the sparse coordinates of the M2D matrix.
use core::cmp::Ordering;

use num_traits::{AsPrimitive, One, Zero};

use crate::prelude::*;

/// Iterator of the sparse coordinates of the M2D matrix.
pub struct M2DValues<'a, M: SparseValuedMatrix2D> {
    /// The M matrix.
    matrix: &'a M,
    /// The row index.
    next_row: M::RowIndex,
    /// The end row index.
    back_row: M::RowIndex,
    /// The row associated with the index at the beginning of the iteration.
    next: Option<M::SparseRowValues<'a>>,
    /// The row associated with the index at the end of the iteration.
    back: Option<M::SparseRowValues<'a>>,
}

impl<M: SparseValuedMatrix2D> Iterator for M2DValues<'_, M> {
    type Item = M::Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.next.as_mut()?.next() {
                return Some(value);
            }

            match self.next_row.cmp(&self.back_row) {
                Ordering::Less => {
                    self.next_row += M::RowIndex::one();
                    if self.next_row == self.back_row {
                        self.next = self.back.take();
                    } else {
                        self.next = Some(self.matrix.sparse_row_values(self.next_row));
                    }
                }
                Ordering::Equal | Ordering::Greater => {
                    return self.back.as_mut().and_then(Iterator::next);
                }
            }
        }
    }
}

impl<'matrix, M: SizedSparseMatrix2D + SparseValuedMatrix2D> ExactSizeIterator
    for M2DValues<'matrix, M>
where
    M::SparseRowValues<'matrix>: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        if self.next.is_none() {
            return 0;
        }
        if self.back.is_none() {
            return self.next.as_ref().map_or(0, ExactSizeIterator::len);
        }

        let next_row_rank = self.matrix.rank_row(self.next_row).as_();
        let already_observed_in_next_row =
            self.matrix.number_of_defined_values_in_row(self.next_row).as_()
                - self.next.as_ref().map_or(0, ExactSizeIterator::len);
        let back_row_rank = self.matrix.rank_row(self.back_row).as_();
        let already_observed_in_back_row = self.back.as_ref().map_or(0, |back| {
            self.matrix.number_of_defined_values_in_row(self.back_row).as_() - back.len()
        });
        back_row_rank - next_row_rank - already_observed_in_next_row - already_observed_in_back_row
    }
}

impl<M: SparseValuedMatrix2D> DoubleEndedIterator for M2DValues<'_, M> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(back) = self.back.as_mut() {
                if let Some(value) = back.next_back() {
                    return Some(value);
                }

                if self.back_row > self.next_row {
                    self.back_row -= M::RowIndex::one();
                    if self.back_row == self.next_row {
                        self.back = None;
                    } else {
                        self.back = Some(self.matrix.sparse_row_values(self.back_row));
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

impl<'a, M: SparseValuedMatrix2D> From<&'a M> for M2DValues<'a, M> {
    #[inline]
    fn from(matrix: &'a M) -> Self {
        let next_row = M::RowIndex::zero();
        let mut back_row = M::RowIndex::zero();
        let has_rows = next_row < matrix.number_of_rows();
        let next = has_rows.then(|| matrix.sparse_row_values(next_row));
        let back = if has_rows {
            back_row = matrix.number_of_rows() - M::RowIndex::one();
            (next_row < back_row).then(|| matrix.sparse_row_values(back_row))
        } else {
            None
        };
        Self { matrix, next_row, back_row, next, back }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::traits::{MatrixMut, SparseMatrixMut};

    type TestValuedCSR2D = ValuedCSR2D<usize, usize, usize, i32>;

    fn sample_matrix() -> TestValuedCSR2D {
        let mut matrix = TestValuedCSR2D::with_sparse_shape((3, 4));
        MatrixMut::add(&mut matrix, (0, 1, 10)).expect("insert (0,1)");
        MatrixMut::add(&mut matrix, (0, 3, 30)).expect("insert (0,3)");
        MatrixMut::add(&mut matrix, (2, 0, 5)).expect("insert (2,0)");
        matrix
    }

    #[test]
    fn test_values_forward_backward_and_len() {
        let matrix = sample_matrix();
        let mut values = M2DValues::from(&matrix);

        assert_eq!(values.len(), 2);
        assert_eq!(values.next(), Some(10));
        assert_eq!(values.next_back(), Some(5));
        assert_eq!(values.next(), Some(30));
        assert_eq!(values.next(), None);
        assert_eq!(values.next_back(), None);
    }

    #[test]
    fn test_values_single_row_back_falls_through_to_front_iter() {
        let mut matrix = TestValuedCSR2D::with_sparse_shape((1, 5));
        MatrixMut::add(&mut matrix, (0, 2, 20)).expect("insert (0,2)");
        MatrixMut::add(&mut matrix, (0, 4, 40)).expect("insert (0,4)");

        let mut values = M2DValues::from(&matrix);
        assert_eq!(values.len(), 2);
        assert_eq!(values.next_back(), Some(40));
        assert_eq!(values.len(), 1);
        assert_eq!(values.next(), Some(20));
        assert_eq!(values.next_back(), None);
    }

    #[test]
    fn test_values_empty_matrix() {
        let matrix = TestValuedCSR2D::with_sparse_shape((0, 0));
        let mut values = M2DValues::from(&matrix);

        assert_eq!(values.len(), 0);
        assert_eq!(values.next(), None);
        assert_eq!(values.next_back(), None);
    }

    #[test]
    fn test_values_next_on_single_entry_exhausts_via_equal_branch() {
        let mut matrix = TestValuedCSR2D::with_sparse_shape((1, 3));
        MatrixMut::add(&mut matrix, (0, 1, 11)).expect("insert (0,1)");

        let mut values = M2DValues::from(&matrix);
        assert_eq!(values.next(), Some(11));
        assert_eq!(values.next(), None);
    }

    #[test]
    fn test_values_manual_state_hits_back_none_branch() {
        let mut matrix = TestValuedCSR2D::with_sparse_shape((1, 1));
        MatrixMut::add(&mut matrix, (0, 0, 7)).expect("insert (0,0)");

        let mut values = M2DValues {
            matrix: &matrix,
            next_row: 1,
            back_row: 0,
            next: Some(matrix.sparse_row_values(0)),
            back: Some(matrix.sparse_row_values(0)),
        };
        values.back.as_mut().expect("back iterator exists").next();

        assert_eq!(values.next_back(), Some(7));
        assert_eq!(values.next_back(), None);
        assert!(values.back.is_none());
    }
}
