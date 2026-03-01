//! Submodule providing an implementation of the LAPJV algorithm.
use alloc::vec::Vec;

pub(crate) mod common;
mod errors;
mod inner;

use core::fmt::Debug;

pub use errors::LAPJVError;
use inner::Inner;
use num_traits::Zero;

use super::{LAPError, lap_error::validate_sparse_wrapper_costs};
use crate::{
    impls::PaddedMatrix2D,
    traits::{DenseValuedMatrix2D, Finite, Number, SparseValuedMatrix2D, TotalOrd, TryFromUsize},
};

/// Trait defining the LAPJV algorithm for solving the Weighted Assignment
/// Problem.
pub trait LAPJV: DenseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite + TotalOrd,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using the LAPJV algorithm.
    ///
    /// # Arguments
    ///
    /// * `max_cost`: The upper bound for the cost of the assignment.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the row and column indices of the
    /// assignment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_cost` is not a finite number (`LAPJVError::MaximalCostNotFinite`)
    /// - `max_cost` is not positive (`LAPJVError::MaximalCostNotPositive`)
    /// - The matrix is not square (`LAPJVError::NonSquareMatrix`)
    /// - The matrix is empty (`LAPJVError::EmptyMatrix`)
    /// - The matrix contains zero values (`LAPJVError::ZeroValues`)
    /// - The matrix contains negative values (`LAPJVError::NegativeValues`)
    /// - The matrix contains non-finite values (`LAPJVError::NonFiniteValues`)
    /// - The matrix contains a value larger than the maximum cost
    ///   (`LAPJVError::ValueTooLarge`)
    fn lapjv(
        &self,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPJVError>
    where
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
    {
        if !max_cost.is_finite() {
            return Err(LAPJVError::MaximalCostNotFinite);
        }

        if max_cost <= Self::Value::zero() {
            return Err(LAPJVError::MaximalCostNotPositive);
        }

        let mut inner = Inner::new(self, max_cost)?;
        inner.column_reduction()?;
        inner.reduction_transfer();

        // We execute TWICE augmenting row reductions.
        inner.augmenting_row_reduction();
        inner.augmenting_row_reduction();

        inner.augmentation();

        Ok(inner.into())
    }
}

impl<M: DenseValuedMatrix2D> LAPJV for M
where
    M::Value: Number + Finite + TotalOrd,
    M::ColumnIndex: TryFromUsize,
{
}

/// Trait defining the LAPJV algorithm for solving the Weighted Assignment
/// Problem, adapted for the Sparse Matrix type.
pub trait SparseLAPJV: SparseValuedMatrix2D + Sized
where
    Self::Value: Number,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using the LAPJV algorithm.
    ///
    /// # Arguments
    ///
    /// * `padding_cost`: The cost of padding the matrix to make it square.
    /// * `max_cost`: The upper bound for the cost of the assignment.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the row and column indices of the
    /// assignment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `padding_cost` is not a finite number
    ///   (`LAPError::PaddingValueNotFinite`)
    /// - `padding_cost` is not positive (`LAPError::PaddingValueNotPositive`)
    /// - `padding_cost` is greater than or equal to `max_cost`
    ///   (`LAPError::ValueTooLarge`)
    /// - `max_cost` is not a finite number (`LAPError::MaximalCostNotFinite`)
    /// - `max_cost` is not positive (`LAPError::MaximalCostNotPositive`)
    /// - The matrix is not square after padding (`LAPError::NonSquareMatrix`)
    /// - The matrix contains values that are greater than the padding cost
    /// - The matrix contains zero, negative or non-finite values
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// let csr: ValuedCSR2D<u8, u8, u8, f64> =
    ///     ValuedCSR2D::try_from([[1.0, 0.5, 10.0], [0.5, 10.0, 20.0], [10.0, 20.0, 0.5]])
    ///         .expect("Failed to create CSR matrix");
    ///
    /// let mut assignment = csr.sparse_lapjv(900.0, 1000.0).expect("LAPjv failed");
    /// assignment.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    /// assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
    /// ```
    fn sparse_lapjv(
        &self,
        padding_cost: Self::Value,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPError>
    where
        Self::Value: Finite + TotalOrd,
        <<Self as crate::traits::Matrix2D>::ColumnIndex as TryFrom<usize>>::Error: Debug,
    {
        validate_sparse_wrapper_costs(padding_cost, max_cost)?;
        if self.is_empty() {
            return Ok(vec![]);
        }

        if self.max_sparse_value().is_some_and(|value| value >= padding_cost) {
            return Err(LAPError::PaddingCostTooSmall);
        }

        let padding: PaddedMatrix2D<&'_ Self, _> =
            PaddedMatrix2D::new(self, |_| padding_cost).map_err(|_| LAPError::NonSquareMatrix)?;
        let assignment = padding.lapjv(max_cost).map_err(LAPError::from)?;

        Ok(assignment
            .into_iter()
            .filter(|&(row_index, column_index)| !padding.is_imputed((row_index, column_index)))
            .collect())
    }
}

impl<M: SparseValuedMatrix2D> SparseLAPJV for M
where
    M::Value: Number,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}

impl From<LAPJVError> for LAPError {
    fn from(error: LAPJVError) -> Self {
        match error {
            LAPJVError::NonSquareMatrix => LAPError::NonSquareMatrix,
            LAPJVError::EmptyMatrix => LAPError::EmptyMatrix,
            LAPJVError::ZeroValues => LAPError::ZeroValues,
            LAPJVError::NegativeValues => LAPError::NegativeValues,
            LAPJVError::NonFiniteValues => LAPError::NonFiniteValues,
            LAPJVError::ValueTooLarge => LAPError::ValueTooLarge,
            LAPJVError::MaximalCostNotFinite => LAPError::MaximalCostNotFinite,
            LAPJVError::MaximalCostNotPositive => LAPError::MaximalCostNotPositive,
            LAPJVError::PaddingValueNotFinite => LAPError::PaddingValueNotFinite,
            LAPJVError::PaddingValueNotPositive => LAPError::PaddingValueNotPositive,
        }
    }
}
