//! Hungarian (Kuhn-Munkres) algorithm for the linear assignment problem.
//!
//! This is the classical O(n³) baseline solver. Unlike LAPJV it has no
//! heuristic initialization phases — column duals start at zero and every
//! row is augmented via Dijkstra-style shortest-path search.
use alloc::{vec, vec::Vec};

mod inner;

use core::fmt::Debug;

use inner::HungarianInner;

use super::{
    LAPError,
    lap_error::{
        sparse_padded_lap_impl, validate_lap_entry_costs, validate_sparse_lap_entry_costs,
    },
};
use crate::{
    impls::PaddedMatrix2D,
    traits::{DenseValuedMatrix2D, Finite, Number, SparseValuedMatrix2D, TotalOrd, TryFromUsize},
};

/// Trait defining the Hungarian algorithm for solving the Weighted Assignment
/// Problem on dense square matrices.
pub trait Hungarian: DenseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite + TotalOrd,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using the Hungarian algorithm.
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
    /// - The value type is non-fractional
    ///   (`LAPError::NonFractionalValueTypeUnsupported`)
    /// - `max_cost` is not a finite number (`LAPError::MaximalCostNotFinite`)
    /// - `max_cost` is not positive (`LAPError::MaximalCostNotPositive`)
    /// - The matrix is not square (`LAPError::NonSquareMatrix`)
    /// - The matrix contains zero values (`LAPError::ZeroValues`)
    /// - The matrix contains negative values (`LAPError::NegativeValues`)
    /// - The matrix contains non-finite values (`LAPError::NonFiniteValues`)
    /// - The matrix contains a value larger than the maximum cost
    ///   (`LAPError::ValueTooLarge`)
    #[inline]
    fn hungarian(
        &self,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPError>
    where
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
    {
        validate_lap_entry_costs(max_cost)?;

        let mut inner = HungarianInner::new(self, max_cost)?;
        inner.augmentation();

        Ok(inner.into_assignments())
    }
}

impl<M: DenseValuedMatrix2D> Hungarian for M
where
    M::Value: Number + Finite + TotalOrd,
    M::ColumnIndex: TryFromUsize,
{
}

/// Trait defining the Hungarian algorithm for solving the Weighted Assignment
/// Problem, adapted for the Sparse Matrix type.
pub trait SparseHungarian: SparseValuedMatrix2D + Sized
where
    Self::Value: Number,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using the Hungarian algorithm on a
    /// sparse matrix.
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
    /// - The value type is non-fractional
    ///   (`LAPError::NonFractionalValueTypeUnsupported`)
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
    /// let mut assignment = csr.sparse_hungarian(900.0, 1000.0).expect("Hungarian failed");
    /// assignment.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    /// assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
    /// ```
    #[inline]
    fn sparse_hungarian(
        &self,
        padding_cost: Self::Value,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPError>
    where
        Self::Value: Finite + TotalOrd,
        <<Self as crate::traits::Matrix2D>::ColumnIndex as TryFrom<usize>>::Error: Debug,
    {
        sparse_padded_lap_impl!(self, padding_cost, max_cost, hungarian)
    }
}

impl<M: SparseValuedMatrix2D> SparseHungarian for M
where
    M::Value: Number,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}
