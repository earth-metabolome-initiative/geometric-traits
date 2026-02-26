//! Submodule providing a native sparse implementation of LAPMOD.
//!
//! **LAPMOD** (Volgenant, *Computers & Operations Research* 23, 917–932, 1996)
//! works directly on the CSR sparse structure as the "core".  Unlike
//! [`SparseLAPJV`](super::SparseLAPJV), it never allocates or fills a dense
//! `n × n` matrix, so its memory usage is O(|E|) instead of O(n²).
use alloc::vec::Vec;

mod errors;
mod inner;

use core::fmt::Debug;

pub use errors::LAPMODError;
use inner::LapmodInner;
use num_traits::Zero;

use super::LAPError;
use crate::{
    impls::PaddedMatrix2D,
    traits::{Finite, IntoUsize, Number, SparseValuedMatrix2D, TotalOrd, TryFromUsize},
};

/// Trait providing the LAPMOD algorithm for solving the Weighted Assignment
/// Problem directly over a sparse valued matrix.
///
/// Unlike [`SparseLAPJV`](super::SparseLAPJV), no `padding_cost` parameter
/// is needed.  Missing entries are treated as implicit ∞ and never accessed.
pub trait LAPMOD: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite + TotalOrd,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the optimal weighted assignment using the LAPMOD algorithm.
    ///
    /// The matrix must be **square** and all sparse values must be
    /// **positive** and **finite**, and strictly less than `max_cost`.
    ///
    /// # Arguments
    ///
    /// * `max_cost`: An upper bound on all edge costs.  Must be positive and
    ///   finite.
    ///
    /// # Returns
    ///
    /// A vector of `(row, column)` pairs forming a perfect matching, or an
    /// error if the input is invalid or no perfect matching exists in the
    /// sparse structure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_cost` is not finite ([`LAPMODError::MaximalCostNotFinite`])
    /// - `max_cost` is not positive ([`LAPMODError::MaximalCostNotPositive`])
    /// - The matrix is not square ([`LAPMODError::NonSquareMatrix`])
    /// - Any edge cost is zero ([`LAPMODError::ZeroValues`])
    /// - Any edge cost is negative ([`LAPMODError::NegativeValues`])
    /// - Any edge cost is non-finite ([`LAPMODError::NonFiniteValues`])
    /// - Any edge cost ≥ `max_cost` ([`LAPMODError::ValueTooLarge`])
    /// - The sparse graph has no perfect matching
    ///   ([`LAPMODError::InfeasibleAssignment`])
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// let csr: ValuedCSR2D<u8, u8, u8, f64> =
    ///     ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 1.0, 6.0], [7.0, 8.0, 1.0]])
    ///         .expect("Failed to create CSR matrix");
    ///
    /// let mut assignment = csr.lapmod(1000.0).expect("LAPMOD failed");
    /// assignment.sort_unstable_by_key(|&(r, c)| (r, c));
    /// assert_eq!(assignment, vec![(0, 0), (1, 1), (2, 2)]);
    /// ```
    fn lapmod(
        &self,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPMODError>
    where
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
        <Self::RowIndex as TryFrom<usize>>::Error: Debug,
    {
        if !max_cost.is_finite() {
            return Err(LAPMODError::MaximalCostNotFinite);
        }

        if max_cost <= Self::Value::zero() {
            return Err(LAPMODError::MaximalCostNotPositive);
        }

        let n_rows = self.number_of_rows().into_usize();
        let n_cols = self.number_of_columns().into_usize();

        if n_rows != n_cols {
            return Err(LAPMODError::NonSquareMatrix);
        }

        if n_rows == 0 {
            return Ok(Vec::new());
        }

        let mut inner = LapmodInner::new(self, max_cost)?;

        inner.column_reduction_sparse()?;
        inner.reduction_transfer_sparse();

        // Two passes of augmenting row reduction (same as LAPJV).
        inner.augmenting_row_reduction_sparse();
        inner.augmenting_row_reduction_sparse();

        inner.augmentation_sparse()?;

        Ok(inner.into())
    }
}

impl<M: SparseValuedMatrix2D> LAPMOD for M
where
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}

/// Trait defining the LAPMOD algorithm for solving the Weighted Assignment
/// Problem, adapted for sparse rectangular matrices by padding to square.
pub trait SparseLAPMOD: SparseValuedMatrix2D + Sized
where
    Self::Value: Number,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using LAPMOD on a padded sparse view.
    ///
    /// # Arguments
    ///
    /// * `padding_cost`: The cost of padding the matrix to make it square.
    /// * `max_cost`: The upper bound for the cost of the assignment.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing row/column assignments in the original
    /// matrix coordinates. Any assignment that uses imputed padding values is
    /// filtered out.
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
    /// - The padded matrix cannot be represented (`LAPError::NonSquareMatrix`)
    /// - Matrix values violate LAPMOD input requirements
    fn sparse_lapmod(
        &self,
        padding_cost: Self::Value,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPError>
    where
        Self::Value: Finite + TotalOrd,
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
        <Self::RowIndex as TryFrom<usize>>::Error: Debug,
    {
        if !padding_cost.is_finite() {
            return Err(LAPError::PaddingValueNotFinite);
        }
        if padding_cost <= Self::Value::zero() {
            return Err(LAPError::PaddingValueNotPositive);
        }
        if padding_cost >= max_cost {
            return Err(LAPError::ValueTooLarge);
        }
        if self.is_empty() {
            return Ok(vec![]);
        }

        debug_assert!(
            self.max_sparse_value().unwrap() < padding_cost,
            "The maximum value in the matrix ({:?}) is greater than the padding cost ({:?}).",
            self.max_sparse_value().unwrap(),
            padding_cost
        );

        let padding: PaddedMatrix2D<&'_ Self, _> =
            PaddedMatrix2D::new(self, |_| padding_cost).map_err(|_| LAPError::NonSquareMatrix)?;

        let assignment = padding.lapmod(max_cost).map_err(LAPError::from)?;

        Ok(assignment
            .into_iter()
            .filter(|&(row_index, column_index)| !padding.is_imputed((row_index, column_index)))
            .collect())
    }
}

impl<M: SparseValuedMatrix2D> SparseLAPMOD for M
where
    M::Value: Number,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}

impl From<LAPMODError> for LAPError {
    fn from(error: LAPMODError) -> Self {
        match error {
            LAPMODError::NonSquareMatrix => LAPError::NonSquareMatrix,
            LAPMODError::EmptyMatrix => LAPError::EmptyMatrix,
            LAPMODError::ZeroValues => LAPError::ZeroValues,
            LAPMODError::NegativeValues => LAPError::NegativeValues,
            LAPMODError::NonFiniteValues => LAPError::NonFiniteValues,
            LAPMODError::ValueTooLarge => LAPError::ValueTooLarge,
            LAPMODError::MaximalCostNotFinite => LAPError::MaximalCostNotFinite,
            LAPMODError::MaximalCostNotPositive => LAPError::MaximalCostNotPositive,
            LAPMODError::InfeasibleAssignment => LAPError::InfeasibleAssignment,
        }
    }
}
