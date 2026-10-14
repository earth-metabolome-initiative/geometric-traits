//! Grassmann-Taksar-Heyman stationary distribution solver for dense
//! row-stochastic matrices.
//!
//! Given a dense square row-stochastic matrix **P**, the GTH algorithm computes
//! a stationary distribution **pi** such that `pi P = pi` and `sum(pi) = 1`.
//! The implementation here follows the dense elimination style used by the
//! QuantEcon reference implementation.
//!
//! # Complexity
//!
//! The dense GTH elimination runs in **O(n^3)** time and uses **O(n^2)** space
//! for a working copy of the matrix plus **O(n)** space for the stationary
//! vector.
//!
//! # Reducible chains
//!
//! For irreducible chains the stationary distribution is unique. For reducible
//! chains, GTH still returns one valid stationary distribution, but uniqueness
//! is no longer guaranteed.
//!
//! # Workspace reuse
//!
//! [`Gth::gth()`] allocates a dense working copy on each call. For repeated
//! solves, reuse a [`GthWorkspace`] via [`Gth::gth_with_workspace()`] to avoid
//! repeated buffer allocation.
//!
//! # Reference
//!
//! Grassmann, W. K., Taksar, M. I., & Heyman, D. P. (1985).
//! *Regenerative Analysis and Steady State Distributions for Markov Chains*.
//! Operations Research, 33(5), 1107-1116.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use crate::traits::{DenseValuedMatrix2D, Finite, Number};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the dense GTH stationary-distribution solver.
#[derive(Debug, Clone, PartialEq)]
pub struct GthConfig {
    /// Tolerance used when validating row-stochastic input.
    ///
    /// Each entry must be exactly nonnegative, and each row sum must differ
    /// from `1.0` by at most this tolerance.
    pub stochastic_tolerance: f64,
}

impl Default for GthConfig {
    #[inline]
    fn default() -> Self {
        Self { stochastic_tolerance: 1e-12 }
    }
}

/// Reusable workspace for repeated dense GTH solves.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GthWorkspace {
    work: Vec<f64>,
}

impl GthWorkspace {
    /// Creates an empty workspace.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a workspace pre-allocated for a matrix of the given order.
    #[must_use]
    #[inline]
    pub fn with_capacity(order: usize) -> Self {
        Self { work: Vec::with_capacity(order.saturating_mul(order)) }
    }

    /// Returns the backing buffer capacity in scalar values.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.work.capacity()
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of the dense GTH stationary-distribution solver.
#[derive(Debug, Clone, PartialEq)]
pub struct GthResult {
    stationary: Vec<f64>,
    order: usize,
}

impl GthResult {
    /// Returns the stationary distribution.
    #[must_use]
    #[inline]
    pub fn stationary(&self) -> &[f64] {
        &self.stationary
    }

    /// Returns the matrix order.
    #[must_use]
    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }
}

// ============================================================================
// Error
// ============================================================================

/// Errors that can occur while computing a stationary distribution with GTH.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GthError {
    /// The matrix must be square.
    #[error("The matrix must be square, but has {rows} rows and {columns} columns.")]
    NonSquareMatrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// The matrix is empty.
    #[error("The matrix is empty.")]
    EmptyMatrix,
    /// A matrix entry is not finite.
    #[error("Found a non-finite value at ({row}, {column}).")]
    NonFiniteValue {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// A matrix entry is negative.
    #[error("Found a negative value at ({row}, {column}).")]
    NegativeValue {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// A row sum is not approximately one.
    #[error("Row {row} is not stochastic: row sum is {row_sum}.")]
    NonStochasticRow {
        /// Row index.
        row: usize,
        /// Observed row sum.
        row_sum: f64,
    },
    /// The stochastic tolerance must be finite and strictly positive.
    #[error("The stochastic tolerance must be finite and strictly positive.")]
    InvalidStochasticTolerance,
    /// The elimination produced an invalid intermediate value.
    #[error(
        "The GTH solver encountered a numerical breakdown during {stage} at index {index}: {value}."
    )]
    NumericalBreakdown {
        /// Which phase failed.
        stage: GthBreakdownStage,
        /// Pivot or active index associated with the failure.
        index: usize,
        /// Problematic value.
        value: f64,
    },
}

/// Phase of the dense GTH solve where a numerical breakdown occurred.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GthBreakdownStage {
    /// The elimination scale computed for a pivot row was invalid.
    EliminationScale,
    /// The final stationary mass from back-substitution could not be
    /// normalized.
    StationaryMass,
}

impl core::fmt::Display for GthBreakdownStage {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EliminationScale => f.write_str("elimination"),
            Self::StationaryMass => f.write_str("normalization"),
        }
    }
}

// ============================================================================
// Private helpers
// ============================================================================

fn validate_config(config: &GthConfig) -> Result<(), GthError> {
    if !config.stochastic_tolerance.is_finite() || config.stochastic_tolerance <= 0.0 {
        return Err(GthError::InvalidStochasticTolerance);
    }
    Ok(())
}

fn read_row_stochastic_matrix_into<M>(
    matrix: &M,
    tolerance: f64,
    flat: &mut Vec<f64>,
) -> Result<usize, GthError>
where
    M: DenseValuedMatrix2D,
    M::Value: Number + ToPrimitive + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
    let rows: usize = matrix.number_of_rows().as_();
    let columns: usize = matrix.number_of_columns().as_();
    if rows != columns {
        return Err(GthError::NonSquareMatrix { rows, columns });
    }
    if rows == 0 {
        return Err(GthError::EmptyMatrix);
    }

    let order = rows;
    let required_len = order.saturating_mul(order);
    flat.clear();
    if flat.capacity() < required_len {
        flat.reserve(required_len - flat.capacity());
    }

    for row_id in matrix.row_indices() {
        let row = row_id.as_();
        let mut row_sum = 0.0;
        for (column_id, value) in matrix.column_indices().zip(matrix.row_values(row_id)) {
            let column = column_id.as_();
            if !value.is_finite() {
                return Err(GthError::NonFiniteValue { row, column });
            }
            let value = value.to_f64().ok_or(GthError::NonFiniteValue { row, column })?;
            if !value.is_finite() {
                return Err(GthError::NonFiniteValue { row, column });
            }
            if value < 0.0 {
                return Err(GthError::NegativeValue { row, column });
            }
            row_sum += value;
            flat.push(value);
        }

        if !row_sum.is_finite() || (row_sum - 1.0).abs() > tolerance {
            return Err(GthError::NonStochasticRow { row, row_sum });
        }
    }

    Ok(order)
}

fn gth_solve(work: &mut [f64], order: usize) -> Result<Vec<f64>, GthError> {
    let mut active = order;
    let mut stationary = vec![0.0; order];

    for pivot in 0..(order.saturating_sub(1)) {
        let row_offset = pivot * order;
        let scale: f64 = ((pivot + 1)..active).map(|column| work[row_offset + column]).sum();
        if !scale.is_finite() || scale < 0.0 {
            return Err(GthError::NumericalBreakdown {
                stage: GthBreakdownStage::EliminationScale,
                index: pivot,
                value: scale,
            });
        }
        if scale == 0.0 {
            active = pivot + 1;
            break;
        }

        for row in (pivot + 1)..active {
            work[row * order + pivot] /= scale;
        }

        for row in (pivot + 1)..active {
            let factor = work[row * order + pivot];
            if factor == 0.0 {
                continue;
            }
            let row_offset = row * order;
            for column in (pivot + 1)..active {
                work[row_offset + column] += factor * work[pivot * order + column];
            }
        }
    }

    stationary[active - 1] = 1.0;
    for pivot in (0..(active - 1)).rev() {
        stationary[pivot] =
            ((pivot + 1)..active).map(|row| stationary[row] * work[row * order + pivot]).sum();
    }

    let normalizer: f64 = stationary[..active].iter().sum();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(GthError::NumericalBreakdown {
            stage: GthBreakdownStage::StationaryMass,
            index: active.saturating_sub(1),
            value: normalizer,
        });
    }
    for value in &mut stationary[..active] {
        *value /= normalizer;
    }

    Ok(stationary)
}

// ============================================================================
// Public trait
// ============================================================================

/// Trait providing the dense GTH stationary-distribution algorithm.
///
/// The input matrix must already be a dense square row-stochastic matrix.
/// The algorithm does not normalize arbitrary nonnegative weights internally,
/// and rejects any negative entry exactly.
///
/// # Complexity
///
/// O(n^3) time, O(n^2) space.
///
/// # Examples
///
/// ```
/// use geometric_traits::{impls::VecMatrix2D, prelude::*};
///
/// let matrix = VecMatrix2D::new(2, 2, vec![0.10, 0.90, 0.60, 0.40]);
/// let result = matrix.gth(&GthConfig::default()).unwrap();
///
/// assert!((result.stationary()[0] - 0.4).abs() < 1e-10);
/// assert!((result.stationary()[1] - 0.6).abs() < 1e-10);
/// ```
pub trait Gth: DenseValuedMatrix2D + Sized
where
    Self::Value: Number + ToPrimitive + Finite,
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Computes one stationary distribution while reusing a caller-provided
    /// working buffer.
    ///
    /// This avoids repeated allocation of the dense row-major work matrix when
    /// solving multiple problems of similar size.
    ///
    /// # Errors
    ///
    /// Returns a [`GthError`] if the input is not square, is empty, contains
    /// non-finite or negative values, is not row-stochastic within the
    /// configured tolerance, or if the elimination encounters a numerical
    /// breakdown.
    #[inline]
    fn gth_with_workspace(
        &self,
        config: &GthConfig,
        workspace: &mut GthWorkspace,
    ) -> Result<GthResult, GthError> {
        validate_config(config)?;
        let order = read_row_stochastic_matrix_into(
            self,
            config.stochastic_tolerance,
            &mut workspace.work,
        )?;
        let stationary = gth_solve(&mut workspace.work, order)?;
        Ok(GthResult { stationary, order })
    }

    /// Computes one stationary distribution of a dense row-stochastic matrix.
    ///
    /// # Errors
    ///
    /// Returns a [`GthError`] if the input is not square, is empty, contains
    /// non-finite or negative values, is not row-stochastic within the
    /// configured tolerance, or if the elimination encounters a numerical
    /// breakdown.
    #[inline]
    fn gth(&self, config: &GthConfig) -> Result<GthResult, GthError> {
        let mut workspace = GthWorkspace::default();
        self.gth_with_workspace(config, &mut workspace)
    }
}

impl<M: DenseValuedMatrix2D> Gth for M
where
    M::Value: Number + ToPrimitive + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec;
    use core::{
        num::ParseIntError,
        ops::{Add, Div, Mul, Rem, Sub},
    };

    use num_traits::{Bounded, Num, One, Zero};

    use super::*;
    use crate::impls::VecMatrix2D;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct InfiniteF64(u8);

    impl Add for InfiniteF64 {
        type Output = Self;

        #[inline]
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub for InfiniteF64 {
        type Output = Self;

        #[inline]
        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl Mul for InfiniteF64 {
        type Output = Self;

        #[inline]
        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }

    impl Div for InfiniteF64 {
        type Output = Self;

        #[inline]
        fn div(self, rhs: Self) -> Self::Output {
            Self(self.0 / rhs.0)
        }
    }

    impl Rem for InfiniteF64 {
        type Output = Self;

        #[inline]
        fn rem(self, rhs: Self) -> Self::Output {
            Self(self.0 % rhs.0)
        }
    }

    impl core::ops::AddAssign for InfiniteF64 {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }

    impl core::ops::SubAssign for InfiniteF64 {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }

    impl Zero for InfiniteF64 {
        #[inline]
        fn zero() -> Self {
            Self(0)
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }

    impl One for InfiniteF64 {
        #[inline]
        fn one() -> Self {
            Self(1)
        }
    }

    impl Bounded for InfiniteF64 {
        #[inline]
        fn min_value() -> Self {
            Self(u8::MIN)
        }

        #[inline]
        fn max_value() -> Self {
            Self(u8::MAX)
        }
    }

    impl Num for InfiniteF64 {
        type FromStrRadixErr = ParseIntError;

        #[inline]
        fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            u8::from_str_radix(str, radix).map(Self)
        }
    }

    impl ToPrimitive for InfiniteF64 {
        #[inline]
        fn to_i64(&self) -> Option<i64> {
            Some(i64::from(self.0))
        }

        #[inline]
        fn to_u64(&self) -> Option<u64> {
            Some(u64::from(self.0))
        }

        #[inline]
        fn to_f64(&self) -> Option<f64> {
            Some(f64::INFINITY)
        }
    }

    impl Finite for InfiniteF64 {
        #[inline]
        fn is_finite(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_default_config_uses_documented_tolerance() {
        assert!((GthConfig::default().stochastic_tolerance - 1e-12).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_finite_after_to_f64_conversion_is_rejected() {
        let matrix = VecMatrix2D::new(1, 1, vec![InfiniteF64(1)]);
        let error = matrix.gth(&GthConfig::default()).unwrap_err();
        assert_eq!(error, GthError::NonFiniteValue { row: 0, column: 0 });
    }

    #[test]
    fn test_stationary_mass_breakdown_is_reported_by_private_solver_guard() {
        // This malformed work matrix cannot arise from validated stochastic
        // input, but it exercises the defensive normalizer check directly.
        let mut work = vec![0.0, 1.0, -1.0, 0.0];
        let error = gth_solve(&mut work, 2).unwrap_err();
        assert_eq!(
            error,
            GthError::NumericalBreakdown {
                stage: GthBreakdownStage::StationaryMass,
                index: 1,
                value: 0.0,
            }
        );
    }

    #[test]
    fn test_elimination_breakdown_is_reported_for_negative_scale() {
        let mut work = vec![0.0, -1.0, 1.0, 0.0];
        let error = gth_solve(&mut work, 2).unwrap_err();
        assert_eq!(
            error,
            GthError::NumericalBreakdown {
                stage: GthBreakdownStage::EliminationScale,
                index: 0,
                value: -1.0,
            }
        );
    }
}
