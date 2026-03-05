//! Shared error type for LAP algorithms (`LAPJV`, `LAPMOD`, `SparseLAPJV`,
//! `Jaqaman`).
use crate::traits::{Finite, Number, TotalOrd};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur while executing LAP algorithms.
pub enum LAPError {
    /// The value type is non-fractional, which is not supported by LAP
    /// routines.
    #[error("The matrix value type is non-fractional and is not supported by LAP algorithms.")]
    NonFractionalValueTypeUnsupported,
    /// The matrix is not square.
    #[error("The matrix is not square.")]
    NonSquareMatrix,
    /// The matrix is empty.
    #[error("The matrix is empty.")]
    EmptyMatrix,
    /// The matrix contains zero values.
    #[error("The matrix contains zero values.")]
    ZeroValues,
    /// The matrix contains negative values.
    #[error("The matrix contains negative values.")]
    NegativeValues,
    /// The matrix contains non-finite values.
    #[error("The matrix contains non-finite values.")]
    NonFiniteValues,
    /// The matrix contains a value larger than the maximum cost.
    #[error("The matrix contains a value larger than the maximum cost.")]
    ValueTooLarge,
    /// The provided maximal cost is not a finite number.
    #[error("The provided maximal cost is not a finite number.")]
    MaximalCostNotFinite,
    /// The provided maximal cost is not a positive number.
    #[error("The provided maximal cost is not a positive number.")]
    MaximalCostNotPositive,
    /// The provided padding value is not a finite number.
    #[error("The provided padding value is not a finite number.")]
    PaddingValueNotFinite,
    /// The provided padding value is not a positive number.
    #[error("The provided padding value is not a positive number.")]
    PaddingValueNotPositive,
    /// The padding cost is too small relative to the sparse values.
    ///
    /// The diagonal cost extension requires `padding_cost / 2` to be strictly
    /// greater than every sparse value. This typically happens when the
    /// padding cost is computed with an additive offset that vanishes in
    /// floating point at extreme magnitudes.
    #[error(
        "The padding cost is too small: padding_cost / 2 must be strictly greater than the maximum sparse value."
    )]
    PaddingCostTooSmall,
    /// The expanded matrix construction failed due malformed sparse structure
    /// (e.g. duplicate or unsorted coordinates).
    #[error("Failed to build the expanded sparse matrix from the provided sparse structure.")]
    ExpandedMatrixBuildFailed,
    /// Internal index conversion failed while mapping `usize` back to matrix
    /// row/column index types.
    #[error("Internal index conversion failed while processing the sparse wrapper.")]
    IndexConversionFailed,
    /// The sparse structure has no perfect matching.
    #[error("The sparse structure has no perfect matching (infeasible assignment).")]
    InfeasibleAssignment,
}

/// Validates the common `padding_cost`/`max_cost` contract for sparse wrappers.
///
/// Validation order is intentional and shared across wrappers:
/// 1. `padding_cost` finite/positive
/// 2. `max_cost` finite/positive
/// 3. `padding_cost < max_cost`
pub(crate) fn validate_sparse_wrapper_costs<V>(padding_cost: V, max_cost: V) -> Result<(), LAPError>
where
    V: Number + Finite,
{
    if !padding_cost.is_finite() {
        return Err(LAPError::PaddingValueNotFinite);
    }
    if padding_cost <= V::zero() {
        return Err(LAPError::PaddingValueNotPositive);
    }
    if !max_cost.is_finite() {
        return Err(LAPError::MaximalCostNotFinite);
    }
    if max_cost <= V::zero() {
        return Err(LAPError::MaximalCostNotPositive);
    }
    if padding_cost >= max_cost {
        return Err(LAPError::ValueTooLarge);
    }

    Ok(())
}

/// Validates the common `max_cost` contract for LAP entry points.
///
/// Validation order is intentional:
/// 1. `max_cost` finite
/// 2. `max_cost` positive
pub(crate) fn validate_max_cost<V>(max_cost: V) -> Result<(), LAPError>
where
    V: Number + Finite,
{
    if !max_cost.is_finite() {
        return Err(LAPError::MaximalCostNotFinite);
    }
    if max_cost <= V::zero() {
        return Err(LAPError::MaximalCostNotPositive);
    }
    Ok(())
}

/// Validates that the value domain supports fractional arithmetic needed by
/// LAP reduced-cost updates and epsilon constructions.
pub(crate) fn validate_fractional_value_domain<V>() -> Result<(), LAPError>
where
    V: Number + Finite,
{
    let one = V::one();
    let two = one + one;

    if two == V::zero() || one / two == V::zero() {
        return Err(LAPError::NonFractionalValueTypeUnsupported);
    }

    Ok(())
}

/// Validates the common preflight contract for LAP entry points that only
/// require `max_cost`.
///
/// Validation order:
/// 1. fractional value domain support
/// 2. `max_cost` finite/positive
pub(crate) fn validate_lap_entry_costs<V>(max_cost: V) -> Result<(), LAPError>
where
    V: Number + Finite,
{
    validate_fractional_value_domain::<V>()?;
    validate_max_cost(max_cost)
}

/// Validates the common preflight contract for sparse LAP wrappers with
/// `padding_cost` and `max_cost`.
///
/// Validation order:
/// 1. fractional value domain support
/// 2. sparse wrapper cost contract (`padding_cost`, then `max_cost`)
pub(crate) fn validate_sparse_lap_entry_costs<V>(
    padding_cost: V,
    max_cost: V,
) -> Result<(), LAPError>
where
    V: Number + Finite,
{
    validate_fractional_value_domain::<V>()?;
    validate_sparse_wrapper_costs(padding_cost, max_cost)
}

/// Validates a single matrix value against LAP constraints.
///
/// Unified validation order across LAP inner implementations:
/// 1. finite
/// 2. non-zero
/// 3. non-negative
/// 4. strictly smaller than `max_cost`
pub(crate) fn validate_lap_value_against_max<V>(value: V, max_cost: V) -> Result<(), LAPError>
where
    V: Number + Finite + TotalOrd,
{
    if !value.is_finite() {
        return Err(LAPError::NonFiniteValues);
    }
    if value == V::zero() {
        return Err(LAPError::ZeroValues);
    }
    if value < V::zero() {
        return Err(LAPError::NegativeValues);
    }
    if value >= max_cost {
        return Err(LAPError::ValueTooLarge);
    }

    Ok(())
}
