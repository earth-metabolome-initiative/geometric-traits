//! Shared error type for sparse LAP wrappers (`SparseLAPJV`, `SparseLAPMOD`).

use core::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors that can occur while executing sparse LAP wrapper algorithms.
pub enum LAPError {
    /// The matrix is not square.
    NonSquareMatrix,
    /// The matrix is empty.
    EmptyMatrix,
    /// The matrix contains zero values.
    ZeroValues,
    /// The matrix contains negative values.
    NegativeValues,
    /// The matrix contains non-finite values.
    NonFiniteValues,
    /// The matrix contains a value larger than the maximum cost.
    ValueTooLarge,
    /// The provided maximal cost is not a finite number.
    MaximalCostNotFinite,
    /// The provided maximal cost is not a positive number.
    MaximalCostNotPositive,
    /// The provided padding value is not a finite number.
    PaddingValueNotFinite,
    /// The provided padding value is not a positive number.
    PaddingValueNotPositive,
    /// The sparse structure has no perfect matching.
    InfeasibleAssignment,
}

impl Display for LAPError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LAPError::NonSquareMatrix => write!(f, "The matrix is not square."),
            LAPError::EmptyMatrix => write!(f, "The matrix is empty."),
            LAPError::ZeroValues => write!(f, "The matrix contains zero values."),
            LAPError::NegativeValues => write!(f, "The matrix contains negative values."),
            LAPError::NonFiniteValues => write!(f, "The matrix contains non-finite values."),
            LAPError::ValueTooLarge => {
                write!(f, "The matrix contains a value larger than the maximum cost.")
            }
            LAPError::MaximalCostNotFinite => {
                write!(f, "The provided maximal cost is not a finite number.")
            }
            LAPError::MaximalCostNotPositive => {
                write!(f, "The provided maximal cost is not a positive number.")
            }
            LAPError::PaddingValueNotFinite => {
                write!(f, "The provided padding value is not a finite number.")
            }
            LAPError::PaddingValueNotPositive => {
                write!(f, "The provided padding value is not a positive number.")
            }
            LAPError::InfeasibleAssignment => {
                write!(f, "The sparse structure has no perfect matching (infeasible assignment).")
            }
        }
    }
}

impl core::error::Error for LAPError {}
