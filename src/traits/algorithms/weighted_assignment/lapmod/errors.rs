//! Error types for the LAPMOD algorithm.

use core::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors that can occur during the execution of the LAPMOD algorithm.
pub enum LAPMODError {
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
    /// The matrix contains a value larger than or equal to the maximum cost.
    ValueTooLarge,
    /// The provided maximal cost is not a finite number.
    MaximalCostNotFinite,
    /// The provided maximal cost is not a positive number.
    MaximalCostNotPositive,
    /// The sparse structure has no perfect matching (some row has no edges,
    /// or the bipartite graph admits no perfect matching).
    InfeasibleAssignment,
}

impl Display for LAPMODError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LAPMODError::NonSquareMatrix => write!(f, "The matrix is not square."),
            LAPMODError::EmptyMatrix => write!(f, "The matrix is empty."),
            LAPMODError::ZeroValues => write!(f, "The matrix contains zero values."),
            LAPMODError::NegativeValues => write!(f, "The matrix contains negative values."),
            LAPMODError::NonFiniteValues => {
                write!(f, "The matrix contains non-finite values.")
            }
            LAPMODError::ValueTooLarge => {
                write!(f, "The matrix contains a value larger than or equal to the maximum cost.")
            }
            LAPMODError::MaximalCostNotFinite => {
                write!(f, "The provided maximal cost is not a finite number.")
            }
            LAPMODError::MaximalCostNotPositive => {
                write!(f, "The provided maximal cost is not a positive number.")
            }
            LAPMODError::InfeasibleAssignment => {
                write!(f, "The sparse structure has no perfect matching (infeasible assignment).")
            }
        }
    }
}

impl core::error::Error for LAPMODError {}
