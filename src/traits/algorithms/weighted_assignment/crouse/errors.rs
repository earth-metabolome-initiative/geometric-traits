//! Error types for the Crouse rectangular LAPJV algorithm.

use core::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors that can occur during Crouse rectangular LAPJV execution.
pub enum CrouseError {
    /// The matrix contains zero values.
    ZeroValues,
    /// The matrix contains negative values.
    NegativeValues,
    /// The matrix contains non-finite values.
    NonFiniteValues,
    /// A value is greater than or equal to max_cost.
    ValueTooLarge,
    /// max_cost is not finite.
    MaximalCostNotFinite,
    /// max_cost is not positive.
    MaximalCostNotPositive,
}

impl Display for CrouseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroValues => write!(f, "The matrix contains zero values."),
            Self::NegativeValues => write!(f, "The matrix contains negative values."),
            Self::NonFiniteValues => write!(f, "The matrix contains non-finite values."),
            Self::ValueTooLarge => {
                write!(f, "A value is greater than or equal to max_cost.")
            }
            Self::MaximalCostNotFinite => {
                write!(f, "max_cost is not a finite number.")
            }
            Self::MaximalCostNotPositive => {
                write!(f, "max_cost is not a positive number.")
            }
        }
    }
}

impl core::error::Error for CrouseError {}
