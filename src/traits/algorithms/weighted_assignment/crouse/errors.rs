//! Error types for the Crouse rectangular LAPJV algorithm.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur during Crouse rectangular LAPJV execution.
pub enum CrouseError {
    /// The matrix contains zero values.
    #[error("The matrix contains zero values.")]
    ZeroValues,
    /// The matrix contains negative values.
    #[error("The matrix contains negative values.")]
    NegativeValues,
    /// The matrix contains non-finite values.
    #[error("The matrix contains non-finite values.")]
    NonFiniteValues,
    /// A value is greater than or equal to max_cost.
    #[error("A value is greater than or equal to max_cost.")]
    ValueTooLarge,
    /// max_cost is not finite.
    #[error("max_cost is not a finite number.")]
    MaximalCostNotFinite,
    /// max_cost is not positive.
    #[error("max_cost is not a positive number.")]
    MaximalCostNotPositive,
}
