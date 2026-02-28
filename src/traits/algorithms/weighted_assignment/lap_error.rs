//! Shared error type for sparse LAP wrappers (`SparseLAPJV`, `Jaqaman`).

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur while executing sparse LAP wrapper algorithms.
pub enum LAPError {
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
    /// The sparse structure has no perfect matching.
    #[error("The sparse structure has no perfect matching (infeasible assignment).")]
    InfeasibleAssignment,
}
