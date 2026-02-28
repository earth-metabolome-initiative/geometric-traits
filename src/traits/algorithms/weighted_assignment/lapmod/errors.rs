//! Error types for the LAPMOD algorithm.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Errors that can occur during the execution of the LAPMOD algorithm.
pub enum LAPMODError {
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
    /// The matrix contains a value larger than or equal to the maximum cost.
    #[error("The matrix contains a value larger than or equal to the maximum cost.")]
    ValueTooLarge,
    /// The provided maximal cost is not a finite number.
    #[error("The provided maximal cost is not a finite number.")]
    MaximalCostNotFinite,
    /// The provided maximal cost is not a positive number.
    #[error("The provided maximal cost is not a positive number.")]
    MaximalCostNotPositive,
    /// The sparse structure has no perfect matching (some row has no edges,
    /// or the bipartite graph admits no perfect matching).
    #[error("The sparse structure has no perfect matching (infeasible assignment).")]
    InfeasibleAssignment,
}
