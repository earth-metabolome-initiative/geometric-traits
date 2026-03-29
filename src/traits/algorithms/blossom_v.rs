//! Submodule providing the `BlossomV` trait for minimum-cost perfect matching
//! in general (non-bipartite) weighted graphs using the Edmonds blossom
//! algorithm with variable dual updates and priority queues.
//! This is a clean-room implementation based on the paper:
//!
//! > V. Kolmogorov, "Blossom V: A new implementation of a minimum cost perfect
//! > matching algorithm," *Mathematical Programming Computation* 1(1):43–67,
//! > 2009.
//!
//! The algorithm combines Cook and Rohe's "variable δ" dual update approach
//! with pairing-heap priority queues, maintaining an auxiliary graph whose
//! nodes correspond to alternating trees. It uses greedy initialization.

use alloc::vec::Vec;
use core::fmt;

mod inner;
mod pairing_heap;

use inner::BlossomVState;
use num_traits::AsPrimitive;

use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

type MatchingResult<R, C> = Result<Vec<(R, C)>, BlossomVError>;

/// Error type for the Blossom V algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlossomVError {
    /// The input graph does not contain a perfect matching of finite cost.
    NoPerfectMatching,
}

impl fmt::Display for BlossomVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlossomVError::NoPerfectMatching => {
                write!(f, "No perfect matching exists in the input graph")
            }
        }
    }
}

/// Minimum-cost perfect matching in general weighted graphs via the Edmonds
/// blossom algorithm with variable dual updates and priority queues.
///
/// # Input
///
/// The matrix represents an undirected weighted graph where each entry
/// `(i, j)` with value `w` means there is an edge between vertices `i` and
/// `j` with cost `w`. Costs may be negative. The matrix must be **square**
/// and **symmetric**.
///
/// A perfect matching of finite cost must exist; if not, the algorithm returns
/// [`BlossomVError::NoPerfectMatching`].
///
/// # Output
///
/// On success, returns a vector of `(row, column)` pairs with `row < column`
/// representing the edges in the minimum-cost perfect matching.
///
/// # Complexity
///
/// O(V² · E) time (believed; same as Blossom IV), O(V + E) space.
///
/// # References
///
/// - V. Kolmogorov, "Blossom V: A new implementation of a minimum cost perfect
///   matching algorithm," *MPC* 1(1):43–67, 2009.
/// - W. Cook, A. Rohe, "Computing minimum-weight perfect matchings," *INFORMS
///   J. Computing* 11(2):138–148, 1999.
pub trait BlossomV: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + AsPrimitive<i64>,
    Self::RowIndex: PositiveInteger,
    Self::ColumnIndex: PositiveInteger,
{
    /// Computes a minimum-cost perfect matching.
    ///
    /// # Errors
    ///
    /// Returns [`BlossomVError::NoPerfectMatching`] if no perfect matching
    /// exists.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or has an odd number of vertices.
    #[inline]
    fn blossom_v(&self) -> MatchingResult<Self::RowIndex, Self::ColumnIndex> {
        let n_rows: usize = self.number_of_rows().as_();
        let n_cols: usize = self.number_of_columns().as_();
        assert!(n_rows == n_cols, "BlossomV requires a square matrix, got {n_rows} x {n_cols}");
        assert!(
            n_rows % 2 == 0,
            "BlossomV requires an even number of vertices for a perfect matching, got {n_rows}"
        );
        BlossomVState::new(self).solve()
    }
}

/// Benchmark-only entrypoint that skips the Rust-only support-feasibility
/// guards and runs the main Blossom V solver directly.
///
/// # Safety contract
///
/// Callers must only use this on support-feasible graphs. The default
/// [`BlossomV::blossom_v`] entrypoint should be preferred everywhere else.
#[doc(hidden)]
#[inline]
pub fn blossom_v_unchecked_support_feasible<M>(
    matrix: &M,
) -> MatchingResult<M::RowIndex, M::ColumnIndex>
where
    M: SparseValuedMatrix2D + Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    let n_rows: usize = matrix.number_of_rows().as_();
    let n_cols: usize = matrix.number_of_columns().as_();
    assert!(n_rows == n_cols, "BlossomV requires a square matrix, got {n_rows} x {n_cols}");
    assert!(
        n_rows % 2 == 0,
        "BlossomV requires an even number of vertices for a perfect matching, got {n_rows}"
    );
    BlossomVState::new(matrix).solve_unchecked_support_feasible()
}

impl<M: SparseValuedMatrix2D> BlossomV for M
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
}
