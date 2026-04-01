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
mod inner;
mod pairing_heap;

use inner::BlossomVState;
use num_traits::AsPrimitive;

use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

type MatchingResult<R, C> = Result<Vec<(R, C)>, BlossomVError>;

/// Error type for the Blossom V algorithm.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BlossomVError {
    /// The input graph does not contain a perfect matching of finite cost.
    #[error("No perfect matching exists in the input graph")]
    NoPerfectMatching,
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
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// type Graph = ValuedCSR2D<usize, usize, usize, i32>;
    ///
    /// let mut graph: Graph = SparseMatrixMut::with_sparse_shaped_capacity((4, 4), 8);
    /// for edge in
    ///     [(0, 1, 1), (0, 2, 9), (1, 0, 1), (1, 3, 9), (2, 0, 9), (2, 3, 1), (3, 1, 9), (3, 2, 1)]
    /// {
    ///     MatrixMut::add(&mut graph, edge).unwrap();
    /// }
    ///
    /// let mut matching = graph.blossom_v().unwrap();
    /// matching.sort_unstable();
    /// assert_eq!(matching, vec![(0, 1), (2, 3)]);
    /// ```
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// type Graph = ValuedCSR2D<usize, usize, usize, i32>;
    ///
    /// let mut graph: Graph = SparseMatrixMut::with_sparse_shaped_capacity((4, 4), 6);
    /// for edge in [(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)] {
    ///     MatrixMut::add(&mut graph, edge).unwrap();
    /// }
    ///
    /// let err = graph.blossom_v().unwrap_err();
    /// assert_eq!(err, BlossomVError::NoPerfectMatching);
    /// ```
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

impl<M: SparseValuedMatrix2D> BlossomV for M
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
}
