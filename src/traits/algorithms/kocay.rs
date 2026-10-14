//! Submodule providing the `Kocay` trait for maximum balanced flow
//! in general (non-bipartite) graphs using the Kocay-Stone Balanced Network
//! Search algorithm.
//!
//! Unlike simple maximum matching (all capacities = 1), this computes integer
//! flow per edge where edges have integer capacities and vertices have budgets.
//! The output is a set of flow triples `(i, j, flow)` maximizing total flow.
use alloc::vec::Vec;

mod inner;

use inner::KocayState;
use num_traits::AsPrimitive;

use crate::traits::{PositiveInteger, SparseValuedMatrix2D};

/// Maximum balanced flow in general graphs via the Kocay-Stone Balanced
/// Network Search algorithm.
///
/// # Input
///
/// The matrix represents an undirected graph where each entry `(i, j)` with
/// value `c` means there is an edge between vertices `i` and `j` with
/// capacity `c`. The matrix must be **square** (same number of rows and
/// columns). It should be **symmetric** — non-symmetric input gives
/// unspecified results.
///
/// Each vertex has a **budget** (maximum total flow through that vertex),
/// passed via the `vertex_budgets` parameter.
///
/// # Output
///
/// A vector of `(row, column, flow)` triples with `row < column` and
/// `flow > 0`, representing the assigned flow (bond order) on each edge.
///
/// # References
///
/// - W. Kocay, D. Stone, "An Algorithm for Balanced Flows", *J. Combin. Math.
///   Combin. Comput.*, vol. 19 (1995) pp. 3–31.
/// - W. Kocay, D. Stone, "Balanced network flows", *Bull. Inst. Combin. Appl.*,
///   vol. 7 (1993), pp. 17–32.
pub trait Kocay: SparseValuedMatrix2D + Sized
where
    Self::Value: PositiveInteger,
    Self::RowIndex: PositiveInteger,
    Self::ColumnIndex: PositiveInteger,
{
    /// Computes a maximum balanced flow.
    ///
    /// # Arguments
    ///
    /// * `vertex_budgets` — budget (maximum total flow) per vertex. Must have
    ///   length equal to the matrix order.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The matrix is not square.
    /// - `vertex_budgets.len()` does not equal the matrix order.
    ///
    /// # Complexity
    ///
    /// O(K · (V + E)) time where K is the maximum flow value, O(V + E) space.
    #[inline]
    fn kocay(
        &self,
        vertex_budgets: &[Self::Value],
    ) -> Vec<(Self::RowIndex, Self::ColumnIndex, Self::Value)> {
        self.kocay_with_initial_flow(vertex_budgets, &[])
    }

    /// Computes a maximum balanced flow starting from a pre-initialized
    /// feasible flow.
    ///
    /// When multiple optimal solutions exist (same total flow, different edge
    /// assignments), the solver picks one arbitrarily. By accepting a feasible
    /// starting flow, the solver preserves the desired optimum when it is
    /// already maximal, only augmenting further if additional flow is
    /// possible.
    ///
    /// # Arguments
    ///
    /// * `vertex_budgets` — budget (maximum total flow) per vertex.
    /// * `initial_flow` — slice of `(row, col, flow)` triples specifying the
    ///   starting flow. Each triple must satisfy:
    ///   - `row < col`
    ///   - `flow > 0`
    ///   - The edge `(row, col)` must exist in the matrix with `flow <=
    ///     capacity`.
    ///   - No duplicate edges.
    ///   - The sum of incident flows at each vertex must not exceed its budget.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, `vertex_budgets` length is wrong, or
    /// any of the above constraints on `initial_flow` are violated.
    ///
    /// # Complexity
    ///
    /// O(K · (V + E)) time where K is the maximum flow value, O(V + E) space.
    #[inline]
    fn kocay_with_initial_flow(
        &self,
        vertex_budgets: &[Self::Value],
        initial_flow: &[(Self::RowIndex, Self::ColumnIndex, Self::Value)],
    ) -> Vec<(Self::RowIndex, Self::ColumnIndex, Self::Value)> {
        let n_rows: usize = self.number_of_rows().as_();
        let n_cols: usize = self.number_of_columns().as_();
        assert!(n_rows == n_cols, "Kocay requires a square matrix, got {n_rows} x {n_cols}");
        assert!(
            vertex_budgets.len() == n_rows,
            "vertex_budgets length {} != matrix order {n_rows}",
            vertex_budgets.len()
        );
        KocayState::new_with_initial_flow(self, vertex_budgets, initial_flow).solve()
    }
}

impl<M: SparseValuedMatrix2D> Kocay for M
where
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
}
