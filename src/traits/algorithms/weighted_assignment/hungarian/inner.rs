//! Hungarian (Kuhn-Munkres) algorithm inner implementation.
//!
//! This is a pure augmentation-only solver for square dense matrices.
//! Column duals start at zero (no heuristic initialization phases),
//! and every row is augmented via Dijkstra-style shortest-path search.
//!
//! Structurally identical to `crouse/inner.rs` but operates on the generic
//! `DenseValuedMatrix2D` trait instead of a flat `&[f64]` slice.
#![cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use num_traits::{AsPrimitive, Bounded, Zero};

use crate::traits::{
    AssignmentState, DenseValuedMatrix2D, Finite, Number, TotalOrd, TryFromUsize,
    algorithms::weighted_assignment::{
        LAPError,
        lap_error::validate_lap_value_against_max,
        lapjv::common::{assignments_from_assigned_rows, augmentation_backtrack, dense_find_path},
    },
};

/// Support struct for computing the weighted assignment using the Hungarian
/// algorithm.
pub(super) struct HungarianInner<'matrix, M: DenseValuedMatrix2D + ?Sized> {
    /// The matrix to compute the assignment on.
    matrix: &'matrix M,
    /// Column dual variables, initialized to zero.
    column_costs: Vec<M::Value>,
    /// The maximum cost of the matrix.
    max_cost: M::Value,
    /// Column → row assignment.
    assigned_rows: Vec<AssignmentState<M::RowIndex>>,
    /// Row → column assignment.
    assigned_columns: Vec<AssignmentState<M::ColumnIndex>>,
}

impl<M: DenseValuedMatrix2D + ?Sized> HungarianInner<'_, M>
where
    M::Value: Number,
    M::ColumnIndex: TryFromUsize,
    <M::ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    #[inline]
    pub(super) fn into_assignments(self) -> Vec<(M::RowIndex, M::ColumnIndex)> {
        assignments_from_assigned_rows(self.assigned_rows, self.matrix.number_of_rows().as_())
    }
}

impl<'matrix, M: DenseValuedMatrix2D + ?Sized> HungarianInner<'matrix, M>
where
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: Bounded,
    M::ColumnIndex: Bounded,
{
    pub(super) fn new(matrix: &'matrix M, max_cost: M::Value) -> Result<Self, LAPError> {
        if matrix.number_of_rows().as_() != matrix.number_of_columns().as_() {
            return Err(LAPError::NonSquareMatrix);
        }

        // Validate all entries.
        for row_index in matrix.row_indices() {
            for value in matrix.row_values(row_index) {
                validate_lap_value_against_max(value, max_cost)?;
            }
        }

        // Column duals start at zero — the key difference from LAPJV.
        let column_costs = vec![M::Value::zero(); matrix.number_of_columns().as_()];

        Ok(HungarianInner {
            matrix,
            column_costs,
            max_cost,
            assigned_rows: vec![AssignmentState::Unassigned; matrix.number_of_columns().as_()],
            assigned_columns: vec![AssignmentState::Unassigned; matrix.number_of_rows().as_()],
        })
    }
}

impl<M: DenseValuedMatrix2D + ?Sized> HungarianInner<'_, M>
where
    M::Value: Number + Finite + TotalOrd,
{
    #[inline]
    pub(super) fn augmentation(&mut self) {
        let n = self.matrix.number_of_columns().as_();
        if n == 0 {
            return;
        }

        let mut to_scan = vec![M::ColumnIndex::max_value(); n];
        let mut predecessors = vec![M::RowIndex::max_value(); n];
        let mut distances = vec![self.max_cost; n];

        for start_row_index in self.matrix.row_indices() {
            let sink_col = dense_find_path::<M>(
                start_row_index,
                &mut to_scan,
                &mut predecessors,
                &mut distances,
                &self.assigned_rows,
                &mut self.column_costs,
                self.matrix,
            );

            augmentation_backtrack(
                sink_col,
                &predecessors,
                &mut self.assigned_rows,
                &mut self.assigned_columns,
                start_row_index,
            );
        }
    }
}
