//! Submodule providing the concrete implementation of the LAPJV algorithm.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt::Debug;

use num_traits::{AsPrimitive, Bounded};

use super::{
    LAPError,
    common::{
        assignments_from_assigned_rows, augmentation_backtrack, augmenting_row_reduction_impl,
        dense_find_path,
    },
};
use crate::traits::{
    AssignmentState, DenseValuedMatrix2D, Finite, Number, TotalOrd, TryFromUsize,
    algorithms::weighted_assignment::lap_error::validate_lap_value_against_max,
};

/// Support struct for computing the weighted assignment using the LAPJV
/// algorithm.
pub(super) struct Inner<'matrix, M: DenseValuedMatrix2D + ?Sized> {
    /// The matrix to compute the assignment on.
    matrix: &'matrix M,
    /// The column costs of the matrix.
    column_costs: Vec<M::Value>,
    /// Vector of unassigned rows.
    unassigned_rows: Vec<M::RowIndex>,
    /// The maximum cost of the matrix.
    max_cost: M::Value,
    /// Vector of assigned rows.
    assigned_rows: Vec<AssignmentState<M::RowIndex>>,
    /// Vector of assigned columns.
    assigned_columns: Vec<AssignmentState<M::ColumnIndex>>,
}

impl<M: DenseValuedMatrix2D + ?Sized> Inner<'_, M>
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

impl<'matrix, M: DenseValuedMatrix2D + ?Sized> Inner<'matrix, M>
where
    M::Value: Number + TotalOrd,
    M::RowIndex: Bounded,
    M::ColumnIndex: Bounded,
{
    pub(super) fn new(matrix: &'matrix M, max_cost: M::Value) -> Result<Self, LAPError> {
        // Check if the matrix is square
        if matrix.number_of_rows().as_() != matrix.number_of_columns().as_() {
            return Err(LAPError::NonSquareMatrix);
        }

        let column_costs = vec![max_cost; matrix.number_of_columns().as_()];

        Ok(Inner {
            matrix,
            column_costs,
            unassigned_rows: Vec::new(),
            max_cost,
            assigned_rows: vec![AssignmentState::Unassigned; matrix.number_of_rows().as_()],
            assigned_columns: vec![AssignmentState::Unassigned; matrix.number_of_columns().as_()],
        })
    }
}

impl<M: DenseValuedMatrix2D + ?Sized> Inner<'_, M>
where
    M::Value: Number + Finite + TotalOrd,
{
    #[inline]
    pub(super) fn column_reduction(&mut self) -> Result<(), LAPError> {
        debug_assert!(
            self.column_costs.iter().all(|&cost| cost == self.max_cost),
            "We expected the column costs to be initialized to the maximum cost",
        );
        debug_assert!(
            self.assigned_rows.iter().all(AssignmentState::is_unassigned),
            "We expected all rows to be unassigned",
        );
        debug_assert!(
            self.assigned_columns.iter().all(AssignmentState::is_unassigned),
            "We expected all columns to be unassigned",
        );

        for row_index in self.matrix.row_indices() {
            // We retrieve the minimum value and its column on the row.
            for (column_index, value) in
                self.matrix.column_indices().zip(self.matrix.row_values(row_index))
            {
                validate_lap_value_against_max(value, self.max_cost)?;

                if value < self.column_costs[column_index.as_()] {
                    self.assigned_rows[column_index.as_()] = AssignmentState::Assigned(row_index);
                    self.column_costs[column_index.as_()] = value;
                }
            }
        }

        debug_assert!(
            self.assigned_rows.iter().all(AssignmentState::is_assigned),
            "We expected all rows to be assigned",
        );

        for column_index in self.matrix.column_indices().rev() {
            let AssignmentState::Assigned(assigned_row) = self.assigned_rows[column_index.as_()]
            else {
                unreachable!("We expected the assigned row to be in the assigned state");
            };
            match self.assigned_columns.get(assigned_row.as_()) {
                Some(AssignmentState::Unassigned) => {
                    self.assigned_columns[assigned_row.as_()] =
                        AssignmentState::Assigned(column_index);
                }
                Some(
                    AssignmentState::Assigned(assigned_column_index)
                    | AssignmentState::Conflict(assigned_column_index),
                ) => {
                    self.assigned_columns[assigned_row.as_()] =
                        AssignmentState::Conflict(*assigned_column_index);
                    self.assigned_rows[column_index.as_()] = AssignmentState::Unassigned;
                }
                None => {
                    unreachable!("We expected the assigned column to be in the assigned state");
                }
            }
        }

        Ok(())
    }

    #[inline]
    pub(super) fn reduction_transfer(&mut self) {
        debug_assert!(
            self.unassigned_rows.is_empty(),
            "We expected the unassigned rows to be empty",
        );

        for row_index in self.matrix.row_indices() {
            match self.assigned_columns.get(row_index.as_()) {
                Some(AssignmentState::Unassigned) => {
                    self.unassigned_rows.push(row_index);
                }
                Some(AssignmentState::Conflict(conflicted_column)) => {
                    self.assigned_columns[row_index.as_()] =
                        AssignmentState::Assigned(*conflicted_column);
                }
                Some(AssignmentState::Assigned(assigned_column)) => {
                    let minimum_reduced_cost = self
                        .matrix
                        .column_indices()
                        .zip(self.matrix.row_values(row_index))
                        // We remove the columns that match the assigned column
                        .filter_map(|(column_index, value)| {
                            if column_index == *assigned_column {
                                None
                            } else {
                                Some(value - self.column_costs[column_index.as_()])
                            }
                        })
                        .min_by(|&a, &b| a.total_cmp(&b))
                        .unwrap_or(self.max_cost);
                    self.column_costs[assigned_column.as_()] -= minimum_reduced_cost;
                }
                None => {
                    unreachable!("We expected the assigned column to be in the assigned state");
                }
            }
        }
    }

    #[inline]
    pub(super) fn augmenting_row_reduction(&mut self) {
        let matrix = self.matrix;
        let max_cost = self.max_cost;
        let number_of_rows = matrix.number_of_rows().as_();
        augmenting_row_reduction_impl(
            &mut self.unassigned_rows,
            &mut self.assigned_rows,
            &mut self.assigned_columns,
            &mut self.column_costs,
            number_of_rows,
            |row, col_costs| {
                let mut iterator = matrix.column_indices().zip(matrix.row_values(row)).map(
                    |(column_index, cost)| (column_index, cost - col_costs[column_index.as_()]),
                );

                let (mut first_minimum_index, mut first_minimum_reduced_cost) =
                    iterator.next().expect("We expected the iterator to have at least one element");

                let mut second_minimum_column_index: Option<M::ColumnIndex> = None;
                let mut second_minimum_reduced_cost = max_cost;
                for (column_index, reduced_cost) in iterator {
                    if reduced_cost < second_minimum_reduced_cost {
                        if reduced_cost >= first_minimum_reduced_cost {
                            second_minimum_column_index = Some(column_index);
                            second_minimum_reduced_cost = reduced_cost;
                        } else {
                            second_minimum_column_index = Some(first_minimum_index);
                            second_minimum_reduced_cost = first_minimum_reduced_cost;
                            first_minimum_index = column_index;
                            first_minimum_reduced_cost = reduced_cost;
                        }
                    }
                }

                (
                    (first_minimum_index, first_minimum_reduced_cost),
                    (second_minimum_column_index, second_minimum_reduced_cost),
                )
            },
        );
    }

    #[inline]
    pub(super) fn augmentation(&mut self) {
        if self.unassigned_rows.is_empty() {
            return;
        }

        let n = self.matrix.number_of_columns().as_();
        let mut to_scan = vec![M::ColumnIndex::max_value(); n];
        let mut predecessors = vec![M::RowIndex::max_value(); n];
        let mut distances = vec![self.max_cost; n];

        while let Some(unassigned_row_index) = self.unassigned_rows.pop() {
            let column_index = dense_find_path::<M>(
                unassigned_row_index,
                &mut to_scan,
                &mut predecessors,
                &mut distances,
                &self.assigned_rows,
                &mut self.column_costs,
                self.matrix,
            );

            augmentation_backtrack(
                column_index,
                &predecessors,
                &mut self.assigned_rows,
                &mut self.assigned_columns,
                unassigned_row_index,
            );
        }
    }
}
