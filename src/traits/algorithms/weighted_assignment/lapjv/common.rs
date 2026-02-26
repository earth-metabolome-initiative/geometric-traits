//! Shared helper functions used by both the LAPJV and LAPMOD algorithms.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::traits::{AssignmentState, IntoUsize, Number, TotalOrd};

/// Finds the minimum distance among the columns in `to_scan[lower_bound..]`,
/// rearranges them so that all minimum-distance columns come immediately after
/// `to_scan[lower_bound]`, and returns the new upper-bound index.
///
/// # Arguments
///
/// * `lower_bound`: The index from which to start scanning.
/// * `distances`: The distances vector indexed by column usize.
/// * `to_scan`: The slice of columns to scan (modified in place).
///
/// # Returns
///
/// The index one past the last column at the minimum distance.
pub(crate) fn find_minimum_distance<C, V>(
    lower_bound: usize,
    distances: &[V],
    to_scan: &mut [C],
) -> usize
where
    C: IntoUsize + Copy,
    V: PartialOrd + Copy,
{
    debug_assert!(
        lower_bound < to_scan.len(),
        "We expected the lower bound to be less than the length of the to scan vector"
    );
    let mut upper_bound = lower_bound + 1;
    let column_index = to_scan[lower_bound];
    let mut minimum_distance = distances[column_index.into_usize()];

    for k in lower_bound + 1..to_scan.len() {
        let column_index = to_scan[k];
        let distance = distances[column_index.into_usize()];
        if distance <= minimum_distance {
            if distance < minimum_distance {
                upper_bound = lower_bound;
                minimum_distance = distance;
            }
            to_scan[k] = to_scan[upper_bound];
            to_scan[upper_bound] = column_index;
            upper_bound += 1;
        }
    }

    upper_bound
}

/// Backtracks along the predecessor chain to update the assignment after an
/// augmenting path has been found.
///
/// # Arguments
///
/// * `column_index`: The sink column found by the path search.
/// * `predecessors`: The predecessor map indexed by column usize.
/// * `assigned_rows`: Column → row assignment (modified in place).
/// * `assigned_columns`: Row → column assignment (modified in place).
/// * `start_row`: The unassigned row that started the augmenting path.
pub(crate) fn augmentation_backtrack<R, C>(
    mut column_index: C,
    predecessors: &[R],
    assigned_rows: &mut [AssignmentState<R>],
    assigned_columns: &mut [AssignmentState<C>],
    start_row: R,
) where
    R: Copy + Eq + IntoUsize,
    C: Copy + IntoUsize,
{
    let mut number_of_steps = 0usize;
    let max_steps = assigned_rows.len();

    loop {
        number_of_steps += 1;
        assert!(
            number_of_steps <= max_steps,
            "augmentation_backtrack detected a predecessor cycle"
        );

        let row_index = predecessors[column_index.into_usize()];

        assigned_rows[column_index.into_usize()] = AssignmentState::Assigned(row_index);

        // Root of the augmenting path: this row was intentionally unassigned
        // when the search started, so it has no previous column to follow.
        if row_index == start_row {
            assigned_columns[row_index.into_usize()] = AssignmentState::Assigned(column_index);
            break;
        }

        let AssignmentState::Assigned(old_column_index) = assigned_columns[row_index.into_usize()]
        else {
            unreachable!("We expected the assigned column to be in the assigned state");
        };

        assigned_columns[row_index.into_usize()] = AssignmentState::Assigned(column_index);
        column_index = old_column_index;
    }
}

/// Core loop of the augmenting row reduction phase, parameterised by the
/// function that computes the first and second minimum reduced costs for a row.
///
/// The `first_and_second_min` closure receives the row index and a shared slice
/// of the current column costs, so it can be called even while `column_costs`
/// is owned mutably by this function.
///
/// # Arguments
///
/// * `unassigned_rows`: The current list of unassigned rows (modified).
/// * `assigned_rows`: Column → row assignment (indexed by column, modified).
/// * `assigned_columns`: Row → column assignment (indexed by row, modified).
/// * `column_costs`: Current column dual variables (modified).
/// * `number_of_rows`: Total number of rows (used for iteration-count guard).
/// * `first_and_second_min`: Closure `(row, &col_costs) -> ((best_col,
///   best_val), (Option<second_col>, second_val))`.
pub(crate) fn augmenting_row_reduction_impl<R, C, V>(
    unassigned_rows: &mut Vec<R>,
    assigned_rows: &mut [AssignmentState<R>],
    assigned_columns: &mut [AssignmentState<C>],
    column_costs: &mut Vec<V>,
    number_of_rows: usize,
    mut first_and_second_min: impl FnMut(R, &[V]) -> ((C, V), (Option<C>, V)),
) where
    R: Copy + IntoUsize,
    C: Copy + IntoUsize,
    V: Number + TotalOrd,
{
    if unassigned_rows.is_empty() {
        return;
    }

    let mut current_unassigned_row_index = 0;
    let mut updated_number_of_unassigned_rows = 0;
    let mut number_of_iterations = 0;
    let original_number_of_unassigned_rows = unassigned_rows.len();

    while current_unassigned_row_index < original_number_of_unassigned_rows {
        let unassigned_row_index = unassigned_rows[current_unassigned_row_index];
        current_unassigned_row_index += 1;
        number_of_iterations += 1;

        // We determine the first and second minimum reduced costs of the row.
        // The immutable borrow of column_costs ends when this call returns.
        let (
            (mut first_minimum_column_index, first_minimum_value),
            (second_minimum_column_index, second_minimum_value),
        ) = first_and_second_min(unassigned_row_index, column_costs.as_slice());

        let mut row_index = assigned_rows[first_minimum_column_index.into_usize()];

        if number_of_iterations < current_unassigned_row_index * number_of_rows {
            if first_minimum_value < second_minimum_value {
                column_costs[first_minimum_column_index.into_usize()] -=
                    second_minimum_value - first_minimum_value;
            } else if let (AssignmentState::Assigned(_), Some(second_minimum_column_index)) =
                (row_index, second_minimum_column_index)
            {
                first_minimum_column_index = second_minimum_column_index;
                row_index = assigned_rows[first_minimum_column_index.into_usize()];
            }
            if let AssignmentState::Assigned(assigned_row) = row_index {
                if first_minimum_value < second_minimum_value {
                    current_unassigned_row_index -= 1;
                    unassigned_rows[current_unassigned_row_index] = assigned_row;
                } else {
                    unassigned_rows[updated_number_of_unassigned_rows] = assigned_row;
                    updated_number_of_unassigned_rows += 1;
                }
            }
        } else if let AssignmentState::Assigned(assigned_row) = row_index {
            unassigned_rows[updated_number_of_unassigned_rows] = assigned_row;
            updated_number_of_unassigned_rows += 1;
        }

        // We update the assigned row with the new column index.
        assigned_rows[first_minimum_column_index.into_usize()] =
            AssignmentState::Assigned(unassigned_row_index);
        // We update the assigned column with the new row index.
        assigned_columns[unassigned_row_index.into_usize()] =
            AssignmentState::Assigned(first_minimum_column_index);
    }

    unassigned_rows.truncate(updated_number_of_unassigned_rows);
}
