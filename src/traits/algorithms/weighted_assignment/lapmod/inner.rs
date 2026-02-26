//! Concrete implementation of the LAPMOD algorithm over a sparse valued matrix.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt::Debug;

use num_traits::{Bounded, Zero};

use super::LAPMODError;
use crate::traits::{
    AssignmentState, Finite, IntoUsize, Number, SparseValuedMatrix2D, TotalOrd, TryFromUsize,
    algorithms::weighted_assignment::lapjv::common::{
        augmentation_backtrack, augmenting_row_reduction_impl,
    },
};

/// Support struct for computing the weighted assignment using the LAPMOD
/// algorithm operating directly on a sparse valued matrix.
pub(super) struct LapmodInner<'matrix, M: SparseValuedMatrix2D + ?Sized> {
    /// The sparse matrix to compute the assignment on.
    matrix: &'matrix M,
    /// Column dual variables (indexed by column usize).
    column_costs: Vec<M::Value>,
    /// Rows not yet augmented.
    unassigned_rows: Vec<M::RowIndex>,
    /// Sentinel: `distances[j] == max_cost` means column `j` not yet reached.
    max_cost: M::Value,
    /// For each column `j`, which row is currently assigned to it.
    assigned_rows: Vec<AssignmentState<M::RowIndex>>,
    /// For each row `i`, which column it is currently assigned to.
    assigned_columns: Vec<AssignmentState<M::ColumnIndex>>,
}

// ---------------------------------------------------------------------------
// From<LapmodInner> → Vec of assignments
// ---------------------------------------------------------------------------

impl<'matrix, M: SparseValuedMatrix2D + ?Sized> From<LapmodInner<'matrix, M>>
    for Vec<(M::RowIndex, M::ColumnIndex)>
where
    M::Value: Number,
    M::ColumnIndex: TryFromUsize,
    <M::ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    fn from(inner: LapmodInner<'matrix, M>) -> Self {
        let mut assignments: Vec<(M::RowIndex, M::ColumnIndex)> =
            Vec::with_capacity(inner.matrix.number_of_rows().into_usize());
        for (col_usize, state) in inner.assigned_rows.into_iter().enumerate() {
            let AssignmentState::Assigned(row) = state else {
                // Column was not assigned — only happens if we returned an
                // error, so this branch is unreachable on the Ok path.
                unreachable!("We expected every column to be assigned in a perfect matching");
            };
            assignments.push((row, M::ColumnIndex::try_from_usize(col_usize).unwrap()));
        }
        assignments
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'matrix, M: SparseValuedMatrix2D + ?Sized> LapmodInner<'matrix, M>
where
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: Bounded,
    M::ColumnIndex: Bounded,
{
    pub(super) fn new(matrix: &'matrix M, max_cost: M::Value) -> Result<Self, LAPMODError> {
        if matrix.number_of_rows().into_usize() != matrix.number_of_columns().into_usize() {
            return Err(LAPMODError::NonSquareMatrix);
        }
        let n = matrix.number_of_columns().into_usize();
        Ok(LapmodInner {
            matrix,
            column_costs: vec![max_cost; n],
            unassigned_rows: Vec::new(),
            max_cost,
            assigned_rows: vec![AssignmentState::Unassigned; n],
            assigned_columns: vec![AssignmentState::Unassigned; n],
        })
    }
}

// ---------------------------------------------------------------------------
// Algorithm phases
// ---------------------------------------------------------------------------

impl<M: SparseValuedMatrix2D + ?Sized> LapmodInner<'_, M>
where
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: Bounded,
    M::ColumnIndex: Bounded,
{
    /// Phase 1: sparse column reduction.
    ///
    /// For each sparse entry `(row, col, cost)` in row order, assign `col` to
    /// `row` if `cost < column_costs[col]`.  After all entries are visited, a
    /// reverse column scan resolves conflicts (a column is "won" by the last
    /// row that achieved its minimum).
    ///
    /// Returns `Err(InfeasibleAssignment)` as soon as a row with no entries is
    /// found.
    #[inline]
    pub(super) fn column_reduction_sparse(&mut self) -> Result<(), LAPMODError> {
        // Check that every row has at least one sparse entry.
        for row in self.matrix.row_indices() {
            if self.matrix.sparse_row(row).next().is_none() {
                return Err(LAPMODError::InfeasibleAssignment);
            }
        }

        // Update column_costs and assigned_rows from sparse entries.
        for row in self.matrix.row_indices() {
            for (col, cost) in self.matrix.sparse_row(row).zip(self.matrix.sparse_row_values(row)) {
                // Validate the entry.
                if !cost.is_finite() {
                    return Err(LAPMODError::NonFiniteValues);
                }
                if cost == M::Value::zero() {
                    return Err(LAPMODError::ZeroValues);
                }
                if cost < M::Value::zero() {
                    return Err(LAPMODError::NegativeValues);
                }
                if cost >= self.max_cost {
                    return Err(LAPMODError::ValueTooLarge);
                }

                if cost < self.column_costs[col.into_usize()] {
                    self.assigned_rows[col.into_usize()] = AssignmentState::Assigned(row);
                    self.column_costs[col.into_usize()] = cost;
                }
            }
        }

        // Reverse-scan columns to resolve conflicts.
        for col in self.matrix.column_indices().rev() {
            let AssignmentState::Assigned(row) = self.assigned_rows[col.into_usize()] else {
                // Column has no sparse entries or was displaced — skip.
                continue;
            };
            match self.assigned_columns.get(row.into_usize()) {
                Some(AssignmentState::Unassigned) => {
                    self.assigned_columns[row.into_usize()] = AssignmentState::Assigned(col);
                }
                Some(
                    AssignmentState::Assigned(other_col) | AssignmentState::Conflict(other_col),
                ) => {
                    self.assigned_columns[row.into_usize()] = AssignmentState::Conflict(*other_col);
                    self.assigned_rows[col.into_usize()] = AssignmentState::Unassigned;
                }
                None => {
                    unreachable!("Row index out of bounds during column reduction reverse scan");
                }
            }
        }

        Ok(())
    }

    /// Phase 2: sparse reduction transfer.
    ///
    /// - Unassigned rows are pushed onto `unassigned_rows`.
    /// - Conflict rows are resolved (the last-won column is kept).
    /// - Assigned rows contribute the minimum reduced cost of their other
    ///   sparse neighbours to lower the dual variable of the assigned column.
    #[inline]
    pub(super) fn reduction_transfer_sparse(&mut self) {
        for row in self.matrix.row_indices() {
            match self.assigned_columns[row.into_usize()] {
                AssignmentState::Unassigned => {
                    self.unassigned_rows.push(row);
                }
                AssignmentState::Conflict(col) => {
                    self.assigned_columns[row.into_usize()] = AssignmentState::Assigned(col);
                }
                AssignmentState::Assigned(col) => {
                    let min_reduced = self
                        .matrix
                        .sparse_row(row)
                        .zip(self.matrix.sparse_row_values(row))
                        .filter_map(|(c, cost)| {
                            if c.into_usize() == col.into_usize() {
                                None
                            } else {
                                Some(cost - self.column_costs[c.into_usize()])
                            }
                        })
                        .min_by(TotalOrd::total_cmp)
                        .unwrap_or(self.max_cost);
                    self.column_costs[col.into_usize()] -= min_reduced;
                }
            }
        }
    }

    /// Phase 3: sparse augmenting row reduction (run twice).
    #[inline]
    pub(super) fn augmenting_row_reduction_sparse(&mut self) {
        let matrix = self.matrix;
        let max_cost = self.max_cost;
        let number_of_rows = matrix.number_of_rows().into_usize();
        augmenting_row_reduction_impl(
            &mut self.unassigned_rows,
            &mut self.assigned_rows,
            &mut self.assigned_columns,
            &mut self.column_costs,
            number_of_rows,
            |row, col_costs| {
                let mut iter = matrix
                    .sparse_row(row)
                    .zip(matrix.sparse_row_values(row))
                    .map(|(col, cost)| (col, cost - col_costs[col.into_usize()]));

                let (mut first_col, mut first_val) = iter
                    .next()
                    .expect("Every row must have at least one sparse entry after column_reduction");

                let mut second_col: Option<M::ColumnIndex> = None;
                let mut second_val = max_cost;

                for (col, val) in iter {
                    if val < second_val {
                        if val >= first_val {
                            second_col = Some(col);
                            second_val = val;
                        } else {
                            second_col = Some(first_col);
                            second_val = first_val;
                            first_col = col;
                            first_val = val;
                        }
                    }
                }

                ((first_col, first_val), (second_col, second_val))
            },
        );
    }

    // -----------------------------------------------------------------------
    // Augmentation (Dijkstra-style path search over sparse structure)
    // -----------------------------------------------------------------------

    /// Collects the not-yet-done columns at minimum distance among `todo`.
    ///
    /// Returns the number of columns written into `scan`.
    fn find_minimum_distance_sparse(
        &self,
        distances: &[M::Value],
        scan: &mut [M::ColumnIndex],
        n_todo: usize,
        todo: &[M::ColumnIndex],
        done: &[bool],
    ) -> usize {
        let mut hi = 0usize;
        let mut minimum_distance = self.max_cost;
        let mut has_minimum = false;

        for &col in &todo[0..n_todo] {
            let col_usize = col.into_usize();
            if done[col_usize] {
                continue;
            }

            let distance = distances[col_usize];
            if !has_minimum || distance <= minimum_distance {
                if !has_minimum || distance < minimum_distance {
                    hi = 0;
                    minimum_distance = distance;
                    has_minimum = true;
                }
                scan[hi] = col;
                hi += 1;
            }
        }

        hi
    }

    /// Expands the current minimum-distance frontier (`scan[lo..hi]`) over the
    /// sparse neighbourhoods and updates predecessor/distance structures.
    ///
    /// Returns `Some(sink_col)` if a free column is reached at current minimum
    /// distance, otherwise `None`.
    ///
    /// This mirrors the reference LAPMOD `_scan_sparse_2` semantics:
    /// - `done`: columns that are already in the current/previous scan sets.
    /// - `todo`: discovered columns outside the current minimum frontier.
    /// - `ready`: scanned columns used by the dual update.
    #[allow(clippy::too_many_arguments)]
    fn scan_sparse(
        &mut self,
        lower_bound_ref: &mut usize,
        upper_bound_ref: &mut usize,
        n_todo_ref: &mut usize,
        n_ready_ref: &mut usize,
        scan: &mut [M::ColumnIndex],
        todo: &mut [M::ColumnIndex],
        done: &mut [bool],
        added: &mut [bool],
        ready: &mut [M::ColumnIndex],
        distances: &mut [M::Value],
        predecessors: &mut [M::RowIndex],
    ) -> Option<M::ColumnIndex> {
        let mut lower_bound = *lower_bound_ref;
        let mut upper_bound = *upper_bound_ref;
        let mut n_todo = *n_todo_ref;
        let mut n_ready = *n_ready_ref;

        while lower_bound != upper_bound {
            let col = scan[lower_bound];
            lower_bound += 1;

            let AssignmentState::Assigned(row) = self.assigned_rows[col.into_usize()] else {
                unreachable!("Frontier column must be assigned to a row during augmentation scan");
            };

            ready[n_ready] = col;
            n_ready += 1;

            let minimum_distance = distances[col.into_usize()];

            // Compute h = cost(row,col) - column_cost[col] - d[col].
            let initial_reduced = self
                .matrix
                .sparse_row(row)
                .zip(self.matrix.sparse_row_values(row))
                .find(|&(c, _)| c.into_usize() == col.into_usize())
                .map(|(_, cost)| cost - self.column_costs[col.into_usize()] - minimum_distance)
                .expect("Row assigned to column must have an entry for that column in sparse_row");

            for (neighbour_col, neighbour_cost) in
                self.matrix.sparse_row(row).zip(self.matrix.sparse_row_values(row))
            {
                let nc_usize = neighbour_col.into_usize();
                if done[nc_usize] {
                    continue;
                }

                let new_dist = neighbour_cost - self.column_costs[nc_usize] - initial_reduced;
                if new_dist < distances[nc_usize] {
                    distances[nc_usize] = new_dist;
                    predecessors[nc_usize] = row;

                    if new_dist <= minimum_distance {
                        if self.assigned_rows[nc_usize].is_unassigned() {
                            // Keep caller's bounds/counters untouched to match
                            // the reference LAPMOD early-return behavior.
                            return Some(neighbour_col);
                        }

                        scan[upper_bound] = neighbour_col;
                        upper_bound += 1;
                        done[nc_usize] = true;
                    } else if !added[nc_usize] {
                        todo[n_todo] = neighbour_col;
                        n_todo += 1;
                        added[nc_usize] = true;
                    }
                }
            }
        }

        *lower_bound_ref = lower_bound;
        *upper_bound_ref = upper_bound;
        *n_todo_ref = n_todo;
        *n_ready_ref = n_ready;
        None
    }

    /// Returns the free sink column reached by the sparse shortest augmenting
    /// path search from `start_row`, updating dual variables for columns that
    /// became "ready" (settled) before the sink level.
    #[allow(clippy::too_many_arguments)]
    fn find_path_sparse(
        &mut self,
        start_row: M::RowIndex,
        scan: &mut [M::ColumnIndex],
        todo: &mut [M::ColumnIndex],
        ready: &mut [M::ColumnIndex],
        done: &mut [bool],
        added: &mut [bool],
        predecessors: &mut [M::RowIndex],
        distances: &mut [M::Value],
    ) -> Result<M::ColumnIndex, LAPMODError> {
        let mut lower_bound = 0usize;
        let mut upper_bound = 0usize;
        let mut n_ready = 0usize;
        let mut n_todo = 0usize;

        done.fill(false);
        added.fill(false);
        predecessors.fill(start_row);
        distances.fill(self.max_cost);

        // Seed TODO with sparse neighbours of start_row.
        for (col, cost) in
            self.matrix.sparse_row(start_row).zip(self.matrix.sparse_row_values(start_row))
        {
            let col_usize = col.into_usize();
            let dist = cost - self.column_costs[col_usize];

            if dist < distances[col_usize] {
                distances[col_usize] = dist;
                predecessors[col_usize] = start_row;
            }
            if !added[col_usize] {
                todo[n_todo] = col;
                n_todo += 1;
                added[col_usize] = true;
            }
        }

        let sink_col = 'outer: loop {
            if lower_bound == upper_bound {
                lower_bound = 0;
                upper_bound =
                    self.find_minimum_distance_sparse(distances, scan, n_todo, todo, done);

                if upper_bound == 0 {
                    return Err(LAPMODError::InfeasibleAssignment);
                }

                for &col in &scan[lower_bound..upper_bound] {
                    if self.assigned_rows[col.into_usize()].is_unassigned() {
                        break 'outer col;
                    }
                    done[col.into_usize()] = true;
                }
            }

            if let Some(col) = self.scan_sparse(
                &mut lower_bound,
                &mut upper_bound,
                &mut n_todo,
                &mut n_ready,
                scan,
                todo,
                done,
                added,
                ready,
                distances,
                predecessors,
            ) {
                break 'outer col;
            }
        };

        let minimum_distance = distances[scan[lower_bound].into_usize()];
        for &col in &ready[0..n_ready] {
            self.column_costs[col.into_usize()] += distances[col.into_usize()] - minimum_distance;
        }

        Ok(sink_col)
    }

    /// Phase 4: sparse augmentation loop.
    ///
    /// Calls `find_path_sparse` for each unassigned row and backtracks along
    /// the augmenting path to update the assignment.
    /// Distances are reset to `max_cost` before each path search.
    #[inline]
    pub(super) fn augmentation_sparse(&mut self) -> Result<(), LAPMODError> {
        if self.unassigned_rows.is_empty() {
            return Ok(());
        }

        let n = self.matrix.number_of_columns().into_usize();
        let mut scan: Vec<M::ColumnIndex> = vec![M::ColumnIndex::max_value(); n];
        let mut todo: Vec<M::ColumnIndex> = vec![M::ColumnIndex::max_value(); n];
        let mut ready: Vec<M::ColumnIndex> = vec![M::ColumnIndex::max_value(); n];
        let mut predecessors: Vec<M::RowIndex> = vec![M::RowIndex::max_value(); n];
        let mut distances: Vec<M::Value> = vec![self.max_cost; n];
        let mut done = vec![false; n];
        let mut added = vec![false; n];

        while let Some(unassigned_row) = self.unassigned_rows.pop() {
            let sink_col = self.find_path_sparse(
                unassigned_row,
                &mut scan,
                &mut todo,
                &mut ready,
                &mut done,
                &mut added,
                &mut predecessors,
                &mut distances,
            )?;

            augmentation_backtrack(
                sink_col,
                &predecessors,
                &mut self.assigned_rows,
                &mut self.assigned_columns,
                unassigned_row,
            );
        }

        Ok(())
    }
}
