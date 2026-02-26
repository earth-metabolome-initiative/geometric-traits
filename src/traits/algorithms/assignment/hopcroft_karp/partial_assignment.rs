//! Partial assignment for Hopcroft-Karp algorithm.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::HopcroftKarpError;
use crate::traits::{IntoUsize, Number, SparseMatrix2D};

/// Struct representing a partial assignment.
pub struct PartialAssignment<'a, M: SparseMatrix2D + ?Sized, Distance = u16> {
    predecessors: Vec<Option<M::RowIndex>>,
    successors: Vec<Option<M::ColumnIndex>>,
    left_distances: Vec<Distance>,
    null_distance: Distance,
    matrix: &'a M,
}

impl<'a, M: SparseMatrix2D + ?Sized, Distance: Number> From<PartialAssignment<'a, M, Distance>>
    for Vec<(M::RowIndex, M::ColumnIndex)>
{
    fn from(assignment: PartialAssignment<'a, M, Distance>) -> Self {
        assignment
            .successors
            .iter()
            .copied()
            .filter_map(|right_node_id: Option<M::ColumnIndex>| {
                let right_node_id = right_node_id?;
                let row_index = assignment.predecessors[right_node_id.into_usize()]?;
                Some((row_index, right_node_id))
            })
            .collect()
    }
}

impl<'a, M: SparseMatrix2D + ?Sized, Distance: Number> From<&'a M>
    for PartialAssignment<'a, M, Distance>
{
    fn from(matrix: &'a M) -> Self {
        let predecessors = vec![None; matrix.number_of_columns().into_usize()];
        let successors = vec![None; matrix.number_of_rows().into_usize()];
        let left_distances = vec![Distance::max_value(); matrix.number_of_rows().into_usize()];
        PartialAssignment {
            predecessors,
            successors,
            left_distances,
            matrix,
            null_distance: Distance::max_value(),
        }
    }
}

impl<M: SparseMatrix2D + ?Sized, Distance: Number> PartialAssignment<'_, M, Distance> {
    /// Returns whether the provided left node id has a successor.
    pub(super) fn has_successor(&self, row_index: M::RowIndex) -> bool {
        self.successors[row_index.into_usize()].is_some()
    }

    pub(super) fn bfs(&mut self) -> Result<bool, HopcroftKarpError> {
        let mut frontier = Vec::new();
        for row_index in self.matrix.row_indices() {
            if self.has_successor(row_index) {
                self.left_distances[row_index.into_usize()] = Distance::max_value();
            } else {
                self.left_distances[row_index.into_usize()] = Distance::zero();
                frontier.push(row_index);
            }
        }

        self.null_distance = Distance::max_value();

        while !frontier.is_empty() {
            let mut tmp_frontier = Vec::new();
            for row_index in frontier {
                let left_distance = self.left_distances[row_index.into_usize()];
                if left_distance < self.null_distance {
                    if left_distance == Distance::max_value() - Distance::one() {
                        return Err(HopcroftKarpError::InsufficientDistanceType);
                    }
                    for right_node_id in self.matrix.sparse_row(row_index) {
                        let maybe_predecessor_id = self.predecessors[right_node_id.into_usize()];
                        let predecessor_distance = self.left_distance_mut(maybe_predecessor_id);
                        if *predecessor_distance == Distance::max_value() {
                            *predecessor_distance = left_distance + Distance::one();
                            tmp_frontier.extend(maybe_predecessor_id);
                        }
                    }
                }
            }
            frontier = tmp_frontier;
        }

        Ok(self.null_distance != Distance::max_value())
    }

    /// Returns the distance of the provided left node id.
    ///
    /// # Arguments
    ///
    /// * `row_index`: The identifier of the left node.
    fn left_distance(&self, row_index: Option<M::RowIndex>) -> Distance {
        let Some(row_index) = row_index else {
            return self.null_distance;
        };
        self.left_distances[row_index.into_usize()]
    }

    /// Returns a mutable reference to the distance of the provided left node
    /// id.
    ///
    /// # Arguments
    ///
    /// * `row_index`: The identifier of the left node.
    fn left_distance_mut(&mut self, row_index: Option<M::RowIndex>) -> &mut Distance {
        let Some(row_index) = row_index else {
            return &mut self.null_distance;
        };
        &mut self.left_distances[row_index.into_usize()]
    }

    /// Iterative augmenting-path search (replaces the previous recursive DFS).
    ///
    /// Finds an augmenting path starting from `initial_row` and, if one
    /// exists, commits the new assignment along that path and returns `true`.
    /// Returns `false` and marks unreachable rows with `Distance::max_value()`
    /// when no augmenting path is reachable.
    pub(super) fn dfs(&mut self, initial_row: Option<M::RowIndex>) -> bool {
        let Some(start) = initial_row else {
            return true;
        };

        // Stack: (row_index, collected column successors, next-column index)
        let mut stack: Vec<(M::RowIndex, Vec<M::ColumnIndex>, usize)> = Vec::new();
        // `chosen` records the (row, column) pairs along the current path,
        // one entry per stack frame above the bottom.
        // Invariant: len(chosen) == len(stack) - 1.
        let mut chosen: Vec<(M::RowIndex, M::ColumnIndex)> = Vec::new();

        let init_succ: Vec<M::ColumnIndex> = self.matrix.sparse_row(start).collect();
        stack.push((start, init_succ, 0));

        loop {
            if stack.is_empty() {
                return false;
            }

            let top_row = stack.last().unwrap().0;
            let top_idx = stack.last().unwrap().2;
            let top_len = stack.last().unwrap().1.len();

            if top_idx < top_len {
                let successor_id = stack.last().unwrap().1[top_idx];
                stack.last_mut().unwrap().2 += 1;

                let left_distance = self.left_distances[top_row.into_usize()];
                let maybe_pred = self.predecessors[successor_id.into_usize()];

                if self.left_distance(maybe_pred) == left_distance + Distance::one() {
                    if let Some(pred) = maybe_pred {
                        // Extend the path to pred.
                        chosen.push((top_row, successor_id));
                        let pred_succ: Vec<M::ColumnIndex> =
                            self.matrix.sparse_row(pred).collect();
                        stack.push((pred, pred_succ, 0));
                    } else {
                        // Base case: reached the unmatched end of the path.
                        // Commit all edges in `chosen` plus this final edge.
                        chosen.push((top_row, successor_id));
                        for (r, c) in chosen {
                            self.successors[r.into_usize()] = Some(c);
                            self.predecessors[c.into_usize()] = Some(r);
                        }
                        return true;
                    }
                }
            } else {
                // This row is exhausted; backtrack.
                stack.pop();
                if !chosen.is_empty() {
                    chosen.pop();
                }
                self.left_distances[top_row.into_usize()] = Distance::max_value();
            }
        }
    }
}
