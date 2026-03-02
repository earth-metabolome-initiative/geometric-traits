//! Rectangular LAPJV solver: augmentation-only algorithm for nr ≤ nc dense
//! matrices, based on Crouse 2016.
//!
//! This is the LAPJV augmentation phase adapted for rectangular (nr × nc with
//! nr ≤ nc) cost matrices.  No initialization phases are needed — all column
//! costs start at zero, all rows are unassigned.
#![cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use super::errors::CrouseError;
use crate::traits::{
    AssignmentState,
    algorithms::weighted_assignment::lapjv::common::{
        augmentation_backtrack, find_minimum_distance,
    },
};

/// Solve a rectangular linear assignment problem on a dense nr × nc cost
/// matrix stored in row-major order, where nr ≤ nc.
///
/// All costs must be positive, finite, and strictly less than `max_cost`.
///
/// Returns a vector of (row, column) assignments.  Every row gets exactly
/// one column; some columns may be unassigned.
pub(crate) fn crouse_inner(
    data: &[f64],
    nr: usize,
    nc: usize,
    max_cost: f64,
) -> Result<Vec<(usize, usize)>, CrouseError> {
    debug_assert!(nr <= nc, "crouse_inner requires nr <= nc");
    debug_assert_eq!(data.len(), nr * nc);

    if nr == 0 {
        return Ok(Vec::new());
    }

    // Validate costs.
    for &cost in data {
        if !cost.is_finite() {
            return Err(CrouseError::NonFiniteValues);
        }
        if cost <= 0.0 {
            if cost == 0.0 {
                return Err(CrouseError::ZeroValues);
            }
            return Err(CrouseError::NegativeValues);
        }
        if cost >= max_cost {
            return Err(CrouseError::ValueTooLarge);
        }
    }

    // Column dual variables (length nc), initialized to zero.
    let mut column_costs = vec![0.0f64; nc];

    // Assignment state: assigned_rows[col] = which row is assigned to col.
    let mut assigned_rows: Vec<AssignmentState<usize>> = vec![AssignmentState::Unassigned; nc];
    // assigned_columns[row] = which col is assigned to row.
    let mut assigned_columns: Vec<AssignmentState<usize>> = vec![AssignmentState::Unassigned; nr];

    // Augmentation buffers.
    let mut to_scan: Vec<usize> = vec![0; nc];
    let mut predecessors: Vec<usize> = vec![0; nc];
    let mut distances: Vec<f64> = vec![max_cost; nc];

    // Augment for each row.
    for row in 0..nr {
        // Initialize distances from this row.
        for col in 0..nc {
            to_scan[col] = col;
            predecessors[col] = row;
            distances[col] = data[row * nc + col] - column_costs[col];
        }

        let mut lower_bound: usize = 0;
        let mut upper_bound: usize = 0;
        let mut n_ready: usize = 0;

        // Find augmenting path.
        let sink_col = 'outer: loop {
            if lower_bound == upper_bound {
                n_ready = lower_bound;
                upper_bound = find_minimum_distance(lower_bound, &distances, &mut to_scan);

                for &col in &to_scan[lower_bound..upper_bound] {
                    if assigned_rows[col].is_unassigned() {
                        break 'outer col;
                    }
                }
            }

            // Scan: expand frontier.
            if let Some(col) = scan(
                &mut lower_bound,
                &mut upper_bound,
                &mut to_scan,
                &mut distances,
                &mut predecessors,
                &assigned_rows,
                &column_costs,
                data,
                nc,
            ) {
                break 'outer col;
            }
        };

        // Update dual variables for settled columns.
        let min_dist = distances[to_scan[lower_bound]];
        for &col in &to_scan[0..n_ready] {
            column_costs[col] += distances[col] - min_dist;
        }

        // Backtrack along predecessor chain.
        augmentation_backtrack(
            sink_col,
            &predecessors,
            &mut assigned_rows,
            &mut assigned_columns,
            row,
        );
    }

    // Collect assignments.
    let mut result = Vec::with_capacity(nr);
    for (row, state) in assigned_columns.into_iter().enumerate() {
        let AssignmentState::Assigned(col) = state else {
            unreachable!("Every row should be assigned after augmentation");
        };
        result.push((row, col));
    }

    Ok(result)
}

/// Scan phase: expand the minimum-distance frontier over neighbours of
/// assigned rows.
#[allow(clippy::too_many_arguments)]
fn scan(
    lower_bound_ref: &mut usize,
    upper_bound_ref: &mut usize,
    to_scan: &mut [usize],
    distances: &mut [f64],
    predecessors: &mut [usize],
    assigned_rows: &[AssignmentState<usize>],
    column_costs: &[f64],
    data: &[f64],
    nc: usize,
) -> Option<usize> {
    let mut lower_bound = *lower_bound_ref;
    let mut upper_bound = *upper_bound_ref;

    while lower_bound != upper_bound {
        let col = to_scan[lower_bound];
        lower_bound += 1;

        let AssignmentState::Assigned(row) = assigned_rows[col] else {
            unreachable!("Frontier column must be assigned during scan");
        };

        let min_dist = distances[col];
        let initial_reduced = data[row * nc + col] - column_costs[col] - min_dist;

        let current_upper = upper_bound;
        for k in current_upper..to_scan.len() {
            let neighbour_col = to_scan[k];
            let reduced =
                data[row * nc + neighbour_col] - column_costs[neighbour_col] - initial_reduced;

            if reduced < distances[neighbour_col] {
                distances[neighbour_col] = reduced;
                predecessors[neighbour_col] = row;

                if reduced.total_cmp(&min_dist).is_eq() {
                    if assigned_rows[neighbour_col].is_unassigned() {
                        *lower_bound_ref = lower_bound;
                        *upper_bound_ref = upper_bound;
                        return Some(neighbour_col);
                    }
                    to_scan[k] = to_scan[upper_bound];
                    to_scan[upper_bound] = neighbour_col;
                    upper_bound += 1;
                }
            }
        }
    }

    *lower_bound_ref = lower_bound;
    *upper_bound_ref = upper_bound;
    None
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn test_square_identity() {
        // 3×3 identity-like costs: diagonal = 1, off-diagonal = 10.
        let data = vec![
            1.0, 10.0, 10.0, //
            10.0, 1.0, 10.0, //
            10.0, 10.0, 1.0, //
        ];
        let mut result = crouse_inner(&data, 3, 3, 100.0).unwrap();
        result.sort_unstable();
        assert_eq!(result, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_rectangular_2x3() {
        // 2×3: rows should pick best columns.
        let data = vec![
            10.0, 1.0, 5.0, //
            1.0, 10.0, 5.0, //
        ];
        let mut result = crouse_inner(&data, 2, 3, 100.0).unwrap();
        result.sort_unstable();
        assert_eq!(result, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn test_empty() {
        let result = crouse_inner(&[], 0, 0, 100.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_row() {
        let data = vec![5.0, 1.0, 3.0];
        let result = crouse_inner(&data, 1, 3, 100.0).unwrap();
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_validation_errors() {
        assert!(matches!(
            crouse_inner(&[f64::NAN], 1, 1, 100.0),
            Err(CrouseError::NonFiniteValues)
        ));
        assert!(matches!(crouse_inner(&[0.0], 1, 1, 100.0), Err(CrouseError::ZeroValues)));
        assert!(matches!(crouse_inner(&[-1.0], 1, 1, 100.0), Err(CrouseError::NegativeValues)));
        assert!(matches!(crouse_inner(&[100.0], 1, 1, 100.0), Err(CrouseError::ValueTooLarge)));
    }
}
