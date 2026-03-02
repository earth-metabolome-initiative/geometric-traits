//! Rectangular LAPJV solver: augmentation-only algorithm for nr ≤ nc dense
//! matrices, based on Crouse 2016.
//!
//! This is the LAPJV augmentation phase adapted for rectangular (nr × nc with
//! nr ≤ nc) cost matrices.  No initialization phases are needed — all column
//! costs start at zero, all rows are unassigned.
//!
//! # Floating-point tolerance
//!
//! The Dijkstra-based shortest-augmenting-path search compares reduced costs
//! for exact equality when deciding whether a column joins the minimum-distance
//! frontier.  In the rectangular augmentation-only variant, dual variables
//! (column costs) are updated once per augmenting path, accumulating
//! floating-point rounding errors proportional to the number of augmentation
//! steps (i.e. the number of rows, nr).
//!
//! Bijsterbosch & Volgenant (2010), "Solving the Rectangular assignment problem
//! and applications", *Annals of Operations Research*, 181(1), 443–462, showed
//! that the LAPJV initialisation phases (column reduction and augmenting row
//! reduction) violate the dual feasibility constraint v ≤ 0 when applied to
//! rectangular problems, which is why this solver correctly omits them.
//! However, the absence of init phases means dual variables start at zero and
//! are shaped entirely by augmentation-step updates, making them more
//! susceptible to accumulated rounding error.
//!
//! For near-degenerate cost matrices — where many entries are within a few ULPs
//! of each other — the accumulated error causes the exact-equality check to
//! miss columns that should tie at the minimum distance.  This manifests as
//! suboptimal augmenting paths and, ultimately, suboptimal total cost.  In
//! spectral cosine similarity with `mz_power=0` and wide m/z tolerance, the
//! cost matrix is nearly uniform (all entries ≈ 1 + ε), triggering this issue
//! for spectra with ≥100 peaks.
//!
//! The fix follows the approach used by SciPy (`scipy.optimize.linear_sum_
//! assignment`), the Princeton Java LAPJV reference, and the `lap-jv` crate:
//! replace exact equality with an epsilon-ball comparison, where epsilon scales
//! with problem size.  Specifically, `epsilon = nr × f64::EPSILON`.  This
//! bounds the total cost error to nr × epsilon ≈ nr² × f64::EPSILON ≈ 1e-11
//! for typical problem sizes, which is negligible compared to real cost
//! differences.
//!
//! ## References
//!
//! - Crouse, D.F. (2016). "On implementing 2D rectangular assignment
//!   algorithms." *IEEE Transactions on Aerospace and Electronic Systems*,
//!   52(4), 1679–1696.
//! - Bijsterbosch, J. & Volgenant, A. (2010). "Solving the Rectangular
//!   assignment problem and applications." *Annals of Operations Research*,
//!   181(1), 443–462.
//! - Jonker, R. & Volgenant, A. (1987). "A shortest augmenting path algorithm
//!   for dense and sparse linear assignment problems." *Computing*, 38(4),
//!   325–340.
#![cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use super::errors::CrouseError;
use crate::traits::{
    AssignmentState,
    algorithms::weighted_assignment::lapjv::common::augmentation_backtrack,
};

/// Finds the minimum distance among columns in `to_scan[lower_bound..]`,
/// grouping all columns within `epsilon` of the minimum into the frontier.
///
/// This is an f64-specific variant of the generic [`find_minimum_distance`]
/// from `lapjv/common.rs`.  The generic version uses exact `PartialOrd`
/// comparisons, which works for the square LAPJV where init phases establish
/// accurate dual variables.  In the rectangular augmentation-only Crouse
/// variant, dual variables accumulate rounding errors over nr augmentation
/// steps, so exact comparisons miss true ties in near-degenerate matrices.
///
/// The two changes from the generic version:
/// - `distance <= minimum_distance` becomes `distance <= minimum_distance + epsilon`
///   (include columns within epsilon of the current minimum)
/// - `distance < minimum_distance` becomes `distance < minimum_distance - epsilon`
///   (only reset the frontier when strictly better by more than epsilon)
///
/// This matches the epsilon-clamping approach used in:
/// - SciPy's `linear_sum_assignment` (C implementation)
/// - Princeton's Java LAPJV reference implementation
/// - The `lap-jv` Rust crate
///
/// See module-level documentation for full references.
fn find_minimum_distance_tolerant(
    lower_bound: usize,
    distances: &[f64],
    to_scan: &mut [usize],
    epsilon: f64,
) -> usize {
    debug_assert!(
        lower_bound < to_scan.len(),
        "We expected the lower bound to be less than the length of the to scan vector"
    );
    let mut upper_bound = lower_bound + 1;
    let column_index = to_scan[lower_bound];
    let mut minimum_distance = distances[column_index];

    for k in lower_bound + 1..to_scan.len() {
        let column_index = to_scan[k];
        let distance = distances[column_index];
        if distance <= minimum_distance + epsilon {
            if distance < minimum_distance - epsilon {
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

    // Epsilon tolerance for floating-point degeneracy in the Dijkstra scan.
    //
    // In the augmentation-only rectangular LAPJV (Crouse 2016), there are no
    // initialisation phases — dual variables start at zero and are updated
    // once per augmenting path.  Each update subtracts `distances[col] -
    // min_dist`, introducing ~1 ULP of rounding error per step.  After nr
    // augmentation steps, the accumulated error is ~nr × machine-epsilon.
    //
    // Bijsterbosch & Volgenant (2010) showed that LAPJV init phases violate
    // dual feasibility (v ≤ 0) for rectangular problems, confirming that
    // augmentation-only is the correct approach — but it makes us more
    // reliant on this epsilon to handle the resulting rounding accumulation.
    //
    // The total assignment cost error from this tolerance is bounded by
    // nr × epsilon = nr² × f64::EPSILON ≈ 1e-11 for nr ≤ 512, which is
    // negligible for any practical cost function.
    let epsilon = nr as f64 * f64::EPSILON;

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
                upper_bound =
                    find_minimum_distance_tolerant(lower_bound, &distances, &mut to_scan, epsilon);

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
                epsilon,
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
///
/// This is the inner loop of the Dijkstra-based shortest-augmenting-path
/// search from Jonker & Volgenant (1987), §3.  For each frontier column,
/// we relax edges to unsettled columns and check whether the new reduced
/// cost places them on the current minimum-distance frontier.
///
/// The comparison `reduced <= min_dist + epsilon` (rather than exact
/// equality) accounts for accumulated floating-point rounding in the dual
/// variables.  See module-level documentation and
/// [`find_minimum_distance_tolerant`] for details.
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
    epsilon: f64,
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

                // Epsilon-tolerant frontier test.  The original Jonker &
                // Volgenant (1987) algorithm uses exact equality here
                // (`reduced == min_dist`), which is correct in exact
                // arithmetic.  In floating-point, accumulated dual-variable
                // rounding errors can make `reduced` differ from `min_dist`
                // by a few ULPs even when they represent the same
                // shortest-path distance.  The epsilon ball ensures these
                // near-tied columns still join the frontier, preserving
                // optimality of the augmenting path.
                if reduced <= min_dist + epsilon {
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
