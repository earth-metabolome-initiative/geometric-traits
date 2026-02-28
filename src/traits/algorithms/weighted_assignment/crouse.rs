//! Crouse rectangular LAPJV solver for sparse matrices.
//!
//! This module provides [`Crouse`], a trait that solves the weighted
//! assignment problem on sparse rectangular matrices by:
//! 1. **Compactifying** the sparse matrix to eliminate empty rows/columns.
//! 2. **Building** a dense rectangular cost matrix (non-edge cost for missing
//!    entries).
//! 3. **Solving** with the Crouse 2016 rectangular LAPJV algorithm
//!    (augmentation phase only, handles nr ≤ nc natively).
//! 4. **Mapping** compact indices back to the original matrix coordinates.
//! 5. **Filtering** assignments that correspond to non-edge entries.
//!
//! Unlike [`Jaqaman`](super::Jaqaman), which forces all peaks to match by
//! charging η > 2×max_edge_cost per unmatched peak, this approach charges
//! only `non_edge_cost` for unmatched entries, correctly allowing the solver
//! to leave peaks unmatched when that yields a better assignment.
//!
//! # Reference
//!
//! Crouse, D. F. (2016). "On implementing 2D rectangular assignment
//! algorithms." *IEEE Transactions on Aerospace and Electronic Systems*,
//! 52(4), 1679–1696.
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

pub mod errors;
mod inner;

pub use errors::CrouseError;

use crate::{
    impls::compactify,
    traits::{Number, SparseMatrix2D, SparseValuedMatrix2D, TryFromUsize},
};

/// Trait providing the Crouse rectangular LAPJV solver for sparse valued
/// matrices.
///
/// The solver compactifies the sparse matrix, builds a dense rectangular
/// matrix with `non_edge_cost` for missing entries, and runs the Crouse 2016
/// augmentation-only LAPJV.  The result is filtered to exclude assignments
/// at `non_edge_cost`.
pub trait Crouse: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + Into<f64>,
    Self::RowIndex: TryFromUsize + Ord,
    Self::ColumnIndex: TryFromUsize + Ord,
{
    #[allow(clippy::type_complexity)]
    /// Solve the sparse rectangular assignment problem.
    ///
    /// # Arguments
    ///
    /// * `non_edge_cost` - Cost assigned to entries not present in the sparse
    ///   matrix. Assignments at this cost are filtered from the result.
    /// * `max_cost` - Upper bound on all costs; must be > `non_edge_cost`.
    ///
    /// # Returns
    ///
    /// A vector of `(row, column)` pairs in the original matrix coordinates,
    /// containing only assignments that correspond to actual sparse edges.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_cost` is not finite (`CrouseError::MaximalCostNotFinite`)
    /// - `max_cost` is not positive (`CrouseError::MaximalCostNotPositive`)
    /// - Any matrix value is non-finite (`CrouseError::NonFiniteValues`)
    /// - Any matrix value is zero (`CrouseError::ZeroValues`)
    /// - Any matrix value is negative (`CrouseError::NegativeValues`)
    /// - Any matrix value is greater than or equal to `max_cost`
    ///   (`CrouseError::ValueTooLarge`)
    fn crouse(
        &self,
        non_edge_cost: f64,
        max_cost: f64,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, CrouseError>
    where
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
        <Self::RowIndex as TryFrom<usize>>::Error: Debug,
    {
        if !max_cost.is_finite() {
            return Err(CrouseError::MaximalCostNotFinite);
        }
        if max_cost <= 0.0 {
            return Err(CrouseError::MaximalCostNotPositive);
        }
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Compactify — remap to 0..n_unique_rows × 0..n_unique_cols.
        let compact = compactify(self);
        let nr = compact.row_map.len();
        let nc = compact.col_map.len();

        if nr == 0 || nc == 0 {
            return Ok(Vec::new());
        }

        // Step 2: Build dense rectangular matrix.
        // Ensure nr ≤ nc: if there are more unique rows than columns, we
        // transpose and solve with the transposed matrix.
        let transposed = nr > nc;
        let (eff_nr, eff_nc) = if transposed { (nc, nr) } else { (nr, nc) };

        let mut dense = vec![non_edge_cost; eff_nr * eff_nc];

        if transposed {
            // Fill with transposed sparse entries.
            for compact_row in 0..nr {
                for (compact_col, value) in compact
                    .matrix
                    .sparse_row(compact_row)
                    .zip(compact.matrix.sparse_row_values(compact_row))
                {
                    // Transposed: row→col, col→row.
                    dense[compact_col * eff_nc + compact_row] = value;
                }
            }
        } else {
            // Fill with sparse entries directly.
            for compact_row in 0..nr {
                for (compact_col, value) in compact
                    .matrix
                    .sparse_row(compact_row)
                    .zip(compact.matrix.sparse_row_values(compact_row))
                {
                    dense[compact_row * eff_nc + compact_col] = value;
                }
            }
        }

        // Step 3: Solve with Crouse rectangular LAPJV.
        let assignments = inner::crouse_inner(&dense, eff_nr, eff_nc, max_cost)?;

        // Step 4: Map back to original indices and filter non-edges.
        let mut result = Vec::with_capacity(assignments.len());
        for (eff_row, eff_col) in assignments {
            // Un-transpose if needed.
            let (compact_row, compact_col) =
                if transposed { (eff_col, eff_row) } else { (eff_row, eff_col) };

            // Check if this is a real edge (not a non_edge_cost entry).
            let has_edge = compact.matrix.sparse_row(compact_row).any(|c| c == compact_col);

            if has_edge {
                let orig_row = compact.row_map[compact_row];
                let orig_col = compact.col_map[compact_col];
                result.push((orig_row, orig_col));
            }
        }

        Ok(result)
    }
}

impl<M: SparseValuedMatrix2D> Crouse for M
where
    M::Value: Number + Into<f64>,
    M::RowIndex: TryFromUsize + Ord,
    M::ColumnIndex: TryFromUsize + Ord,
{
}
