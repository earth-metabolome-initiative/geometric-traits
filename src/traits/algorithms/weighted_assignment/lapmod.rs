//! Submodule providing a native sparse implementation of LAPMOD.
//!
//! **LAPMOD** (Volgenant, *Computers & Operations Research* 23, 917–932, 1996)
//! works directly on the CSR sparse structure as the "core".  Unlike
//! [`SparseLAPJV`](super::SparseLAPJV), it never allocates or fills a dense
//! `n × n` matrix, so its memory usage is O(|E|) instead of O(n²).
//!
//! ## Rectangular / unbalanced matrices
//!
//! [`Jaqaman`] handles non-square L × R matrices by expanding them to a
//! square (L+R) × (L+R) system using the **diagonal cost extension** described
//! by Jaqaman *et al.* (*Nature Methods* 5, 695–702, 2008) and formally
//! analysed by Ramshaw & Tarjan (HP Labs HPL-2012-40, 2012).  The expansion
//! adds only 2|E| + L + R edges — preserving the sparsity that LAPMOD relies
//! on — whereas the naïve padding approach (used by [`SparseLAPJV`](super::SparseLAPJV))
//! produces a dense n × n matrix and loses the O(|E|) advantage.
use alloc::vec::Vec;

mod errors;
mod inner;

use core::fmt::Debug;

pub use errors::LAPMODError;
use inner::LapmodInner;
use num_traits::{One, Zero};

use super::LAPError;
use num_traits::AsPrimitive;

use crate::{
    impls::ValuedCSR2D,
    traits::{
        Finite, MatrixMut, Number, SparseMatrixMut, SparseValuedMatrix2D, TotalOrd,
        TryFromUsize,
    },
};

/// Trait providing the LAPMOD algorithm for solving the Weighted Assignment
/// Problem directly over a sparse valued matrix.
///
/// Unlike [`SparseLAPJV`](super::SparseLAPJV), no `padding_cost` parameter
/// is needed.  Missing entries are treated as implicit ∞ and never accessed.
pub trait LAPMOD: SparseValuedMatrix2D + Sized
where
    Self::Value: Number + Finite + TotalOrd,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the optimal weighted assignment using the LAPMOD algorithm.
    ///
    /// The matrix must be **square** and all sparse values must be
    /// **positive** and **finite**, and strictly less than `max_cost`.
    ///
    /// # Arguments
    ///
    /// * `max_cost`: An upper bound on all edge costs.  Must be positive and
    ///   finite.
    ///
    /// # Returns
    ///
    /// A vector of `(row, column)` pairs forming a perfect matching, or an
    /// error if the input is invalid or no perfect matching exists in the
    /// sparse structure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_cost` is not finite ([`LAPMODError::MaximalCostNotFinite`])
    /// - `max_cost` is not positive ([`LAPMODError::MaximalCostNotPositive`])
    /// - The matrix is not square ([`LAPMODError::NonSquareMatrix`])
    /// - Any edge cost is zero ([`LAPMODError::ZeroValues`])
    /// - Any edge cost is negative ([`LAPMODError::NegativeValues`])
    /// - Any edge cost is non-finite ([`LAPMODError::NonFiniteValues`])
    /// - Any edge cost ≥ `max_cost` ([`LAPMODError::ValueTooLarge`])
    /// - The sparse graph has no perfect matching
    ///   ([`LAPMODError::InfeasibleAssignment`])
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// let csr: ValuedCSR2D<u8, u8, u8, f64> =
    ///     ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 1.0, 6.0], [7.0, 8.0, 1.0]])
    ///         .expect("Failed to create CSR matrix");
    ///
    /// let mut assignment = csr.lapmod(1000.0).expect("LAPMOD failed");
    /// assignment.sort_unstable_by_key(|&(r, c)| (r, c));
    /// assert_eq!(assignment, vec![(0, 0), (1, 1), (2, 2)]);
    /// ```
    fn lapmod(
        &self,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPMODError>
    where
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
        <Self::RowIndex as TryFrom<usize>>::Error: Debug,
    {
        if !max_cost.is_finite() {
            return Err(LAPMODError::MaximalCostNotFinite);
        }

        if max_cost <= Self::Value::zero() {
            return Err(LAPMODError::MaximalCostNotPositive);
        }

        let n_rows = self.number_of_rows().as_();
        let n_cols = self.number_of_columns().as_();

        if n_rows != n_cols {
            return Err(LAPMODError::NonSquareMatrix);
        }

        if n_rows == 0 {
            return Ok(Vec::new());
        }

        let mut inner = LapmodInner::new(self, max_cost)?;

        inner.column_reduction_sparse()?;
        inner.reduction_transfer_sparse();

        // Two passes of augmenting row reduction (same as LAPJV).
        inner.augmenting_row_reduction_sparse();
        inner.augmenting_row_reduction_sparse();

        inner.augmentation_sparse()?;

        Ok(inner.into())
    }
}

impl<M: SparseValuedMatrix2D> LAPMOD for M
where
    M::Value: Number + Finite + TotalOrd,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}

/// Trait providing the **Jaqaman diagonal cost extension** for solving the
/// weighted assignment problem on sparse rectangular matrices.
///
/// Instead of filling missing entries with a uniform padding value (which
/// destroys sparsity), this implementation uses the **diagonal cost
/// extension** of Jaqaman *et al.* (2008) / Ramshaw & Tarjan (2012) to
/// expand an L × R matrix to a square (L+R) × (L+R) system with only
/// 2|E| + L + R edges:
///
/// ```text
///               real cols (0..R)         dummy cols (R..R+L)
///             ┌───────────────────────┬───────────────────────┐
/// real rows   │  C[i,j]               │  Diag(η/2)           │
/// (0..L)      │  (|E| entries)        │  (L entries)         │
///             ├───────────────────────┼───────────────────────┤
/// dummy rows  │  Diag(η/2)           │  ε at (L+j, R+i)     │
/// (L..L+R)    │  (R entries)          │  wherever (i,j) ∈ E  │
///             └───────────────────────┴───────────────────────┘
/// ```
///
/// Here η = `padding_cost` and ε ≈ 0 (a negligible positive value required
/// by LAPMOD's strict-positivity constraint).  An unmatched real row i pays
/// η/2 (via its dummy column R+i), and the displaced dummy row also pays
/// η/2, giving a total unmatched cost of η — matching the semantics of the
/// `padding_cost` parameter.
///
/// # Limitations
///
/// The unmatching cost η must satisfy η/2 > max(sparse values), so leaving
/// a peak unmatched always costs more than any real edge.  This forces the
/// solver to maximise the number of matches, which is not always optimal
/// (e.g. when a low-product match would lower the overall score).
///
/// # References
///
/// * Jaqaman *et al.*, "Robust single-particle tracking in live-cell
///   time-lapse sequences", *Nature Methods* 5, 695–702, 2008.
/// * Ramshaw & Tarjan, "On minimum-cost assignments in unbalanced bipartite
///   graphs", HP Labs HPL-2012-40, 2012.
pub trait Jaqaman: SparseValuedMatrix2D + Sized
where
    Self::Value: Number,
    Self::RowIndex: TryFromUsize,
    Self::ColumnIndex: TryFromUsize,
{
    #[allow(clippy::type_complexity)]
    /// Computes the weighted assignment using LAPMOD on a sparsity-preserving
    /// expansion of the matrix.
    ///
    /// # Arguments
    ///
    /// * `padding_cost`: The total cost charged for leaving a row/column
    ///   unmatched (η).  Must satisfy η/2 > max(sparse values) so that the
    ///   diagonal entries dominate all real edges.
    /// * `max_cost`: An upper bound strictly greater than `padding_cost`.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing row/column assignments in the original
    /// matrix coordinates.  Assignments routed through the dummy layer
    /// (unmatched rows/columns) are filtered out.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `padding_cost` is not a finite number
    ///   (`LAPError::PaddingValueNotFinite`)
    /// - `padding_cost` is not positive (`LAPError::PaddingValueNotPositive`)
    /// - `padding_cost` is greater than or equal to `max_cost`
    ///   (`LAPError::ValueTooLarge`)
    /// - `max_cost` is not a finite number (`LAPError::MaximalCostNotFinite`)
    /// - `max_cost` is not positive (`LAPError::MaximalCostNotPositive`)
    /// - The expanded matrix cannot be solved (`LAPError::InfeasibleAssignment`)
    /// - Matrix values violate LAPMOD input requirements
    fn jaqaman(
        &self,
        padding_cost: Self::Value,
        max_cost: Self::Value,
    ) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, LAPError>
    where
        Self::Value: Finite + TotalOrd,
        <Self::ColumnIndex as TryFrom<usize>>::Error: Debug,
        <Self::RowIndex as TryFrom<usize>>::Error: Debug,
    {
        if !padding_cost.is_finite() {
            return Err(LAPError::PaddingValueNotFinite);
        }
        if padding_cost <= Self::Value::zero() {
            return Err(LAPError::PaddingValueNotPositive);
        }
        if padding_cost >= max_cost {
            return Err(LAPError::ValueTooLarge);
        }
        if self.is_empty() {
            return Ok(vec![]);
        }

        let one = Self::Value::one();
        let two = one + one;
        let half_eta = padding_cost / two;

        // The diagonal entries (η/2) must strictly dominate all real edges
        // for the Jaqaman construction to be correct.  This can fail when
        // padding_cost is computed with an additive offset that vanishes in
        // floating point (e.g. (max + 1.0) * 2.0 where max + 1.0 == max).
        if self.max_sparse_value().unwrap() >= half_eta {
            return Err(LAPError::PaddingCostTooSmall);
        }

        let n_rows = self.number_of_rows().as_();
        let n_cols = self.number_of_columns().as_();
        let n = n_rows + n_cols;

        if n == 0 {
            return Ok(vec![]);
        }

        // Compute a very small positive value for the bottom-right block
        // entries. Ideally these would be zero (Jaqaman construction), but
        // LAPMOD requires strictly positive costs. We use half_eta / 2^40
        // ≈ half_eta × 1e-12 which is negligible for any practical matching.
        let p2 = two * two;
        let p4 = p2 * p2;
        let p8 = p4 * p4;
        let p16 = p8 * p8;
        let p32 = p16 * p16;
        let p40 = p32 * p8;
        let bottom_right_cost = half_eta / p40;

        // Collect the transpose structure: for each column j, the sorted list
        // of source rows i that have an edge (i, j) in the original matrix.
        let mut col_to_rows: Vec<Vec<usize>> = vec![Vec::new(); n_cols];
        let mut n_edges: usize = 0;
        for i in 0..n_rows {
            let row_idx = Self::RowIndex::try_from_usize(i).unwrap();
            for col in self.sparse_row(row_idx) {
                col_to_rows[col.as_()].push(i);
                n_edges += 1;
            }
        }

        let total_entries = 2 * n_edges + n_rows + n_cols;

        // Build the (L+R) × (L+R) expanded matrix using the Jaqaman /
        // Ramshaw-Tarjan diagonal cost extension. Total edges: 2|E| + L + R.
        //
        //               real cols (0..R)         dummy cols (R..R+L)
        //             ┌───────────────────────┬───────────────────────┐
        // real rows   │  C[i,j]               │  Diag(η/2)           │
        // (0..L)      │  (|E| entries)         │  (L entries)         │
        //             ├───────────────────────┼───────────────────────┤
        // dummy rows  │  Diag(η/2)            │  ε at (L+j, R+i)     │
        // (L..L+R)    │  (R entries)           │  wherever (i,j) ∈ E  │
        //             └───────────────────────┴───────────────────────┘
        let mut expanded: ValuedCSR2D<usize, usize, usize, Self::Value> =
            SparseMatrixMut::with_sparse_shaped_capacity((n, n), total_entries);

        // Real rows (0..L): original edges + diagonal entry to dummy column.
        for i in 0..n_rows {
            let row_idx = Self::RowIndex::try_from_usize(i).unwrap();
            for (col, value) in self.sparse_row(row_idx).zip(self.sparse_row_values(row_idx)) {
                expanded
                    .add((i, col.as_(), value))
                    .expect("Failed to add real edge to expanded matrix");
            }
            // Diagonal entry (i, R+i) at cost η/2.
            expanded
                .add((i, n_cols + i, half_eta))
                .expect("Failed to add top-right diagonal entry to expanded matrix");
        }

        // Dummy rows (L..L+R): for each j in 0..R:
        //   - Diagonal entry at column j with cost η/2
        //   - For each i such that (i,j) ∈ E: entry at column R+i with cost ε
        for j in 0..n_cols {
            let dummy_row = n_rows + j;
            // Bottom-left diagonal entry (L+j, j) at cost η/2.
            expanded
                .add((dummy_row, j, half_eta))
                .expect("Failed to add bottom-left diagonal entry to expanded matrix");
            // Bottom-right transpose entries.
            for &i in &col_to_rows[j] {
                expanded
                    .add((dummy_row, n_cols + i, bottom_right_cost))
                    .expect("Failed to add bottom-right entry to expanded matrix");
            }
        }

        // Solve the (L+R) × (L+R) assignment problem.
        let assignment = expanded.lapmod(max_cost).map_err(LAPError::from)?;

        // Filter: keep only assignments where row < L and col < R.
        Ok(assignment
            .into_iter()
            .filter(|&(row, col)| row < n_rows && col < n_cols)
            .map(|(row, col)| {
                (
                    Self::RowIndex::try_from_usize(row).unwrap(),
                    Self::ColumnIndex::try_from_usize(col).unwrap(),
                )
            })
            .collect())
    }
}

impl<M: SparseValuedMatrix2D> Jaqaman for M
where
    M::Value: Number,
    M::RowIndex: TryFromUsize,
    M::ColumnIndex: TryFromUsize,
{
}

impl From<LAPMODError> for LAPError {
    fn from(error: LAPMODError) -> Self {
        match error {
            LAPMODError::NonSquareMatrix => LAPError::NonSquareMatrix,
            LAPMODError::EmptyMatrix => LAPError::EmptyMatrix,
            LAPMODError::ZeroValues => LAPError::ZeroValues,
            LAPMODError::NegativeValues => LAPError::NegativeValues,
            LAPMODError::NonFiniteValues => LAPError::NonFiniteValues,
            LAPMODError::ValueTooLarge => LAPError::ValueTooLarge,
            LAPMODError::MaximalCostNotFinite => LAPError::MaximalCostNotFinite,
            LAPMODError::MaximalCostNotPositive => LAPError::MaximalCostNotPositive,
            LAPMODError::InfeasibleAssignment => LAPError::InfeasibleAssignment,
        }
    }
}
