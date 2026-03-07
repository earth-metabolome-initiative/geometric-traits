//! Cyclic Jacobi eigenvalue decomposition for symmetric real matrices.
//!
//! Given a symmetric real matrix **A** of order *n*, the cyclic Jacobi method
//! iteratively applies Givens (plane) rotations to annihilate off-diagonal
//! elements until the matrix is (approximately) diagonal. The diagonal entries
//! then approximate the eigenvalues, and the accumulated rotation matrix
//! approximates the eigenvectors.
//!
//! # Complexity
//!
//! Each rotation costs O(n) and there are O(n²) pairs per sweep, with
//! typically 5–10 sweeps for convergence, giving **O(n³)** total time.
//! Memory usage is **O(n²)** for the working copy of the matrix plus the
//! eigenvector accumulator.
//!
//! # Reference
//!
//! Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.),
//! §8.5. Johns Hopkins University Press.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use crate::traits::{DenseValuedMatrix2D, Finite, Number};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Jacobi eigenvalue solver.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiConfig {
    /// Convergence tolerance for the squared off-diagonal Frobenius norm.
    ///
    /// The solver declares convergence when the sum of squares of all
    /// off-diagonal elements drops below this value. Default: `1e-12`.
    pub tolerance: f64,
    /// Maximum number of full sweeps over all (p, q) pairs before the solver
    /// gives up. Default: `1000`.
    pub max_sweeps: usize,
}

impl Default for JacobiConfig {
    #[inline]
    fn default() -> Self {
        Self { tolerance: 1e-12, max_sweeps: 1000 }
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of a Jacobi eigenvalue decomposition.
///
/// Eigenvalues are sorted in descending order; eigenvectors are stored so
/// that `eigenvector(k)` returns a contiguous slice for the *k*-th
/// eigenvector.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiResult {
    /// Eigenvalues sorted descending.
    eigenvalues: Vec<f64>,
    /// Flat eigenvector matrix (n × n).
    /// Layout: `eigenvectors[k * n + i]` = component *i* of eigenvector *k*.
    eigenvectors: Vec<f64>,
    /// Matrix order.
    order: usize,
}

impl JacobiResult {
    /// Returns the eigenvalues, sorted in descending order.
    #[must_use]
    #[inline]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Returns the *k*-th eigenvector (length *n*).
    ///
    /// # Panics
    ///
    /// Panics if `k >= self.order()`.
    #[must_use]
    #[inline]
    pub fn eigenvector(&self, k: usize) -> &[f64] {
        let start = k * self.order;
        &self.eigenvectors[start..start + self.order]
    }

    /// Returns the raw flat eigenvector storage (length n²).
    #[must_use]
    #[inline]
    pub fn eigenvectors_flat(&self) -> &[f64] {
        &self.eigenvectors
    }

    /// Returns the matrix order *n*.
    #[must_use]
    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }
}

// ============================================================================
// Error
// ============================================================================

/// Errors that can occur during Jacobi eigenvalue decomposition.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JacobiError {
    /// The matrix must be square.
    #[error("The matrix must be square, but has {rows} rows and {columns} columns.")]
    NonSquareMatrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// A matrix entry is not finite (NaN or ±∞).
    #[error("Found a non-finite value at ({row}, {column}).")]
    NonFiniteValue {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// The matrix is not symmetric.
    #[error(
        "The matrix is not symmetric: value at ({row}, {column}) differs from ({column}, {row})."
    )]
    NonSymmetricMatrix {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// The tolerance must be finite and strictly positive.
    #[error("The tolerance must be finite and strictly positive.")]
    InvalidTolerance,
    /// The maximum number of sweeps must be strictly positive.
    #[error("The maximum number of sweeps must be strictly positive.")]
    InvalidMaxSweeps,
    /// The solver did not converge within the allotted sweeps.
    #[error("The Jacobi solver did not converge within {max_sweeps} sweeps.")]
    DidNotConverge {
        /// Number of sweeps that were attempted.
        max_sweeps: usize,
    },
    /// The matrix is empty (0×0).
    #[error("The matrix is empty.")]
    EmptyMatrix,
}

// ============================================================================
// Private helpers
// ============================================================================

/// Validate the user-supplied configuration.
fn validate_config(config: &JacobiConfig) -> Result<(), JacobiError> {
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(JacobiError::InvalidTolerance);
    }
    if config.max_sweeps == 0 {
        return Err(JacobiError::InvalidMaxSweeps);
    }
    Ok(())
}

/// Read the matrix into a flat row-major `Vec<f64>`, validating squareness,
/// finiteness, and symmetry along the way.
///
/// Returns `(flat_matrix, order)` where `flat_matrix` has length order².
#[allow(clippy::many_single_char_names)]
fn read_symmetric_matrix<M>(matrix: &M) -> Result<(Vec<f64>, usize), JacobiError>
where
    M: DenseValuedMatrix2D,
    M::Value: Number + ToPrimitive + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
    let num_rows: usize = matrix.number_of_rows().as_();
    let num_cols: usize = matrix.number_of_columns().as_();
    if num_rows != num_cols {
        return Err(JacobiError::NonSquareMatrix { rows: num_rows, columns: num_cols });
    }
    let order = num_rows;
    if order == 0 {
        return Err(JacobiError::EmptyMatrix);
    }

    let mut flat = Vec::with_capacity(order * order);
    for row_id in matrix.row_indices() {
        let row_idx: usize = row_id.as_();
        for (col_id, val) in matrix.column_indices().zip(matrix.row_values(row_id)) {
            let col_idx: usize = col_id.as_();
            if !val.is_finite() {
                return Err(JacobiError::NonFiniteValue { row: row_idx, column: col_idx });
            }
            let value = val
                .to_f64()
                .ok_or(JacobiError::NonFiniteValue { row: row_idx, column: col_idx })?;
            if !value.is_finite() {
                return Err(JacobiError::NonFiniteValue { row: row_idx, column: col_idx });
            }
            flat.push(value);
        }
    }

    // Check symmetry with relative tolerance.
    for row in 0..order {
        for col in (row + 1)..order {
            let upper = flat[row * order + col];
            let lower = flat[col * order + row];
            let scale = upper.abs().max(lower.abs()).max(1.0);
            if (upper - lower).abs() > 16.0 * f64::EPSILON * scale {
                return Err(JacobiError::NonSymmetricMatrix { row, column: col });
            }
        }
    }

    // Force exact symmetry.
    for row in 0..order {
        for col in (row + 1)..order {
            let avg = (flat[row * order + col] + flat[col * order + row]) * 0.5;
            flat[row * order + col] = avg;
            flat[col * order + row] = avg;
        }
    }

    Ok((flat, order))
}

/// Compute the squared Frobenius norm of the off-diagonal elements.
fn off_diag_norm_sq(matrix: &[f64], order: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..order {
        for col in (row + 1)..order {
            let val = matrix[row * order + col];
            sum += 2.0 * val * val;
        }
    }
    sum
}

/// Core cyclic Jacobi sweep loop.
///
/// Operates on `matrix` in-place (row-major, length order²), driving it
/// towards diagonal form. Returns the eigenvector storage where
/// `evecs[k * order + i]` is component *i* of eigenvector *k*.
#[allow(clippy::many_single_char_names)]
fn jacobi_decompose(
    matrix: &mut [f64],
    order: usize,
    tol: f64,
    max_sweeps: usize,
) -> Result<Vec<f64>, JacobiError> {
    let mut evecs = vec![0.0; order * order];
    for idx in 0..order {
        evecs[idx * order + idx] = 1.0;
    }

    for _ in 0..max_sweeps {
        if off_diag_norm_sq(matrix, order) < tol {
            return Ok(evecs);
        }

        for pivot_row in 0..order {
            for pivot_col in (pivot_row + 1)..order {
                let off_diag = matrix[pivot_row * order + pivot_col];
                if off_diag.abs() < 1e-15 {
                    continue;
                }

                // Compute Jacobi rotation parameters.
                let tau = (matrix[pivot_col * order + pivot_col]
                    - matrix[pivot_row * order + pivot_row])
                    / (2.0 * off_diag);
                let tan_theta = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cos_theta = 1.0 / (1.0 + tan_theta * tan_theta).sqrt();
                let sin_theta = tan_theta * cos_theta;

                // Update matrix: rotate rows/columns pivot_row and pivot_col.
                let diag_row = matrix[pivot_row * order + pivot_row];
                let diag_col = matrix[pivot_col * order + pivot_col];
                matrix[pivot_row * order + pivot_row] = diag_row - tan_theta * off_diag;
                matrix[pivot_col * order + pivot_col] = diag_col + tan_theta * off_diag;
                matrix[pivot_row * order + pivot_col] = 0.0;
                matrix[pivot_col * order + pivot_row] = 0.0;

                for other in 0..order {
                    if other == pivot_row || other == pivot_col {
                        continue;
                    }
                    let elem_row = matrix[other * order + pivot_row];
                    let elem_col = matrix[other * order + pivot_col];
                    matrix[other * order + pivot_row] = cos_theta * elem_row - sin_theta * elem_col;
                    matrix[other * order + pivot_col] = sin_theta * elem_row + cos_theta * elem_col;
                    matrix[pivot_row * order + other] = matrix[other * order + pivot_row];
                    matrix[pivot_col * order + other] = matrix[other * order + pivot_col];
                }

                // Update eigenvector storage.
                // evecs[k * order + i] = component i of eigenvector k.
                for idx in 0..order {
                    let vec_row = evecs[pivot_row * order + idx];
                    let vec_col = evecs[pivot_col * order + idx];
                    evecs[pivot_row * order + idx] = cos_theta * vec_row - sin_theta * vec_col;
                    evecs[pivot_col * order + idx] = sin_theta * vec_row + cos_theta * vec_col;
                }
            }
        }
    }

    // Final convergence check after all sweeps.
    if off_diag_norm_sq(matrix, order) < tol {
        Ok(evecs)
    } else {
        Err(JacobiError::DidNotConverge { max_sweeps })
    }
}

/// Sort eigenvalues in descending order and reorder the eigenvector columns
/// to match.
fn sort_eigen(eigenvalues: &mut [f64], eigenvectors: &mut [f64], order: usize) {
    // Build index permutation sorted by descending eigenvalue.
    let mut indices: Vec<usize> = (0..order).collect();
    indices.sort_by(|&idx_a, &idx_b| {
        eigenvalues[idx_b].partial_cmp(&eigenvalues[idx_a]).unwrap_or(core::cmp::Ordering::Equal)
    });

    // Apply permutation to eigenvalues.
    let old_eigenvalues: Vec<f64> = eigenvalues.to_vec();
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        eigenvalues[new_pos] = old_eigenvalues[old_pos];
    }

    // Apply permutation to eigenvector columns.
    // Layout: eigenvectors[k * order + i] = component i of eigenvector k.
    let old_eigenvectors: Vec<f64> = eigenvectors.to_vec();
    for (new_col, &old_col) in indices.iter().enumerate() {
        for idx in 0..order {
            eigenvectors[new_col * order + idx] = old_eigenvectors[old_col * order + idx];
        }
    }
}

// ============================================================================
// Public trait
// ============================================================================

/// Trait providing Jacobi eigenvalue decomposition for dense symmetric real
/// matrices.
///
/// The cyclic Jacobi method iteratively applies Givens rotations to drive the
/// matrix towards diagonal form. On convergence the diagonal contains the
/// eigenvalues and the accumulated rotations form the eigenvector matrix.
///
/// # Complexity
///
/// O(n³) time, O(n²) space.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::{PaddedMatrix2D, ValuedCSR2D},
///     prelude::*,
///     traits::{DenseValuedMatrix2D, EdgesBuilder},
/// };
///
/// // Build the 2×2 symmetric matrix [[2, 1], [1, 2]].
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(2)
///         .expected_shape((2, 2))
///         .edges(vec![(0, 1, 1.0), (1, 0, 1.0)].into_iter())
///         .build()
///         .unwrap();
///
/// let padded = PaddedMatrix2D::new(
///     csr,
///     |coords: (usize, usize)| {
///         if coords.0 == coords.1 { 2.0 } else { 0.0 }
///     },
/// )
/// .unwrap();
///
/// let config = JacobiConfig::default();
/// let result = padded.jacobi(&config).unwrap();
///
/// // Eigenvalues of [[2,1],[1,2]] are 3.0 and 1.0.
/// assert!((result.eigenvalues()[0] - 3.0).abs() < 1e-10);
/// assert!((result.eigenvalues()[1] - 1.0).abs() < 1e-10);
/// ```
pub trait Jacobi: DenseValuedMatrix2D + Sized
where
    Self::Value: Number + ToPrimitive + Finite,
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Computes the eigenvalue decomposition of a symmetric real matrix.
    ///
    /// # Arguments
    ///
    /// * `config` – solver configuration (tolerance and sweep limit).
    ///
    /// # Returns
    ///
    /// A [`JacobiResult`] containing eigenvalues (descending) and
    /// eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns a [`JacobiError`] if the matrix is non-square, non-symmetric,
    /// empty, contains non-finite values, or if the solver does not converge.
    #[inline]
    fn jacobi(&self, config: &JacobiConfig) -> Result<JacobiResult, JacobiError> {
        validate_config(config)?;
        let (mut flat, order) = read_symmetric_matrix(self)?;
        let mut evecs = jacobi_decompose(&mut flat, order, config.tolerance, config.max_sweeps)?;

        let mut eigenvalues: Vec<f64> = (0..order).map(|i| flat[i * order + i]).collect();
        sort_eigen(&mut eigenvalues, &mut evecs, order);

        Ok(JacobiResult { eigenvalues, eigenvectors: evecs, order })
    }
}

impl<M: DenseValuedMatrix2D> Jacobi for M
where
    M::Value: Number + ToPrimitive + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}
