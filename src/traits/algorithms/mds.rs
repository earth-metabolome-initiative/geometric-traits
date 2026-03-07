//! Classical Multidimensional Scaling (Torgerson MDS).
//!
//! Given a symmetric *n × n* dissimilarity (distance) matrix **D**, classical
//! MDS embeds the *n* objects into a *k*-dimensional Euclidean space so that
//! inter-point distances approximate the original dissimilarities.
//!
//! # Algorithm
//!
//! 1. Square element-wise: D²
//! 2. Double-center: B\[i,j\] = −0.5 × (D²\[i,j\] − row\_mean\[i\] −
//!    row\_mean\[j\] + grand\_mean)
//! 3. Eigendecompose B via the Jacobi solver → eigenvalues λ, eigenvectors V
//! 4. Coordinates: X\[i,d\] = √max(λ\[d\], 0) × V\[d\]\[i\] for top-*k*
//! 5. Kruskal stress-1: √(Σ(d\_ij − d̂\_ij)² / Σ d\_ij²)
//!
//! # Complexity
//!
//! O(n³) time (dominated by the Jacobi solver), O(n²) space.
//!
//! # Reference
//!
//! Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
//! *Psychometrika*, 17(4), 401–419.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::jacobi::{JacobiConfig, JacobiError, jacobi_decompose, sort_eigen, validate_config};
use crate::traits::{DenseValuedMatrix2D, Finite, Number};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for classical MDS.
#[derive(Debug, Clone, PartialEq)]
pub struct MdsConfig {
    /// Number of embedding dimensions (default: 2).
    pub dimensions: usize,
    /// Jacobi eigenvalue solver configuration.
    pub jacobi: JacobiConfig,
}

impl Default for MdsConfig {
    #[inline]
    fn default() -> Self {
        Self { dimensions: 2, jacobi: JacobiConfig::default() }
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of a classical MDS embedding.
///
/// Contains the embedded coordinates, the top-*k* eigenvalues used, and the
/// Kruskal stress-1 fit measure.
#[derive(Debug, Clone, PartialEq)]
pub struct MdsResult {
    /// Flat coordinate storage: point *i*, dimension *d* → `coords[i * dims +
    /// d]`.
    coordinates: Vec<f64>,
    /// Top-*k* eigenvalues used for the embedding.
    eigenvalues: Vec<f64>,
    /// Kruskal stress-1.
    stress: f64,
    /// Number of points.
    n: usize,
    /// Number of dimensions.
    dims: usize,
}

impl MdsResult {
    /// Returns the coordinates of point *i* (length =
    /// [`dimensions`](Self::dimensions)).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_points()`.
    #[must_use]
    #[inline]
    pub fn point(&self, i: usize) -> &[f64] {
        let start = i * self.dims;
        &self.coordinates[start..start + self.dims]
    }

    /// Returns the flat coordinate storage (length = `num_points() ×
    /// dimensions()`).
    #[must_use]
    #[inline]
    pub fn coordinates_flat(&self) -> &[f64] {
        &self.coordinates
    }

    /// Returns the top-*k* eigenvalues used for embedding.
    #[must_use]
    #[inline]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Returns the Kruskal stress-1 goodness-of-fit measure.
    ///
    /// A stress of 0.0 indicates a perfect embedding.
    #[must_use]
    #[inline]
    pub fn stress(&self) -> f64 {
        self.stress
    }

    /// Returns the number of embedding dimensions.
    #[must_use]
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dims
    }

    /// Returns the number of embedded points.
    #[must_use]
    #[inline]
    pub fn num_points(&self) -> usize {
        self.n
    }
}

// ============================================================================
// Error
// ============================================================================

/// Errors that can occur during classical MDS.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MdsError {
    /// The distance matrix must be square.
    #[error("The distance matrix must be square, but has {rows} rows and {columns} columns.")]
    NonSquareMatrix {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// The distance matrix is not symmetric.
    #[error(
        "The distance matrix is not symmetric: value at ({row}, {column}) differs from ({column}, {row})."
    )]
    NonSymmetricMatrix {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// A matrix entry is not finite (NaN or ±∞).
    #[error("Found a non-finite value at ({row}, {column}).")]
    NonFiniteValue {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// A distance is negative.
    #[error("Found a negative distance at ({row}, {column}).")]
    NegativeDistance {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// The distance matrix is empty (0×0).
    #[error("The distance matrix is empty.")]
    EmptyMatrix,
    /// The number of embedding dimensions must be at least 1.
    #[error("The number of embedding dimensions must be at least 1, but got {0}.")]
    InvalidDimensions(usize),
    /// The requested number of dimensions exceeds the number of points.
    #[error(
        "Requested {dimensions} embedding dimensions, but the matrix has only {num_points} points."
    )]
    DimensionsExceedPoints {
        /// Requested dimensions.
        dimensions: usize,
        /// Number of points in the matrix.
        num_points: usize,
    },
    /// The maximum distance is too large; squaring it would overflow f64.
    #[error("The maximum distance {0} is too large; squaring it would overflow f64.")]
    DistanceTooLarge(f64),
    /// A diagonal entry is not zero.
    #[error("The diagonal entry at ({index}, {index}) must be zero, but is {value}.")]
    DiagonalNotZero {
        /// Diagonal index.
        index: usize,
        /// Observed value.
        value: f64,
    },
    /// Need at least 2 points for a meaningful embedding.
    #[error("Need at least 2 points for MDS, but got {0}.")]
    TooFewPoints(usize),
    /// The Jacobi eigenvalue solver failed.
    #[error(transparent)]
    JacobiError(#[from] JacobiError),
}

// ============================================================================
// Trait
// ============================================================================

/// Trait providing classical (Torgerson) Multidimensional Scaling.
///
/// Classical MDS embeds the rows of a symmetric distance matrix into
/// *k*-dimensional Euclidean space by eigendecomposing the double-centered
/// squared-distance matrix.
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
/// // Build a 3×3 distance matrix for an equilateral triangle (all d=1).
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(6)
///         .expected_shape((3, 3))
///         .edges(
///             vec![(0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0)]
///                 .into_iter(),
///         )
///         .build()
///         .unwrap();
///
/// let padded = PaddedMatrix2D::new(
///     csr,
///     |coords: (usize, usize)| {
///         if coords.0 == coords.1 { 0.0 } else { 0.0 }
///     },
/// )
/// .unwrap();
///
/// let config = MdsConfig::default(); // 2D
/// let result = padded.classical_mds(&config).unwrap();
///
/// assert_eq!(result.num_points(), 3);
/// assert_eq!(result.dimensions(), 2);
/// assert!(result.stress() < 1e-6);
/// ```
pub trait ClassicalMds: DenseValuedMatrix2D + Sized
where
    Self::Value: Number + ToPrimitive + Finite,
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Embeds a distance matrix into Euclidean space via classical MDS.
    ///
    /// # Arguments
    ///
    /// * `config` – MDS configuration (dimensions and Jacobi sub-config).
    ///
    /// # Returns
    ///
    /// An [`MdsResult`] containing the embedding coordinates, eigenvalues,
    /// and Kruskal stress-1.
    ///
    /// # Errors
    ///
    /// Returns an [`MdsError`] if the input is invalid or the Jacobi solver
    /// fails.
    fn classical_mds(&self, config: &MdsConfig) -> Result<MdsResult, MdsError> {
        // ----- Validate config -----
        if config.dimensions == 0 {
            return Err(MdsError::InvalidDimensions(0));
        }
        validate_config(&config.jacobi)?;

        // ----- Read and validate distance matrix -----
        let num_rows: usize = self.number_of_rows().as_();
        let num_cols: usize = self.number_of_columns().as_();
        if num_rows != num_cols {
            return Err(MdsError::NonSquareMatrix { rows: num_rows, columns: num_cols });
        }
        let n = num_rows;
        if n == 0 {
            return Err(MdsError::EmptyMatrix);
        }
        if n == 1 {
            return Err(MdsError::TooFewPoints(1));
        }
        if config.dimensions > n {
            return Err(MdsError::DimensionsExceedPoints {
                dimensions: config.dimensions,
                num_points: n,
            });
        }

        let mut dist = Vec::with_capacity(n * n);
        for row_id in self.row_indices() {
            let row_idx: usize = row_id.as_();
            for (col_id, val) in self.column_indices().zip(self.row_values(row_id)) {
                let col_idx: usize = col_id.as_();
                if !val.is_finite() {
                    return Err(MdsError::NonFiniteValue { row: row_idx, column: col_idx });
                }
                let value = val
                    .to_f64()
                    .ok_or(MdsError::NonFiniteValue { row: row_idx, column: col_idx })?;
                if !value.is_finite() {
                    return Err(MdsError::NonFiniteValue { row: row_idx, column: col_idx });
                }
                if value < 0.0 {
                    return Err(MdsError::NegativeDistance { row: row_idx, column: col_idx });
                }
                dist.push(value);
            }
        }

        // Check diagonal is zero.
        for i in 0..n {
            let d = dist[i * n + i];
            if d != 0.0 {
                return Err(MdsError::DiagonalNotZero { index: i, value: d });
            }
        }

        // Check symmetry.
        for i in 0..n {
            for j in (i + 1)..n {
                let upper = dist[i * n + j];
                let lower = dist[j * n + i];
                let scale = upper.abs().max(lower.abs()).max(1.0);
                if (upper - lower).abs() > 16.0 * f64::EPSILON * scale {
                    return Err(MdsError::NonSymmetricMatrix { row: i, column: j });
                }
            }
        }

        // Force exact symmetry.
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (dist[i * n + j] + dist[j * n + i]) * 0.5;
                dist[i * n + j] = avg;
                dist[j * n + i] = avg;
            }
        }

        // Guard against overflow: d² would exceed f64::MAX for d > ~1e154.
        let max_dist = dist.iter().copied().fold(0.0_f64, f64::max);
        if max_dist > 1e154 {
            return Err(MdsError::DistanceTooLarge(max_dist));
        }

        // Keep a copy for stress computation.
        let orig_dist = dist.clone();

        // Square, double-center, eigendecompose, embed, compute stress.
        let mut b = square_and_double_center(&dist, n);

        let mut evecs =
            jacobi_decompose(&mut b, n, config.jacobi.tolerance, config.jacobi.max_sweeps)?;
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| b[i * n + i]).collect();
        sort_eigen(&mut eigenvalues, &mut evecs, n);

        let k = config.dimensions;
        let top_eigenvalues: Vec<f64> = eigenvalues[..k].to_vec();
        let mut coordinates = vec![0.0; n * k];
        for d in 0..k {
            let sqrt_lambda = eigenvalues[d].max(0.0).sqrt();
            for i in 0..n {
                coordinates[i * k + d] = sqrt_lambda * evecs[d * n + i];
            }
        }

        let stress = compute_stress(&orig_dist, &coordinates, n, k);

        Ok(MdsResult { coordinates, eigenvalues: top_eigenvalues, stress, n, dims: k })
    }
}

impl<M: DenseValuedMatrix2D> ClassicalMds for M
where
    M::Value: Number + ToPrimitive + Finite,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}

/// Square element-wise and double-center the distance matrix to produce B.
#[allow(clippy::cast_precision_loss)]
fn square_and_double_center(dist: &[f64], n: usize) -> Vec<f64> {
    let mut d_sq = vec![0.0; n * n];
    for (sq, &d) in d_sq.iter_mut().zip(dist.iter()) {
        *sq = d * d;
    }

    let mut row_means = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += d_sq[i * n + j];
        }
        row_means[i] = sum / n as f64;
    }
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d_sq[i * n + j] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    // Force exact symmetry of B.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (b[i * n + j] + b[j * n + i]) * 0.5;
            b[i * n + j] = avg;
            b[j * n + i] = avg;
        }
    }

    b
}

/// Compute Kruskal stress-1: √(Σ(d_ij − d̂_ij)² / Σ d_ij²)
fn compute_stress(orig_dist: &[f64], coordinates: &[f64], n: usize, k: usize) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let d_orig = orig_dist[i * n + j];
            // Euclidean distance in the embedding.
            let mut d_embed_sq = 0.0;
            for d in 0..k {
                let diff = coordinates[i * k + d] - coordinates[j * k + d];
                d_embed_sq += diff * diff;
            }
            let d_embed = d_embed_sq.sqrt();
            let residual = d_orig - d_embed;
            numerator += residual * residual;
            denominator += d_orig * d_orig;
        }
    }

    if denominator == 0.0 { 0.0 } else { (numerator / denominator).sqrt() }
}
