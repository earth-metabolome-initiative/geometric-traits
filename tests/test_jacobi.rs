//! Tests for the Jacobi eigenvalue decomposition trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::EdgesBuilder,
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;
type TestPadded = PaddedMatrix2D<TestValCSR, Box<dyn Fn((usize, usize)) -> f64>>;

/// Helper: build a dense n×n matrix from a flat row-major slice.
fn matrix_from_flat(values: &[f64], n: usize) -> TestPadded {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let v = values[i * n + j];
            if v != 0.0 {
                edges.push((i, j, v));
            }
        }
    }
    let inner = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .unwrap();
    PaddedMatrix2D::new(
        inner,
        Box::new(move |_: (usize, usize)| 0.0) as Box<dyn Fn((usize, usize)) -> f64>,
    )
    .unwrap()
}

/// Helper: build a dense n×n matrix from a flat row-major slice using a custom
/// padding function that returns the actual desired values.
fn matrix_from_dense(values: &[f64], n: usize) -> TestPadded {
    let owned: Vec<f64> = values.to_vec();
    let inner = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(0)
        .expected_shape((n, n))
        .edges(core::iter::empty())
        .build()
        .unwrap();
    PaddedMatrix2D::new(
        inner,
        Box::new(move |coords: (usize, usize)| owned[coords.0 * n + coords.1])
            as Box<dyn Fn((usize, usize)) -> f64>,
    )
    .unwrap()
}

// ============================================================================
// Positive tests
// ============================================================================

#[test]
fn test_1x1_matrix() {
    let m = matrix_from_dense(&[42.0], 1);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert_eq!(result.order(), 1);
    assert!((result.eigenvalues()[0] - 42.0).abs() < 1e-10);
    assert!((result.eigenvector(0)[0].abs() - 1.0).abs() < 1e-10);
}

#[test]
fn test_2x2_diagonal() {
    let m = matrix_from_dense(&[5.0, 0.0, 0.0, 3.0], 2);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert_eq!(result.order(), 2);
    assert!((result.eigenvalues()[0] - 5.0).abs() < 1e-10);
    assert!((result.eigenvalues()[1] - 3.0).abs() < 1e-10);
}

#[test]
fn test_2x2_symmetric() {
    // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
    let m = matrix_from_dense(&[2.0, 1.0, 1.0, 2.0], 2);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert!((result.eigenvalues()[0] - 3.0).abs() < 1e-10);
    assert!((result.eigenvalues()[1] - 1.0).abs() < 1e-10);

    // Check orthonormality.
    let v0 = result.eigenvector(0);
    let v1 = result.eigenvector(1);
    let dot: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
    assert!(dot.abs() < 1e-10, "eigenvectors should be orthogonal, dot = {dot}");
    let norm0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm0 - 1.0).abs() < 1e-10);
    assert!((norm1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_3x3_identity() {
    let m = matrix_from_dense(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert_eq!(result.order(), 3);
    for &ev in result.eigenvalues() {
        assert!((ev - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_3x3_known_matrix() {
    // [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    #[rustfmt::skip]
    let vals = [
        4.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    ];
    let m = matrix_from_dense(&vals, 3);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();

    // Verify Av = λv for each eigenpair.
    for k in 0..3 {
        let lambda = result.eigenvalues()[k];
        let v = result.eigenvector(k);
        for i in 0..3 {
            let av_i: f64 = (0..3).map(|j| vals[i * 3 + j] * v[j]).sum();
            assert!(
                (av_i - lambda * v[i]).abs() < 1e-10,
                "Av != λv for eigenpair {k}, row {i}: Av_i={av_i}, λv_i={}",
                lambda * v[i]
            );
        }
    }
}

#[test]
fn test_reconstruction() {
    // A = VΛVᵀ
    #[rustfmt::skip]
    let vals = [
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0,
    ];
    let m = matrix_from_dense(&vals, 3);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    let n = 3;

    for i in 0..n {
        for j in 0..n {
            let mut reconstructed = 0.0;
            for k in 0..n {
                let component_i = result.eigenvector(k)[i];
                let component_j = result.eigenvector(k)[j];
                reconstructed += result.eigenvalues()[k] * component_i * component_j;
            }
            assert!(
                (reconstructed - vals[i * n + j]).abs() < 1e-8,
                "Reconstruction failed at ({i}, {j}): expected {}, got {reconstructed}",
                vals[i * n + j]
            );
        }
    }
}

#[test]
fn test_orthonormality() {
    #[rustfmt::skip]
    let vals = [
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0,
    ];
    let m = matrix_from_dense(&vals, 3);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    let n = 3;

    // VᵀV should be the identity.
    for k in 0..n {
        for l in 0..n {
            let dot: f64 =
                (0..n).map(|i| result.eigenvector(k)[i] * result.eigenvector(l)[i]).sum();
            let expected = if k == l { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-10, "VᵀV[{k},{l}] = {dot}, expected {expected}");
        }
    }
}

#[test]
fn test_eigenvalue_ordering() {
    #[rustfmt::skip]
    let vals = [
        1.0, 0.5,
        0.5, 3.0,
    ];
    let m = matrix_from_dense(&vals, 2);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert!(result.eigenvalues()[0] >= result.eigenvalues()[1]);
}

#[test]
fn test_determinism() {
    #[rustfmt::skip]
    let vals = [
        4.0, 2.0, 1.0,
        2.0, 5.0, 3.0,
        1.0, 3.0, 6.0,
    ];
    let m = matrix_from_dense(&vals, 3);
    let config = JacobiConfig::default();
    let r1 = m.jacobi(&config).unwrap();
    let r2 = m.jacobi(&config).unwrap();
    assert_eq!(r1.eigenvalues(), r2.eigenvalues());
    assert_eq!(r1.eigenvectors_flat(), r2.eigenvectors_flat());
}

#[test]
fn test_sparse_symmetric_matrix() {
    // Build via sparse edges: [[2, 1], [1, 2]].
    let m = matrix_from_flat(&[2.0, 1.0, 1.0, 2.0], 2);
    let result = m.jacobi(&JacobiConfig::default()).unwrap();
    assert!((result.eigenvalues()[0] - 3.0).abs() < 1e-10);
    assert!((result.eigenvalues()[1] - 1.0).abs() < 1e-10);
}

// ============================================================================
// Error tests
// ============================================================================

// Note: NonSquareMatrix cannot be triggered via PaddedMatrix2D because it
// always pads to a square matrix. The validation exists for custom
// DenseValuedMatrix2D implementations.

#[test]
fn test_non_symmetric_matrix() {
    let m = matrix_from_dense(&[1.0, 2.0, 100.0, 4.0], 2);
    let result = m.jacobi(&JacobiConfig::default());
    assert!(matches!(result, Err(JacobiError::NonSymmetricMatrix { row: 0, column: 1 })));
}

#[test]
fn test_non_finite_nan() {
    let m = matrix_from_dense(&[1.0, f64::NAN, f64::NAN, 1.0], 2);
    let result = m.jacobi(&JacobiConfig::default());
    assert!(matches!(result, Err(JacobiError::NonFiniteValue { .. })));
}

#[test]
fn test_non_finite_inf() {
    let m = matrix_from_dense(&[1.0, f64::INFINITY, f64::INFINITY, 1.0], 2);
    let result = m.jacobi(&JacobiConfig::default());
    assert!(matches!(result, Err(JacobiError::NonFiniteValue { .. })));
}

#[test]
fn test_empty_matrix() {
    let inner = GenericEdgesBuilder::<_, TestValCSR>::default()
        .expected_number_of_edges(0)
        .expected_shape((0, 0))
        .edges(core::iter::empty())
        .build()
        .unwrap();
    let padded = PaddedMatrix2D::new(
        inner,
        Box::new(|_: (usize, usize)| 0.0) as Box<dyn Fn((usize, usize)) -> f64>,
    )
    .unwrap();
    let result = padded.jacobi(&JacobiConfig::default());
    assert!(matches!(result, Err(JacobiError::EmptyMatrix)));
}

#[test]
fn test_zero_tolerance() {
    let m = matrix_from_dense(&[1.0], 1);
    let config = JacobiConfig { tolerance: 0.0, max_sweeps: 100 };
    let result = m.jacobi(&config);
    assert!(matches!(result, Err(JacobiError::InvalidTolerance)));
}

#[test]
fn test_negative_tolerance() {
    let m = matrix_from_dense(&[1.0], 1);
    let config = JacobiConfig { tolerance: -1.0, max_sweeps: 100 };
    let result = m.jacobi(&config);
    assert!(matches!(result, Err(JacobiError::InvalidTolerance)));
}

#[test]
fn test_nan_tolerance() {
    let m = matrix_from_dense(&[1.0], 1);
    let config = JacobiConfig { tolerance: f64::NAN, max_sweeps: 100 };
    let result = m.jacobi(&config);
    assert!(matches!(result, Err(JacobiError::InvalidTolerance)));
}

#[test]
fn test_zero_max_sweeps() {
    let m = matrix_from_dense(&[1.0], 1);
    let config = JacobiConfig { tolerance: 1e-12, max_sweeps: 0 };
    let result = m.jacobi(&config);
    assert!(matches!(result, Err(JacobiError::InvalidMaxSweeps)));
}
