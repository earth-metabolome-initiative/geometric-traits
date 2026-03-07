//! Tests for the classical MDS (Torgerson) trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::EdgesBuilder,
};

type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;
type TestPadded = PaddedMatrix2D<TestValCSR, Box<dyn Fn((usize, usize)) -> f64>>;

/// Build a dense n×n distance matrix from a flat row-major slice.
fn dist_matrix(values: &[f64], n: usize) -> TestPadded {
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

/// Compute pairwise Euclidean distance from flat coordinates (n points, k
/// dims).
fn pairwise_dist(coords: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sq = 0.0;
            for dim in 0..k {
                let diff = coords[i * k + dim] - coords[j * k + dim];
                sq += diff * diff;
            }
            d[i * n + j] = sq.sqrt();
        }
    }
    d
}

// ============================================================================
// Positive tests
// ============================================================================

#[test]
fn test_2_points_1d() {
    // 2 points at distance 5.0 → embed into 1D.
    #[rustfmt::skip]
    let d = [
        0.0, 5.0,
        5.0, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let config = MdsConfig { dimensions: 1, ..MdsConfig::default() };
    let result = m.classical_mds(&config).unwrap();

    assert_eq!(result.num_points(), 2);
    assert_eq!(result.dimensions(), 1);

    // Distance between the two embedded points should be ≈ 5.0.
    let embed_dist = (result.point(0)[0] - result.point(1)[0]).abs();
    assert!((embed_dist - 5.0).abs() < 1e-8, "expected distance 5.0, got {embed_dist}");
    assert!(result.stress() < 1e-6, "stress should be ≈ 0, got {}", result.stress());
}

#[test]
fn test_equilateral_triangle_2d() {
    // 3 points, all pairwise distances = 1.0.
    #[rustfmt::skip]
    let d = [
        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
    ];
    let m = dist_matrix(&d, 3);
    let config = MdsConfig::default(); // 2D
    let result = m.classical_mds(&config).unwrap();

    assert_eq!(result.num_points(), 3);
    assert_eq!(result.dimensions(), 2);

    // All pairwise distances should be ≈ 1.0.
    let embed_dists = pairwise_dist(result.coordinates_flat(), 3, 2);
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!(
                (embed_dists[i * 3 + j] - 1.0).abs() < 1e-6,
                "distance({i},{j}) = {}, expected 1.0",
                embed_dists[i * 3 + j]
            );
        }
    }
    assert!(result.stress() < 1e-6, "stress = {}", result.stress());
}

#[test]
fn test_unit_square_2d() {
    // 4 points forming a unit square.
    // Vertices: (0,0), (1,0), (1,1), (0,1)
    // d(0,1)=1, d(0,2)=√2, d(0,3)=1, d(1,2)=1, d(1,3)=√2, d(2,3)=1
    let s2 = core::f64::consts::SQRT_2;
    #[rustfmt::skip]
    let d = [
        0.0, 1.0,  s2, 1.0,
        1.0, 0.0, 1.0,  s2,
         s2, 1.0, 0.0, 1.0,
        1.0,  s2, 1.0, 0.0,
    ];
    let m = dist_matrix(&d, 4);
    let config = MdsConfig::default(); // 2D
    let result = m.classical_mds(&config).unwrap();

    assert_eq!(result.num_points(), 4);

    // Check pairwise distances match the input.
    let embed_dists = pairwise_dist(result.coordinates_flat(), 4, 2);
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert!(
                (embed_dists[i * 4 + j] - d[i * 4 + j]).abs() < 1e-6,
                "distance({i},{j}) = {}, expected {}",
                embed_dists[i * 4 + j],
                d[i * 4 + j]
            );
        }
    }
    assert!(result.stress() < 1e-6, "stress = {}", result.stress());
}

#[test]
fn test_regular_tetrahedron_3d() {
    // 4 points, all pairwise distances = 1.0.
    #[rustfmt::skip]
    let d = [
        0.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0,
    ];
    let m = dist_matrix(&d, 4);
    let config = MdsConfig { dimensions: 3, ..MdsConfig::default() };
    let result = m.classical_mds(&config).unwrap();

    assert_eq!(result.num_points(), 4);
    assert_eq!(result.dimensions(), 3);

    let embed_dists = pairwise_dist(result.coordinates_flat(), 4, 3);
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert!(
                (embed_dists[i * 4 + j] - 1.0).abs() < 1e-6,
                "distance({i},{j}) = {}, expected 1.0",
                embed_dists[i * 4 + j]
            );
        }
    }
    assert!(result.stress() < 1e-6, "stress = {}", result.stress());
}

#[test]
fn test_collinear_3_points_1d() {
    // 3 collinear points: A=0, B=1, C=3.
    #[rustfmt::skip]
    let distance_matrix_data = [
        0.0, 1.0, 3.0,
        1.0, 0.0, 2.0,
        3.0, 2.0, 0.0,
    ];
    let matrix = dist_matrix(&distance_matrix_data, 3);
    let config = MdsConfig { dimensions: 1, ..MdsConfig::default() };
    let result = matrix.classical_mds(&config).unwrap();

    assert_eq!(result.num_points(), 3);
    assert_eq!(result.dimensions(), 1);

    // Distances should be preserved.
    let first_position = result.point(0)[0];
    let second_position = result.point(1)[0];
    let third_position = result.point(2)[0];
    assert!(((first_position - second_position).abs() - 1.0).abs() < 1e-6, "d(A,B) ≈ 1");
    assert!(((second_position - third_position).abs() - 2.0).abs() < 1e-6, "d(B,C) ≈ 2");
    assert!(((first_position - third_position).abs() - 3.0).abs() < 1e-6, "d(A,C) ≈ 3");
    assert!(result.stress() < 1e-6, "stress = {}", result.stress());
}

#[test]
fn test_all_zero_distances() {
    // All distances zero → all points at origin.
    #[rustfmt::skip]
    let d = [
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let m = dist_matrix(&d, 3);
    let config = MdsConfig::default();
    let result = m.classical_mds(&config).unwrap();

    for i in 0..3 {
        for &c in result.point(i) {
            assert!(c.abs() < 1e-10, "expected all at origin, got {c}");
        }
    }
    assert!(result.stress().abs() < 1e-10);
}

#[test]
fn test_determinism() {
    #[rustfmt::skip]
    let d = [
        0.0, 1.0, 2.0,
        1.0, 0.0, 1.5,
        2.0, 1.5, 0.0,
    ];
    let m = dist_matrix(&d, 3);
    let config = MdsConfig::default();
    let r1 = m.classical_mds(&config).unwrap();
    let r2 = m.classical_mds(&config).unwrap();
    assert_eq!(r1.coordinates_flat(), r2.coordinates_flat());
    assert_eq!(r1.eigenvalues(), r2.eigenvalues());
    assert!((r1.stress() - r2.stress()).abs() <= f64::EPSILON);
}

#[test]
fn test_stress_near_zero_for_euclidean_input() {
    // Generate distances from known Euclidean points:
    // (0,0), (3,0), (0,4)
    let pts = [(0.0_f64, 0.0), (3.0, 0.0), (0.0, 4.0)];
    let n = 3;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = pts[i].0 - pts[j].0;
            let dy = pts[i].1 - pts[j].1;
            d[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let m = dist_matrix(&d, n);
    let config = MdsConfig::default();
    let result = m.classical_mds(&config).unwrap();
    assert!(result.stress() < 1e-6, "stress = {}", result.stress());

    // Verify pairwise distances.
    let embed_dists = pairwise_dist(result.coordinates_flat(), n, 2);
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                (embed_dists[i * n + j] - d[i * n + j]).abs() < 1e-6,
                "distance({i},{j}) = {}, expected {}",
                embed_dists[i * n + j],
                d[i * n + j]
            );
        }
    }
}

#[test]
fn test_eigenvalues_returned() {
    #[rustfmt::skip]
    let d = [
        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
    ];
    let m = dist_matrix(&d, 3);
    let config = MdsConfig::default();
    let result = m.classical_mds(&config).unwrap();
    assert_eq!(result.eigenvalues().len(), 2);
    // Eigenvalues should be finite.
    for &ev in result.eigenvalues() {
        assert!(ev.is_finite(), "non-finite eigenvalue: {ev}");
    }
}

// ============================================================================
// Error tests
// ============================================================================

// Note: NonSquareMatrix cannot be triggered via PaddedMatrix2D because it
// always pads to a square matrix. The validation exists for custom
// DenseValuedMatrix2D implementations.

#[test]
fn test_non_symmetric_matrix() {
    #[rustfmt::skip]
    let d = [
        0.0,  1.0,
        99.0, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::NonSymmetricMatrix { row: 0, column: 1 })));
}

#[test]
fn test_nan_value() {
    let d = [0.0, f64::NAN, f64::NAN, 0.0];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::NonFiniteValue { .. })));
}

#[test]
fn test_infinity_value() {
    let d = [0.0, f64::INFINITY, f64::INFINITY, 0.0];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::NonFiniteValue { .. })));
}

#[test]
fn test_negative_distance() {
    #[rustfmt::skip]
    let d = [
        0.0, -1.0,
       -1.0,  0.0,
    ];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::NegativeDistance { .. })));
}

#[test]
fn test_non_zero_diagonal() {
    #[rustfmt::skip]
    let d = [
        1.0, 2.0,
        2.0, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::DiagonalNotZero { index: 0, .. })));
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
    let result = padded.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::EmptyMatrix)));
}

#[test]
fn test_dimensions_zero() {
    #[rustfmt::skip]
    let d = [
        0.0, 1.0,
        1.0, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let config = MdsConfig { dimensions: 0, ..MdsConfig::default() };
    let result = m.classical_mds(&config);
    assert!(matches!(result, Err(MdsError::InvalidDimensions(0))));
}

#[test]
fn test_dimensions_greater_than_n() {
    #[rustfmt::skip]
    let d = [
        0.0, 1.0,
        1.0, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let config = MdsConfig { dimensions: 3, ..MdsConfig::default() };
    let result = m.classical_mds(&config);
    assert!(matches!(
        result,
        Err(MdsError::DimensionsExceedPoints { dimensions: 3, num_points: 2 })
    ));
}

#[test]
fn test_too_few_points() {
    // 1×1 matrix → TooFewPoints.
    let d = [0.0];
    let m = dist_matrix(&d, 1);
    let config = MdsConfig { dimensions: 1, ..MdsConfig::default() };
    let result = m.classical_mds(&config);
    assert!(matches!(result, Err(MdsError::TooFewPoints(1))));
}

#[test]
fn test_distance_too_large() {
    let big = 1e155;
    #[rustfmt::skip]
    let d = [
        0.0, big,
        big, 0.0,
    ];
    let m = dist_matrix(&d, 2);
    let result = m.classical_mds(&MdsConfig::default());
    assert!(matches!(result, Err(MdsError::DistanceTooLarge(_))));
}
