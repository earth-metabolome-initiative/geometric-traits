//! Tests for LouvainError Display, From conversions, and validation paths.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LouvainConfig, LouvainError},
};

// ============================================================================
// LouvainError Display
// ============================================================================

#[test]
fn test_louvain_error_invalid_resolution() {
    let e = LouvainError::InvalidResolution;
    let s = format!("{e}");
    assert!(s.contains("resolution"));
    assert!(s.contains("finite"));
}

#[test]
fn test_louvain_error_invalid_modularity_threshold() {
    let e = LouvainError::InvalidModularityThreshold;
    let s = format!("{e}");
    assert!(s.contains("modularity threshold"));
}

#[test]
fn test_louvain_error_invalid_max_levels() {
    let e = LouvainError::InvalidMaxLevels;
    let s = format!("{e}");
    assert!(s.contains("maximum number of levels"));
}

#[test]
fn test_louvain_error_invalid_max_local_passes() {
    let e = LouvainError::InvalidMaxLocalPasses;
    let s = format!("{e}");
    assert!(s.contains("maximum number of local passes"));
}

#[test]
fn test_louvain_error_non_square_matrix() {
    let e = LouvainError::NonSquareMatrix { rows: 3, columns: 5 };
    let s = format!("{e}");
    assert!(s.contains("square"));
    assert!(s.contains('3'));
    assert!(s.contains('5'));
}

#[test]
fn test_louvain_error_unrepresentable_weight() {
    let e = LouvainError::UnrepresentableWeight { source_id: 1, destination_id: 2 };
    let s = format!("{e}");
    assert!(s.contains("represented"));
    assert!(s.contains("(1, 2)"));
}

#[test]
fn test_louvain_error_non_finite_weight() {
    let e = LouvainError::NonFiniteWeight { source_id: 0, destination_id: 3 };
    let s = format!("{e}");
    assert!(s.contains("non-finite"));
    assert!(s.contains("(0, 3)"));
}

#[test]
fn test_louvain_error_non_positive_weight() {
    let e = LouvainError::NonPositiveWeight { source_id: 2, destination_id: 4 };
    let s = format!("{e}");
    assert!(s.contains("non-positive"));
    assert!(s.contains("(2, 4)"));
}

#[test]
fn test_louvain_error_non_symmetric_edge() {
    let e = LouvainError::NonSymmetricEdge { source_id: 1, destination_id: 5 };
    let s = format!("{e}");
    assert!(s.contains("symmetric"));
    assert!(s.contains("(1, 5)"));
}

#[test]
fn test_louvain_error_too_many_communities() {
    let e = LouvainError::TooManyCommunities;
    let s = format!("{e}");
    assert!(s.contains("community marker type"));
}

#[test]
fn test_louvain_error_is_std_error() {
    fn check<E: std::error::Error>(_: E) {}
    check(LouvainError::InvalidResolution);
}

#[test]
fn test_louvain_error_debug_clone_eq() {
    let e = LouvainError::InvalidResolution;
    let e2 = e.clone();
    assert_eq!(e, e2);
    assert!(format!("{e:?}").contains("InvalidResolution"));
}

// ============================================================================
// Louvain validation via louvain() with invalid config
// ============================================================================

fn make_symmetric_2x2() -> ValuedCSR2D<usize, usize, usize, f64> {
    ValuedCSR2D::try_from([[1.0, 2.0], [2.0, 1.0]]).unwrap()
}

#[test]
fn test_louvain_invalid_resolution_zero() {
    let m = make_symmetric_2x2();
    let config = LouvainConfig { resolution: 0.0, ..LouvainConfig::default() };
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_invalid_resolution_nan() {
    let m = make_symmetric_2x2();
    let config = LouvainConfig { resolution: f64::NAN, ..LouvainConfig::default() };
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_invalid_modularity_threshold_negative() {
    let m = make_symmetric_2x2();
    let config = LouvainConfig { modularity_threshold: -0.1, ..LouvainConfig::default() };
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_invalid_max_levels_zero() {
    let m = make_symmetric_2x2();
    let config = LouvainConfig { max_levels: 0, ..LouvainConfig::default() };
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_invalid_max_local_passes_zero() {
    let m = make_symmetric_2x2();
    let config = LouvainConfig { max_local_passes: 0, ..LouvainConfig::default() };
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_non_square_matrix() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_non_finite_weight() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[f64::NAN, 1.0], [1.0, f64::NAN]]).unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_non_positive_weight() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[-1.0, 2.0], [2.0, -1.0]]).unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_non_symmetric() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 1.0]]).unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_successful_run() {
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [2.0, 1.0]]).unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config);
    assert!(result.is_ok());
    let res = result.unwrap();
    assert!(!res.levels().is_empty());
    let level = &res.levels()[0];
    assert_eq!(level.partition().len(), 2);
    assert!(level.modularity().is_finite());
}

#[test]
fn test_louvain_4_node_two_clusters() {
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 10.0, 0.1, 0.1],
        [10.0, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 10.0],
        [0.1, 0.1, 10.0, 1.0],
    ])
    .unwrap();
    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&m, &config).unwrap();
    let partition = result.levels().last().unwrap().partition();
    assert_eq!(partition[0], partition[1]);
    assert_eq!(partition[2], partition[3]);
    assert_ne!(partition[0], partition[2]);
}
