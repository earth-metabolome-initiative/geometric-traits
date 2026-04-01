//! Additional LAP and Louvain coverage by algorithm domain.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    prelude::*,
    traits::LAPJV,
};

#[test]
fn test_lapjv_conflict_heavy() {
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0, 1.0],
        [50.0, 50.0, 50.0, 1.0, 50.0],
    ])
    .unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_lapjv_asymmetric_costs() {
    let m: ValuedCSR2D<u8, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]).unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 3);
    assert!(result.contains(&(0, 0)));
    assert!(result.contains(&(1, 1)));
    assert!(result.contains(&(2, 2)));
}

#[test]
fn test_lapjv_chain_conflicts() {
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 99.0, 99.0, 99.0, 99.0],
        [99.0, 1.0, 2.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 1.0, 2.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 1.0, 2.0, 99.0],
        [99.0, 99.0, 99.0, 99.0, 1.0, 2.0],
        [2.0, 99.0, 99.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 6);
}

#[test]
fn test_lapjv_identity_preference() {
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0, 99.0],
        [99.0, 1.0, 99.0, 99.0],
        [99.0, 99.0, 1.0, 99.0],
        [99.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 4);
    assert!(result.contains(&(0, 0)));
    assert!(result.contains(&(1, 1)));
    assert!(result.contains(&(2, 2)));
    assert!(result.contains(&(3, 3)));
}

#[test]
fn test_louvain_non_square_matrix() {
    let m: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let result: Result<LouvainResult<usize>, _> = m.louvain(&LouvainConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_louvain_config_validation() {
    let config = LouvainConfig { resolution: -1.0, ..Default::default() };
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([[1.0, 2.0], [2.0, 1.0]]).unwrap();
    let result: Result<LouvainResult<usize>, _> = m.louvain(&config);
    assert!(result.is_err());
}

#[test]
fn test_louvain_error_display_non_square() {
    let err = ModularityError::NonSquareMatrix { rows: 3, columns: 5 };
    let display = format!("{err}");
    assert!(display.contains("square"));
    assert!(display.contains('3'));
    assert!(display.contains('5'));
}

#[test]
fn test_louvain_error_display_unrepresentable() {
    let err = ModularityError::UnrepresentableWeight { source_id: 0, destination_id: 1 };
    let display = format!("{err}");
    assert!(display.contains("cannot be represented"));
}

#[test]
fn test_louvain_error_display_all_variants() {
    let variants: Vec<ModularityError> = vec![
        ModularityError::InvalidResolution,
        ModularityError::InvalidModularityThreshold,
        ModularityError::InvalidMaxLevels,
        ModularityError::InvalidMaxLocalPasses,
        ModularityError::InvalidMaxRefinementPasses,
        ModularityError::InvalidTheta,
        ModularityError::NonSquareMatrix { rows: 2, columns: 3 },
        ModularityError::UnrepresentableWeight { source_id: 0, destination_id: 1 },
        ModularityError::NonFiniteWeight { source_id: 0, destination_id: 1 },
        ModularityError::NonPositiveWeight { source_id: 0, destination_id: 1 },
        ModularityError::NonSymmetricEdge { source_id: 0, destination_id: 1 },
        ModularityError::TooManyCommunities,
    ];
    for err in &variants {
        assert!(!format!("{err}").is_empty());
        assert!(!format!("{err:?}").is_empty());
    }
}
