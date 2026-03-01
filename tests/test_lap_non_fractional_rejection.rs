//! Regression tests ensuring LAP routines reject non-fractional value domains.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{PaddedMatrix2D, ValuedCSR2D},
    traits::{Jaqaman, LAPError, LAPJV, LAPJVError, LAPMOD, LAPMODError, SparseLAPJV},
};

#[test]
fn test_lapjv_rejects_non_fractional_values() {
    let csr: ValuedCSR2D<u8, u8, u8, u8> = ValuedCSR2D::try_from([[3_u8, 1_u8], [2_u8, 4_u8]]).unwrap();
    let padded: PaddedMatrix2D<_, _> = PaddedMatrix2D::new(csr, |_: (u8, u8)| 150_u8).unwrap();

    assert_eq!(
        padded.lapjv(200_u8),
        Err(LAPJVError::NonFractionalValueTypeUnsupported)
    );
}

#[test]
fn test_sparse_lapjv_rejects_non_fractional_values() {
    let csr: ValuedCSR2D<u8, u8, u8, u8> = ValuedCSR2D::try_from([[3_u8, 1_u8], [2_u8, 4_u8]]).unwrap();

    assert_eq!(
        csr.sparse_lapjv(150_u8, 200_u8),
        Err(LAPError::NonFractionalValueTypeUnsupported)
    );
}

#[test]
fn test_lapmod_rejects_non_fractional_values() {
    let csr: ValuedCSR2D<u8, u8, u8, u8> = ValuedCSR2D::try_from([[3_u8, 1_u8], [2_u8, 4_u8]]).unwrap();

    assert_eq!(
        csr.lapmod(200_u8),
        Err(LAPMODError::NonFractionalValueTypeUnsupported)
    );
}

#[test]
fn test_jaqaman_rejects_non_fractional_values() {
    let csr_u64: ValuedCSR2D<u8, u8, u8, u64> =
        ValuedCSR2D::try_from([[1_u64, 2_u64], [3_u64, 4_u64]]).unwrap();
    assert_eq!(
        csr_u64.jaqaman(100_u64, 200_u64),
        Err(LAPError::NonFractionalValueTypeUnsupported)
    );

    let csr_u8: ValuedCSR2D<u8, u8, u8, u8> =
        ValuedCSR2D::try_from([[1_u8, 2_u8], [3_u8, 4_u8]]).unwrap();
    assert_eq!(
        csr_u8.jaqaman(10_u8, 20_u8),
        Err(LAPError::NonFractionalValueTypeUnsupported)
    );
}
