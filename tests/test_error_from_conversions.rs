//! Tests for MutabilityError From conversions between matrix wrapper types:
//! SquareCSR2D -> UpperTriangularCSR2D, UpperTriangularCSR2D -> SymmetricCSR2D,
//! M -> SquareCSR2D, CSR2D -> ValuedCSR2D.
#![cfg(feature = "std")]

use geometric_traits::impls::{
    CSR2D, MutabilityError, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D, ValuedCSR2D,
};

type TestCSR = CSR2D<usize, usize, usize>;
type TestSquareCSR = SquareCSR2D<TestCSR>;
type TestUpperTriCSR = UpperTriangularCSR2D<TestCSR>;
type TestSymCSR = SymmetricCSR2D<TestCSR>;
type TestValCSR = ValuedCSR2D<usize, usize, usize, f64>;

// ============================================================================
// From<MutabilityError<SquareCSR2D<M>>> for
// MutabilityError<UpperTriangularCSR2D<M>> (UpperTriangularCSR2D doesn't impl
// PartialEq, so use matches!)
// ============================================================================

#[test]
fn test_square_to_upper_tri_unordered() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::UnorderedCoordinate((1, 2))));
}

#[test]
fn test_square_to_upper_tri_duplicated() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::DuplicatedEntry((3, 4))));
}

#[test]
fn test_square_to_upper_tri_out_of_bounds() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx");
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::OutOfBounds(..)));
}

#[test]
fn test_square_to_upper_tri_maxed_row() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::MaxedOutRowIndex;
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutRowIndex));
}

#[test]
fn test_square_to_upper_tri_maxed_col() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::MaxedOutColumnIndex;
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutColumnIndex));
}

#[test]
fn test_square_to_upper_tri_maxed_sparse() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::MaxedOutSparseIndex;
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutSparseIndex));
}

#[test]
fn test_square_to_upper_tri_incompatible() {
    let src: MutabilityError<TestSquareCSR> = MutabilityError::IncompatibleShape;
    let dst: MutabilityError<TestUpperTriCSR> = src.into();
    assert!(matches!(dst, MutabilityError::IncompatibleShape));
}

// ============================================================================
// From<MutabilityError<UpperTriangularCSR2D<M>>> for
// MutabilityError<SymmetricCSR2D<M>> (SymmetricCSR2D impls PartialEq, so
// assert_eq works)
// ============================================================================

#[test]
fn test_upper_tri_to_sym_unordered() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::UnorderedCoordinate((1, 2)));
}

#[test]
fn test_upper_tri_to_sym_duplicated() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::DuplicatedEntry((3, 4)));
}

#[test]
fn test_upper_tri_to_sym_out_of_bounds() {
    let src: MutabilityError<TestUpperTriCSR> =
        MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx");
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx"));
}

#[test]
fn test_upper_tri_to_sym_maxed_row() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::MaxedOutRowIndex;
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutRowIndex);
}

#[test]
fn test_upper_tri_to_sym_maxed_col() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::MaxedOutColumnIndex;
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutColumnIndex);
}

#[test]
fn test_upper_tri_to_sym_maxed_sparse() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::MaxedOutSparseIndex;
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutSparseIndex);
}

#[test]
fn test_upper_tri_to_sym_incompatible() {
    let src: MutabilityError<TestUpperTriCSR> = MutabilityError::IncompatibleShape;
    let dst: MutabilityError<TestSymCSR> = src.into();
    assert_eq!(dst, MutabilityError::IncompatibleShape);
}

// ============================================================================
// From<MutabilityError<M>> for MutabilityError<SquareCSR2D<M>>
// (SquareCSR2D impls PartialEq)
// ============================================================================

#[test]
fn test_csr_to_square_unordered() {
    let src: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::UnorderedCoordinate((1, 2)));
}

#[test]
fn test_csr_to_square_duplicated() {
    let src: MutabilityError<TestCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::DuplicatedEntry((3, 4)));
}

#[test]
fn test_csr_to_square_out_of_bounds() {
    let src: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx");
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx"));
}

#[test]
fn test_csr_to_square_maxed_row() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutRowIndex);
}

#[test]
fn test_csr_to_square_maxed_col() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutColumnIndex);
}

#[test]
fn test_csr_to_square_maxed_sparse() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::MaxedOutSparseIndex);
}

#[test]
fn test_csr_to_square_incompatible() {
    let src: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let dst: MutabilityError<TestSquareCSR> = src.into();
    assert_eq!(dst, MutabilityError::IncompatibleShape);
}

// ============================================================================
// From<MutabilityError<CSR2D>> for MutabilityError<ValuedCSR2D>
// (ValuedCSR2D doesn't impl PartialEq, so use matches!)
// ============================================================================

#[test]
fn test_csr_to_valued_unordered() {
    let src: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::UnorderedCoordinate((1, 2))));
}

#[test]
fn test_csr_to_valued_duplicated() {
    let src: MutabilityError<TestCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::DuplicatedEntry((3, 4))));
}

#[test]
fn test_csr_to_valued_out_of_bounds() {
    let src: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((5, 6), (10, 10), "ctx");
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::OutOfBounds(..)));
}

#[test]
fn test_csr_to_valued_maxed_row() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutRowIndex));
}

#[test]
fn test_csr_to_valued_maxed_col() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutColumnIndex));
}

#[test]
fn test_csr_to_valued_maxed_sparse() {
    let src: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::MaxedOutSparseIndex));
}

#[test]
fn test_csr_to_valued_incompatible() {
    let src: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let dst: MutabilityError<TestValCSR> = src.into();
    assert!(matches!(dst, MutabilityError::IncompatibleShape));
}
