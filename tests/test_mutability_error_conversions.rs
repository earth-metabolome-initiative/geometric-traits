//! Coverage for mutability error conversions between matrix wrappers.
#![cfg(feature = "std")]

use geometric_traits::impls::{
    CSR2D, MutabilityError, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D, ValuedCSR2D,
};

type TestCSR = CSR2D<usize, usize, usize>;

#[test]
fn test_error_conversion_csr2d_to_square() {
    let err: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Unordered coordinate"));

    let err: MutabilityError<TestCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Duplicated entry"));

    let err: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((0, 0), (5, 5), "test");
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("out of expected bounds"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("shape"));
}

#[test]
fn test_error_conversion_square_to_upper_triangular() {
    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Unordered coordinate"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Duplicated entry"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> =
        MutabilityError::OutOfBounds((0, 0), (5, 5), "ctx");
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("out of expected bounds"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
}

#[test]
fn test_error_conversion_upper_triangular_to_symmetric() {
    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Unordered coordinate"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("Duplicated entry"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::OutOfBounds((0, 0), (5, 5), "sym");
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    assert!(format!("{converted}").contains("out of expected bounds"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
}

#[test]
fn test_error_conversion_csr2d_to_valued() {
    let err: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    assert!(format!("{converted}").contains("Unordered coordinate"));

    let err: MutabilityError<TestCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    assert!(format!("{converted}").contains("Duplicated entry"));

    let err: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((0, 0), (5, 5), "val");
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    assert!(format!("{converted}").contains("out of expected bounds"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
}
