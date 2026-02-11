//! Enumeration for the errors associated with the CSR data structure.

use core::fmt::Debug;

#[cfg(feature = "alloc")]
use super::{CSR2D, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D, ValuedCSR2D};
use crate::traits::Matrix2D;

/// Enumeration for the errors associated with the CSR data structure.
pub enum Error<M: Matrix2D> {
    /// Mutability error.
    Mutability(MutabilityError<M>),
}

impl<M: Matrix2D> Debug for Error<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        <Self as core::fmt::Display>::fmt(self, f)
    }
}

impl<M: Matrix2D> core::error::Error for Error<M> {}

impl<M: Matrix2D> core::fmt::Display for Error<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Error::Mutability(error) => write!(f, "Mutability error: {error}"),
        }
    }
}

#[derive(PartialEq, Eq)]
/// Enumeration for the errors associated with failed mutable operations.
pub enum MutabilityError<M: Matrix2D + ?Sized> {
    /// Unexpected coordinate.
    UnorderedCoordinate(M::Coordinates),
    /// Duplicated entry.
    DuplicatedEntry(M::Coordinates),
    /// Entry out of bounds.
    OutOfBounds(M::Coordinates, M::Coordinates, &'static str),
    /// When the row index type has been maxed out and it cannot
    /// be incremented anymore.
    MaxedOutRowIndex,
    /// When the column index type has been maxed out and it cannot
    /// be incremented anymore.
    MaxedOutColumnIndex,
    /// When the sparse index type has been maxed out and it cannot
    /// be incremented anymore.
    MaxedOutSparseIndex,
    /// When a requested shape to apply is smaller than the current shape.
    IncompatibleShape,
}

impl<M: Matrix2D> Debug for MutabilityError<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        <Self as core::fmt::Display>::fmt(self, f)
    }
}

impl<M: Matrix2D> core::error::Error for MutabilityError<M> {}

impl<M: Matrix2D> From<MutabilityError<M>> for Error<M> {
    fn from(error: MutabilityError<M>) -> Self {
        Error::Mutability(error)
    }
}

impl<M: Matrix2D> core::fmt::Display for MutabilityError<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            MutabilityError::UnorderedCoordinate(coordinates) => {
                write!(f, "Unordered coordinate: {coordinates:?}")
            }
            MutabilityError::DuplicatedEntry(coordinates) => {
                write!(f, "Duplicated entry: {coordinates:?}")
            }
            MutabilityError::OutOfBounds(coordinates, boundaries, context) => {
                write!(
                    f,
                    "Entry out of expected bounds: {coordinates:?}, expected within: {boundaries:?} - {context}"
                )
            }
            MutabilityError::MaxedOutRowIndex => {
                write!(f, "Row index type has been maxed out")
            }
            MutabilityError::MaxedOutColumnIndex => {
                write!(f, "Column index type has been maxed out")
            }
            MutabilityError::MaxedOutSparseIndex => {
                write!(f, "Sparse index type has been maxed out")
            }
            MutabilityError::IncompatibleShape => {
                write!(f, "Requested shape is smaller than the current shape")
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl<M> From<MutabilityError<SquareCSR2D<M>>> for MutabilityError<UpperTriangularCSR2D<M>>
where
    M: Matrix2D,
{
    fn from(error: MutabilityError<SquareCSR2D<M>>) -> Self {
        match error {
            MutabilityError::UnorderedCoordinate(coordinates) => {
                MutabilityError::UnorderedCoordinate(coordinates)
            }
            MutabilityError::DuplicatedEntry(coordinates) => {
                MutabilityError::DuplicatedEntry(coordinates)
            }
            MutabilityError::OutOfBounds(coordinates, boundaries, context) => {
                MutabilityError::OutOfBounds(coordinates, boundaries, context)
            }
            MutabilityError::MaxedOutRowIndex => MutabilityError::MaxedOutRowIndex,
            MutabilityError::MaxedOutColumnIndex => MutabilityError::MaxedOutColumnIndex,
            MutabilityError::MaxedOutSparseIndex => MutabilityError::MaxedOutSparseIndex,
            MutabilityError::IncompatibleShape => MutabilityError::IncompatibleShape,
        }
    }
}

#[cfg(feature = "alloc")]
impl<M> From<MutabilityError<UpperTriangularCSR2D<M>>> for MutabilityError<SymmetricCSR2D<M>>
where
    M: Matrix2D,
{
    fn from(error: MutabilityError<UpperTriangularCSR2D<M>>) -> Self {
        match error {
            MutabilityError::UnorderedCoordinate(coordinates) => {
                MutabilityError::UnorderedCoordinate(coordinates)
            }
            MutabilityError::DuplicatedEntry(coordinates) => {
                MutabilityError::DuplicatedEntry(coordinates)
            }
            MutabilityError::OutOfBounds(coordinates, boundaries, context) => {
                MutabilityError::OutOfBounds(coordinates, boundaries, context)
            }
            MutabilityError::MaxedOutRowIndex => MutabilityError::MaxedOutRowIndex,
            MutabilityError::MaxedOutColumnIndex => MutabilityError::MaxedOutColumnIndex,
            MutabilityError::MaxedOutSparseIndex => MutabilityError::MaxedOutSparseIndex,
            MutabilityError::IncompatibleShape => MutabilityError::IncompatibleShape,
        }
    }
}

#[cfg(feature = "alloc")]
impl<M> From<MutabilityError<M>> for MutabilityError<SquareCSR2D<M>>
where
    M: Matrix2D,
{
    fn from(error: MutabilityError<M>) -> Self {
        match error {
            MutabilityError::UnorderedCoordinate(coordinates) => {
                MutabilityError::UnorderedCoordinate(coordinates)
            }
            MutabilityError::DuplicatedEntry(coordinates) => {
                MutabilityError::DuplicatedEntry(coordinates)
            }
            MutabilityError::OutOfBounds(coordinates, boundaries, context) => {
                MutabilityError::OutOfBounds(coordinates, boundaries, context)
            }
            MutabilityError::MaxedOutRowIndex => MutabilityError::MaxedOutRowIndex,
            MutabilityError::MaxedOutColumnIndex => MutabilityError::MaxedOutColumnIndex,
            MutabilityError::MaxedOutSparseIndex => MutabilityError::MaxedOutSparseIndex,
            MutabilityError::IncompatibleShape => MutabilityError::IncompatibleShape,
        }
    }
}

#[cfg(feature = "alloc")]
impl<SparseIndex, RowIndex, ColumnIndex, Value>
    From<MutabilityError<CSR2D<SparseIndex, RowIndex, ColumnIndex>>>
    for MutabilityError<ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>>
where
    CSR2D<SparseIndex, RowIndex, ColumnIndex>:
        Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
    ValuedCSR2D<SparseIndex, RowIndex, ColumnIndex, Value>:
        Matrix2D<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    fn from(error: MutabilityError<CSR2D<SparseIndex, RowIndex, ColumnIndex>>) -> Self {
        match error {
            MutabilityError::UnorderedCoordinate(coordinates) => {
                MutabilityError::UnorderedCoordinate(coordinates)
            }
            MutabilityError::DuplicatedEntry(coordinates) => {
                MutabilityError::DuplicatedEntry(coordinates)
            }
            MutabilityError::OutOfBounds(coordinates, boundaries, context) => {
                MutabilityError::OutOfBounds(coordinates, boundaries, context)
            }
            MutabilityError::MaxedOutRowIndex => MutabilityError::MaxedOutRowIndex,
            MutabilityError::MaxedOutColumnIndex => MutabilityError::MaxedOutColumnIndex,
            MutabilityError::MaxedOutSparseIndex => MutabilityError::MaxedOutSparseIndex,
            MutabilityError::IncompatibleShape => MutabilityError::IncompatibleShape,
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::string::ToString;

    use super::*;

    // Simple test matrix type for testing error display
    #[derive(PartialEq, Eq)]
    struct TestMatrix;

    impl Matrix2D for TestMatrix {
        type RowIndex = usize;
        type ColumnIndex = usize;

        fn number_of_rows(&self) -> Self::RowIndex {
            0
        }

        fn number_of_columns(&self) -> Self::ColumnIndex {
            0
        }
    }

    impl crate::traits::Matrix for TestMatrix {
        type Coordinates = (usize, usize);

        fn shape(&self) -> alloc::vec::Vec<usize> {
            alloc::vec![0, 0]
        }
    }

    #[test]
    fn test_mutability_error_unordered_coordinate_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::UnorderedCoordinate((1, 2));
        let display = error.to_string();
        assert!(display.contains("Unordered coordinate"));
        assert!(display.contains("(1, 2)"));
    }

    #[test]
    fn test_mutability_error_duplicated_entry_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::DuplicatedEntry((3, 4));
        let display = error.to_string();
        assert!(display.contains("Duplicated entry"));
        assert!(display.contains("(3, 4)"));
    }

    #[test]
    fn test_mutability_error_out_of_bounds_display() {
        let error: MutabilityError<TestMatrix> =
            MutabilityError::OutOfBounds((5, 6), (10, 10), "test context");
        let display = error.to_string();
        assert!(display.contains("out of"));
        assert!(display.contains("bounds"));
        assert!(display.contains("test context"));
    }

    #[test]
    fn test_mutability_error_maxed_out_row_index_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::MaxedOutRowIndex;
        let display = error.to_string();
        assert!(display.contains("Row index"));
        assert!(display.contains("maxed out"));
    }

    #[test]
    fn test_mutability_error_maxed_out_column_index_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::MaxedOutColumnIndex;
        let display = error.to_string();
        assert!(display.contains("Column index"));
        assert!(display.contains("maxed out"));
    }

    #[test]
    fn test_mutability_error_maxed_out_sparse_index_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::MaxedOutSparseIndex;
        let display = error.to_string();
        assert!(display.contains("Sparse index"));
        assert!(display.contains("maxed out"));
    }

    #[test]
    fn test_mutability_error_incompatible_shape_display() {
        let error: MutabilityError<TestMatrix> = MutabilityError::IncompatibleShape;
        let display = error.to_string();
        assert!(display.contains("shape"));
        assert!(display.contains("smaller"));
    }

    #[test]
    fn test_error_display() {
        let mutability_error: MutabilityError<TestMatrix> =
            MutabilityError::UnorderedCoordinate((1, 2));
        let error: Error<TestMatrix> = Error::Mutability(mutability_error);
        let display = error.to_string();
        assert!(display.contains("Mutability error"));
    }

    #[test]
    fn test_error_from_mutability_error() {
        let mutability_error: MutabilityError<TestMatrix> =
            MutabilityError::DuplicatedEntry((1, 2));
        let error: Error<TestMatrix> = mutability_error.into();
        match error {
            Error::Mutability(inner) => {
                assert_eq!(inner, MutabilityError::DuplicatedEntry((1, 2)));
            }
        }
    }

    #[test]
    fn test_mutability_error_equality() {
        let error1: MutabilityError<TestMatrix> = MutabilityError::UnorderedCoordinate((1, 2));
        let error2: MutabilityError<TestMatrix> = MutabilityError::UnorderedCoordinate((1, 2));
        let error3: MutabilityError<TestMatrix> = MutabilityError::UnorderedCoordinate((3, 4));

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_mutability_error_debug() {
        let error: MutabilityError<TestMatrix> = MutabilityError::MaxedOutRowIndex;
        let debug = alloc::format!("{error:?}");
        assert!(debug.contains("Row index"));
    }
}
