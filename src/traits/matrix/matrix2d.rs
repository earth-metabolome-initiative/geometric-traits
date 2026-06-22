//! Trait defining a bidimensional matrix.

use multi_ranged::{SimpleRange, Step};
use num_traits::{AsPrimitive, Zero};

use super::SquareMatrix;
use crate::traits::{Matrix, PositiveInteger};

/// Trait defining a bidimensional matrix.
pub trait Matrix2D:
    Matrix<Coordinates = (<Self as Matrix2D>::RowIndex, <Self as Matrix2D>::ColumnIndex)>
{
    /// Type of the row index.
    type RowIndex: Step + PositiveInteger + AsPrimitive<usize>;
    /// Type of the column index.
    type ColumnIndex: Step + PositiveInteger + AsPrimitive<usize>;

    /// Returns the number of rows of the matrix.
    fn number_of_rows(&self) -> Self::RowIndex;

    /// Returns the number of columns of the matrix.
    fn number_of_columns(&self) -> Self::ColumnIndex;

    #[inline]
    /// Returns an iterator over the rows of the matrix.
    fn row_indices(&self) -> SimpleRange<Self::RowIndex> {
        let number_of_rows = self.number_of_rows();
        if number_of_rows.is_zero() {
            SimpleRange::default()
        } else {
            SimpleRange::try_from((Self::RowIndex::zero(), number_of_rows.prev()))
                .expect("zero <= number_of_rows - 1 holds when number_of_rows > 0")
        }
    }

    #[inline]
    /// Returns an iterator over the columns of the matrix.
    fn column_indices(&self) -> SimpleRange<Self::ColumnIndex> {
        let number_of_columns = self.number_of_columns();
        if number_of_columns.is_zero() {
            SimpleRange::default()
        } else {
            SimpleRange::try_from((Self::ColumnIndex::zero(), number_of_columns.prev()))
                .expect("zero <= number_of_columns - 1 holds when number_of_columns > 0")
        }
    }
}

impl<M: Matrix2D> Matrix2D for &M {
    type RowIndex = M::RowIndex;
    type ColumnIndex = M::ColumnIndex;

    #[inline]
    fn number_of_rows(&self) -> Self::RowIndex {
        (*self).number_of_rows()
    }

    #[inline]
    fn number_of_columns(&self) -> Self::ColumnIndex {
        (*self).number_of_columns()
    }
}

/// Trait defining a bidimensional matrix which can return references to
/// the number of rows and columns.
pub trait Matrix2DRef: Matrix2D {
    /// Returns a reference to the number of rows of the matrix.
    fn number_of_rows_ref(&self) -> &Self::RowIndex;

    /// Returns a reference to the number of columns of the matrix.
    fn number_of_columns_ref(&self) -> &Self::ColumnIndex;
}

/// Trait defining a transposable bidimensional matrix.
pub trait TransposableMatrix2D<
    Transposed: Matrix2D<RowIndex = Self::ColumnIndex, ColumnIndex = Self::RowIndex>,
>: Matrix2D
{
    /// Returns the transpose of the matrix.
    fn transpose(&self) -> Transposed;
}

/// Trait defining a matrix which supports efficient operations on columns.
pub trait BiMatrix2D: Matrix2D {
    /// The type of the underlying matrix.
    type Matrix: Matrix2D<RowIndex = Self::RowIndex, ColumnIndex = Self::ColumnIndex>;
    /// The type of the underlying transposed matrix.
    type TransposedMatrix: Matrix2D<RowIndex = Self::ColumnIndex, ColumnIndex = Self::RowIndex>;

    /// Returns a reference to the underlying matrix.
    fn matrix(&self) -> &Self::Matrix;

    /// Returns a reference to the underlying transposed matrix.
    fn transposed(&self) -> &Self::TransposedMatrix;
}

/// Trait defining a symmetric matrix.
pub trait SymmetricMatrix2D:
    SquareMatrix
    + BiMatrix2D<
        Matrix = <Self as SymmetricMatrix2D>::SymmetricMatrix,
        TransposedMatrix = <Self as SymmetricMatrix2D>::SymmetricMatrix,
    >
{
    /// The underlying symmetric matrix type.
    type SymmetricMatrix: Matrix2D<RowIndex = Self::Index, ColumnIndex = Self::Index>;
}

impl<M> SymmetricMatrix2D for M
where
    M: SquareMatrix + BiMatrix2D<TransposedMatrix = <M as BiMatrix2D>::Matrix>,
    M::Matrix: Matrix2D<RowIndex = Self::RowIndex, ColumnIndex = Self::ColumnIndex>,
{
    type SymmetricMatrix = M::Matrix;
}
