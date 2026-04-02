//! Submodule for the [`SquareMatrix`] trait.

use multi_ranged::Step;
use num_traits::AsPrimitive;

use super::{Matrix2D, SparseMatrix2D, SymmetricMatrix2D};
use crate::traits::PositiveInteger;

/// Trait defining a square matrix.
pub trait SquareMatrix: Matrix2D<RowIndex = Self::Index, ColumnIndex = Self::Index> {
    /// Type of the index for this matrix.
    type Index: Step + PositiveInteger + AsPrimitive<usize>;

    /// Returns the order of the matrix.
    fn order(&self) -> Self::Index;
}

impl<M: SquareMatrix> SquareMatrix for &M {
    type Index = M::Index;

    #[inline]
    fn order(&self) -> Self::Index {
        (*self).order()
    }
}

/// Trait defining a sparse square matrix.
pub trait SparseSquareMatrix: SquareMatrix + SparseMatrix2D {
    /// Returns the number of defined values in the main diagonal.
    fn number_of_defined_diagonal_values(&self) -> Self::Index;

    /// Returns whether the matrix is symmetric.
    ///
    /// The default implementation checks that every off-diagonal entry
    /// `(row, column)` has a matching reverse entry `(column, row)`.
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.row_indices().all(|row| {
            self.sparse_row(row).all(|column| row == column || self.has_entry(column, row))
        })
    }
}

impl<M: SparseSquareMatrix> SparseSquareMatrix for &M {
    #[inline]
    fn number_of_defined_diagonal_values(&self) -> Self::Index {
        (*self).number_of_defined_diagonal_values()
    }

    #[inline]
    fn is_symmetric(&self) -> bool {
        (*self).is_symmetric()
    }
}

/// Trait defining a matrix that can be symmetrized.
pub trait Symmetrize<M: SymmetricMatrix2D>: SquareMatrix {
    /// Returns the symmetrized version of the matrix.
    fn symmetrize(&self) -> M;
}
