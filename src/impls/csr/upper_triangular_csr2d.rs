//! Submodule providing a definition of a CSR matrix.
use alloc::vec::Vec;

use multi_ranged::Step;
use num_traits::{AsPrimitive, Zero};

use crate::{
    impls::{MutabilityError, SquareCSR2D},
    prelude::*,
    traits::{PositiveInteger, TryFromUsize},
};

#[derive(Clone, Debug)]
/// A compressed sparse row matrix.
pub struct UpperTriangularCSR2D<M: Matrix2D> {
    /// The underlying matrix.
    matrix: SquareCSR2D<M>,
}

impl<M> Matrix for UpperTriangularCSR2D<M>
where
    M: Matrix2D,
{
    type Coordinates = (M::RowIndex, M::ColumnIndex);

    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows().as_(), self.number_of_columns().as_()]
    }
}

impl<M> Matrix2D for UpperTriangularCSR2D<M>
where
    M: Matrix2D,
{
    type RowIndex = M::RowIndex;
    type ColumnIndex = M::ColumnIndex;

    fn number_of_rows(&self) -> Self::RowIndex {
        self.matrix.number_of_rows()
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.matrix.number_of_columns()
    }
}

impl<M> SquareMatrix for UpperTriangularCSR2D<M>
where
    M: Matrix2D<ColumnIndex = <M as Matrix2D>::RowIndex>,
{
    type Index = M::RowIndex;

    fn order(&self) -> Self::Index {
        self.matrix.order()
    }
}

impl<M> SparseSquareMatrix for UpperTriangularCSR2D<M>
where
    M: SparseMatrix2D<ColumnIndex = <M as Matrix2D>::RowIndex>,
{
    fn number_of_defined_diagonal_values(&self) -> Self::Index {
        self.matrix.number_of_defined_diagonal_values()
    }
}

impl<M> AsRef<M> for UpperTriangularCSR2D<M>
where
    M: Matrix2D,
{
    fn as_ref(&self) -> &M {
        self.matrix.as_ref()
    }
}

impl<M> Default for UpperTriangularCSR2D<M>
where
    M: Matrix2D + Default,
{
    fn default() -> Self {
        Self { matrix: SquareCSR2D::default() }
    }
}

impl<M> SparseMatrixMut for UpperTriangularCSR2D<M>
where
    M: SparseMatrixMut<
            MinimalShape = Self::Coordinates,
            Entry = Self::Coordinates,
            Error = MutabilityError<M>,
        > + SparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type MinimalShape = M::RowIndex;

    fn with_sparse_capacity(number_of_values: Self::SparseIndex) -> Self {
        Self { matrix: SquareCSR2D::with_sparse_capacity(number_of_values) }
    }

    fn with_sparse_shape(shape: Self::MinimalShape) -> Self {
        Self::with_sparse_shaped_capacity(shape, M::SparseIndex::zero())
    }

    fn with_sparse_shaped_capacity(
        shape: Self::MinimalShape,
        number_of_values: Self::SparseIndex,
    ) -> Self {
        Self { matrix: SquareCSR2D::with_sparse_shaped_capacity(shape, number_of_values) }
    }
}

impl<M> SparseMatrix for UpperTriangularCSR2D<M>
where
    M: SparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type SparseIndex = <SquareCSR2D<M> as SparseMatrix>::SparseIndex;
    type SparseCoordinates<'a>
        = <SquareCSR2D<M> as SparseMatrix>::SparseCoordinates<'a>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.matrix.sparse_coordinates()
    }

    fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.matrix.last_sparse_coordinates()
    }
}

impl<M> SizedSparseMatrix for UpperTriangularCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.matrix.number_of_defined_values()
    }
}

impl<M> RankSelectSparseMatrix for UpperTriangularCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex> + RankSelectSparseMatrix,
{
    fn rank(&self, coordinates: &Self::Coordinates) -> Self::SparseIndex {
        self.matrix.rank(coordinates)
    }

    fn select(&self, sparse_index: Self::SparseIndex) -> Self::Coordinates {
        self.matrix.select(sparse_index)
    }
}

impl<M> SparseMatrix2D for UpperTriangularCSR2D<M>
where
    M: SparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type SparseRow<'a>
        = <SquareCSR2D<M> as SparseMatrix2D>::SparseRow<'a>
    where
        Self: 'a;
    type SparseColumns<'a>
        = <SquareCSR2D<M> as SparseMatrix2D>::SparseColumns<'a>
    where
        Self: 'a;
    type SparseRows<'a>
        = <SquareCSR2D<M> as SparseMatrix2D>::SparseRows<'a>
    where
        Self: 'a;

    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        self.matrix.sparse_row(row)
    }

    #[inline]
    fn has_entry(&self, row: Self::RowIndex, column: Self::ColumnIndex) -> bool {
        self.matrix.has_entry(row, column)
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.matrix.sparse_columns()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.matrix.sparse_rows()
    }
}

impl<M> EmptyRows for UpperTriangularCSR2D<M>
where
    M: EmptyRows<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type EmptyRowIndices<'a>
        = <SquareCSR2D<M> as EmptyRows>::EmptyRowIndices<'a>
    where
        Self: 'a;
    type NonEmptyRowIndices<'a>
        = <SquareCSR2D<M> as EmptyRows>::NonEmptyRowIndices<'a>
    where
        Self: 'a;

    fn empty_row_indices(&self) -> Self::EmptyRowIndices<'_> {
        self.matrix.empty_row_indices()
    }

    fn non_empty_row_indices(&self) -> Self::NonEmptyRowIndices<'_> {
        self.matrix.non_empty_row_indices()
    }

    fn number_of_empty_rows(&self) -> Self::RowIndex {
        self.matrix.number_of_empty_rows()
    }

    fn number_of_non_empty_rows(&self) -> Self::RowIndex {
        self.matrix.number_of_non_empty_rows()
    }
}

impl<M> SizedRowsSparseMatrix2D for UpperTriangularCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type SparseRowSizes<'a>
        = <SquareCSR2D<M> as SizedRowsSparseMatrix2D>::SparseRowSizes<'a>
    where
        Self: 'a;

    fn number_of_defined_values_in_row(&self, row: Self::RowIndex) -> Self::ColumnIndex {
        self.matrix.number_of_defined_values_in_row(row)
    }

    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.matrix.sparse_row_sizes()
    }
}

impl<M> SizedSparseMatrix2D for UpperTriangularCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    fn rank_row(&self, row: M::RowIndex) -> Self::SparseIndex {
        self.matrix.rank_row(row)
    }

    fn select_column(&self, sparse_index: Self::SparseIndex) -> Self::ColumnIndex {
        self.matrix.select_column(sparse_index)
    }

    fn select_row(&self, sparse_index: Self::SparseIndex) -> Self::RowIndex {
        self.matrix.select_row(sparse_index)
    }
}

impl<M> MatrixMut for UpperTriangularCSR2D<M>
where
    M: MatrixMut<Entry = Self::Coordinates, Error = MutabilityError<M>>
        + Matrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    type Entry = Self::Coordinates;
    type Error = crate::impls::MutabilityError<Self>;

    fn add(&mut self, (row, column): Self::Entry) -> Result<(), Self::Error> {
        if row > column {
            return Err(MutabilityError::OutOfBounds(
                (row, column),
                (self.order(), self.order()),
                "In an upper triangular matrix, row indices must be less than or equal to column indices.",
            ));
        }
        self.matrix.add((row, column))?;

        Ok(())
    }

    fn increase_shape(&mut self, shape: Self::Coordinates) -> Result<(), Self::Error> {
        Ok(self.matrix.increase_shape(shape)?)
    }
}

impl<M> TransposableMatrix2D<SquareCSR2D<M>> for UpperTriangularCSR2D<M>
where
    M: TransposableMatrix2D<M, ColumnIndex = <Self as Matrix2D>::RowIndex>,
{
    fn transpose(&self) -> SquareCSR2D<M> {
        self.matrix.transpose()
    }
}

impl<SparseIndex, Idx> Symmetrize<SymmetricCSR2D<CSR2D<SparseIndex, Idx, Idx>>>
    for UpperTriangularCSR2D<CSR2D<SparseIndex, Idx, Idx>>
where
    Idx: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<SparseIndex>,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    fn symmetrize(&self) -> SymmetricCSR2D<CSR2D<SparseIndex, Idx, Idx>> {
        // We initialize the transposed matrix.
        let number_of_expected_column_indices = (self.number_of_defined_values().as_()
            - self.number_of_defined_diagonal_values().as_())
            * 2
            + self.number_of_defined_diagonal_values().as_();

        let mut symmetric: CSR2D<SparseIndex, Idx, Idx> = CSR2D {
            offsets: vec![SparseIndex::zero(); self.order().as_() + 1],
            number_of_columns: self.order(),
            number_of_rows: self.order(),
            column_indices: vec![Idx::zero(); number_of_expected_column_indices],
            number_of_non_empty_rows: Idx::zero(),
        };

        // First, we proceed to compute the number of elements in each column.
        for (row, column) in crate::traits::SparseMatrix::sparse_coordinates(self) {
            // TODO! IF YOU INITIALIZE OFFSETS WITH THE OUT BOUND DEGREES, THERE IS NO NEED
            // FOR ALL OF THE SPARSE ROW ACCESSES!
            symmetric.offsets[row.as_() + 1] += SparseIndex::one();
            symmetric.offsets[column.as_() + 1] +=
                if row == column { SparseIndex::zero() } else { SparseIndex::one() };
        }

        // Then, we compute the prefix sum of the degrees to get the offsets.
        let mut prefix_sum = SparseIndex::zero();
        for offset in &mut symmetric.offsets {
            prefix_sum += *offset;
            symmetric.number_of_non_empty_rows +=
                if *offset > SparseIndex::zero() { Idx::one() } else { Idx::zero() };
            *offset = prefix_sum;
        }

        // Finally, we fill the column indices.
        let mut degree = vec![SparseIndex::zero(); self.order().as_()];
        for (row, column) in crate::traits::SparseMatrix::sparse_coordinates(self) {
            let edges: Vec<(Idx, Idx)> = if row == column {
                vec![(row, column)]
            } else {
                vec![(row, column), (column, row)]
            };
            for (i, j) in edges {
                let current_degree: &mut SparseIndex = &mut degree[i.as_()];
                let index = *current_degree + symmetric.offsets[i.as_()];
                symmetric.column_indices[index.as_()] = j;
                *current_degree += SparseIndex::one();
            }
        }

        debug_assert_eq!(
            symmetric.number_of_defined_values().as_(),
            number_of_expected_column_indices,
            "The number of inserted values is not the expected one. Original number of values: {}. Diagonals: {}",
            self.number_of_defined_values(),
            self.number_of_defined_diagonal_values()
        );

        SymmetricCSR2D {
            matrix: SquareCSR2D {
                matrix: symmetric,
                number_of_diagonal_values: self.number_of_defined_diagonal_values(),
            },
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::{impls::CSR2D, traits::MatrixMut};

    type TestCSR2D = CSR2D<usize, usize, usize>;
    type TestUpperTriangular = UpperTriangularCSR2D<TestCSR2D>;

    #[test]
    fn test_upper_triangular_default() {
        let ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        assert_eq!(ut.number_of_rows(), 0);
        assert_eq!(ut.number_of_columns(), 0);
        assert!(ut.is_empty());
    }

    #[test]
    fn test_upper_triangular_with_sparse_shape() {
        let ut: TestUpperTriangular = SparseMatrixMut::with_sparse_shape(3);
        assert_eq!(ut.order(), 3);
    }

    #[test]
    fn test_upper_triangular_add_valid_entries() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        // Diagonal entry
        assert!(MatrixMut::add(&mut ut, (0, 0)).is_ok());
        // Upper triangular entry (row < column)
        assert!(MatrixMut::add(&mut ut, (0, 1)).is_ok());
        assert!(MatrixMut::add(&mut ut, (0, 2)).is_ok());
        assert!(MatrixMut::add(&mut ut, (1, 1)).is_ok());
        assert!(MatrixMut::add(&mut ut, (1, 2)).is_ok());
        assert_eq!(ut.number_of_defined_values(), 5);
    }

    #[test]
    fn test_upper_triangular_add_lower_triangular_error() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        // Lower triangular entry (row > column) should fail
        assert!(MatrixMut::add(&mut ut, (1, 0)).is_err());
        assert!(MatrixMut::add(&mut ut, (2, 0)).is_err());
        assert!(MatrixMut::add(&mut ut, (2, 1)).is_err());
    }

    #[test]
    fn test_upper_triangular_diagonal_values() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        MatrixMut::add(&mut ut, (0, 0)).unwrap();
        MatrixMut::add(&mut ut, (0, 1)).unwrap();
        MatrixMut::add(&mut ut, (1, 1)).unwrap();
        assert_eq!(ut.number_of_defined_diagonal_values(), 2);
    }

    #[test]
    fn test_upper_triangular_sparse_row() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        MatrixMut::add(&mut ut, (0, 0)).unwrap();
        MatrixMut::add(&mut ut, (0, 1)).unwrap();
        MatrixMut::add(&mut ut, (0, 2)).unwrap();

        let row0: Vec<usize> = ut.sparse_row(0).collect();
        assert_eq!(row0, vec![0, 1, 2]);
    }

    #[test]
    fn test_upper_triangular_has_entry() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        MatrixMut::add(&mut ut, (0, 1)).unwrap();
        MatrixMut::add(&mut ut, (1, 2)).unwrap();

        assert!(!ut.has_entry(0, 0));
        assert!(ut.has_entry(0, 1));
        assert!(!ut.has_entry(1, 0));
        assert!(ut.has_entry(1, 2));
    }

    #[test]
    fn test_upper_triangular_shape() {
        let ut: TestUpperTriangular = SparseMatrixMut::with_sparse_shape(4);
        assert_eq!(ut.shape(), vec![4, 4]);
    }

    #[test]
    fn test_upper_triangular_symmetrize() {
        let mut ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        MatrixMut::add(&mut ut, (0, 0)).unwrap();
        MatrixMut::add(&mut ut, (0, 1)).unwrap();
        MatrixMut::add(&mut ut, (1, 1)).unwrap();

        let sym = ut.symmetrize();
        // After symmetrization, (1, 0) should exist
        assert!(sym.has_entry(0, 0));
        assert!(sym.has_entry(0, 1));
        assert!(sym.has_entry(1, 0)); // Added by symmetrization
        assert!(sym.has_entry(1, 1));
    }

    #[test]
    fn test_upper_triangular_debug() {
        let ut: TestUpperTriangular = UpperTriangularCSR2D::default();
        let debug = alloc::format!("{ut:?}");
        assert!(debug.contains("UpperTriangularCSR2D"));
    }
}
