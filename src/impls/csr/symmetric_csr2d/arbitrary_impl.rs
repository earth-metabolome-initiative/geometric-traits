//! Implementation of the `Arbitrary` trait for the `SymmetricCSR2D` struct.

use arbitrary::{Arbitrary, Unstructured};
use num_traits::AsPrimitive;

use crate::{
    impls::{MutabilityError, SquareCSR2D, SymmetricCSR2D},
    prelude::{Matrix2D, MatrixMut, SparseMatrix2D},
    traits::{PositiveInteger, SparseMatrixMut, TryFromUsize},
};

impl<'a, M> Arbitrary<'a> for SymmetricCSR2D<M>
where
    M: Arbitrary<'a>
        + MatrixMut<
            Entry = (<M as Matrix2D>::RowIndex, <M as Matrix2D>::RowIndex),
            Error = MutabilityError<M>,
        > + SparseMatrixMut<
            MinimalShape = (<M as Matrix2D>::RowIndex, <M as Matrix2D>::RowIndex),
            Entry = (<M as Matrix2D>::RowIndex, <M as Matrix2D>::RowIndex),
            Error = MutabilityError<M>,
        > + SparseMatrix2D<ColumnIndex = <M as Matrix2D>::RowIndex>,
    M::RowIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let source: SquareCSR2D<M> = SquareCSR2D::arbitrary(u)?;
        let n: usize = source.number_of_rows().as_();

        // Collect canonical edges: for each (r, c) with r != c, keep (min, max).
        // Also keep diagonal entries (r, r).
        let mut edges: alloc::vec::Vec<(usize, usize)> = alloc::vec::Vec::new();
        for (r, c) in crate::traits::SparseMatrix::sparse_coordinates(&source) {
            let ri: usize = r.as_();
            let ci: usize = c.as_();
            if ri == ci {
                edges.push((ri, ci));
            } else {
                edges.push((ri.min(ci), ri.max(ci)));
            }
        }
        edges.sort_unstable();
        edges.dedup();

        // Build symmetric edge set.
        let mut sym_edges: alloc::vec::Vec<(usize, usize)> =
            alloc::vec::Vec::with_capacity(edges.len() * 2);
        for (r, c) in edges {
            sym_edges.push((r, c));
            if r != c {
                sym_edges.push((c, r));
            }
        }
        sym_edges.sort_unstable();

        // Build a new SquareCSR2D from the symmetric edges.
        let order =
            M::RowIndex::try_from_usize(n).map_err(|_| arbitrary::Error::IncorrectFormat)?;
        let mut inner: SquareCSR2D<M> = SparseMatrixMut::with_sparse_shape(order);
        for (r, c) in sym_edges {
            let ri =
                M::RowIndex::try_from_usize(r).map_err(|_| arbitrary::Error::IncorrectFormat)?;
            let ci =
                M::RowIndex::try_from_usize(c).map_err(|_| arbitrary::Error::IncorrectFormat)?;
            inner.add((ri, ci)).map_err(|_| arbitrary::Error::IncorrectFormat)?;
        }

        Ok(SymmetricCSR2D { matrix: inner })
    }
}
