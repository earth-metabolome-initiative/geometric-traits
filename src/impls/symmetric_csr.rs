//! Submodule implementing Edges-related traits for [`SymmetricCSR2D`].

use crate::{
    impls::SymmetricCSR2D,
    traits::{
        Edges, FromDirectedMonopartiteEdges, Matrix, Matrix2D, MonopartiteEdges,
        SizedSparseMatrix2D, SparseMatrix, Symmetrize, TryFromUsize,
    },
};

impl<M> Edges for SymmetricCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    type Edge = <Self as Matrix>::Coordinates;
    type SourceNodeId = <Self as Matrix2D>::RowIndex;
    type DestinationNodeId = <Self as Matrix2D>::RowIndex;
    type EdgeId = <Self as SparseMatrix>::SparseIndex;
    type Matrix = Self;

    #[inline]
    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

impl<M, DE: MonopartiteEdges> FromDirectedMonopartiteEdges<DE> for SymmetricCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
    DE::MonopartiteMatrix: Symmetrize<Self>,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    #[inline]
    fn from_directed_edges(edges: DE) -> Self {
        edges.matrix().symmetrize()
    }
}
