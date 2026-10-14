//! Submodule implementing Edges-related traits for [`SymmetricCSR2D`].

use num_traits::Zero;

use crate::{
    impls::SymmetricCSR2D,
    traits::{
        Edges, FromDirectedMonopartiteEdges, Graph, Matrix, Matrix2D, MonopartiteEdges,
        MonoplexGraph, SizedSparseMatrix, SizedSparseMatrix2D, SparseMatrix, Symmetrize,
        TryFromUsize,
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

impl<M> Graph for SymmetricCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    #[inline]
    fn has_nodes(&self) -> bool {
        self.number_of_rows() > <Self as Matrix2D>::RowIndex::zero()
            && self.number_of_columns() > <Self as Matrix2D>::ColumnIndex::zero()
    }

    #[inline]
    fn has_edges(&self) -> bool {
        self.number_of_defined_values() > <Self as SparseMatrix>::SparseIndex::zero()
    }
}

impl<M> MonoplexGraph for SymmetricCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    type Edge = <Self as Matrix>::Coordinates;
    type Edges = Self;

    #[inline]
    fn edges(&self) -> &Self::Edges {
        self
    }
}
