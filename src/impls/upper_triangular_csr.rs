//! Submodule implementing Edges-related traits for
//! [`UpperTriangularCSR2D`].

use core::fmt::Debug;

use crate::{
    errors::builder::edges::EdgesBuilderError,
    impls::{MutabilityError, UpperTriangularCSR2D},
    traits::{
        Edges, GrowableEdges, Matrix, Matrix2D, SizedSparseMatrix2D, SparseMatrix, SparseMatrixMut,
        TryFromUsize,
    },
};

impl<M> Edges for UpperTriangularCSR2D<M>
where
    M: SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    type Edge = <Self as Matrix>::Coordinates;
    type SourceNodeId = <Self as Matrix2D>::RowIndex;
    type DestinationNodeId = <Self as Matrix2D>::ColumnIndex;
    type EdgeId = <Self as SparseMatrix>::SparseIndex;
    type Matrix = Self;

    #[inline]
    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

impl<M> GrowableEdges for UpperTriangularCSR2D<M>
where
    M: Debug
        + SparseMatrixMut<
            MinimalShape = <Self as Matrix>::Coordinates,
            Entry = <Self as Matrix>::Coordinates,
            Error = MutabilityError<M>,
        > + Default
        + SizedSparseMatrix2D<ColumnIndex = <Self as Matrix2D>::RowIndex>
        + 'static,
    M::RowIndex: TryFromUsize,
    M::SparseIndex: TryFromUsize,
{
    type GrowableMatrix = Self;
    type Error = EdgesBuilderError<Self>;

    #[inline]
    fn matrix_mut(&mut self) -> &mut Self::GrowableMatrix {
        self
    }

    #[inline]
    fn with_capacity(number_of_edges: Self::EdgeId) -> Self {
        <Self as SparseMatrixMut>::with_sparse_capacity(number_of_edges)
    }

    #[inline]
    fn with_shape(shape: <Self::GrowableMatrix as SparseMatrixMut>::MinimalShape) -> Self {
        <Self as SparseMatrixMut>::with_sparse_shape(shape)
    }

    #[inline]
    fn with_shaped_capacity(
        shape: <Self::GrowableMatrix as SparseMatrixMut>::MinimalShape,
        number_of_edges: Self::EdgeId,
    ) -> Self {
        <Self as SparseMatrixMut>::with_sparse_shaped_capacity(shape, number_of_edges)
    }
}
