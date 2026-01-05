//! Submodule implementing Edges for CSR2D.

use crate::impls::csr::CSR2D;
use crate::traits::{
    BidirectionalVocabulary, BipartiteGraph, Edges, Graph, GrowableEdges, Matrix, Matrix2D,
    Matrix2DRef, MonoplexGraph, SizedSparseMatrix, SparseMatrixMut,
};
use crate::traits::{IntoUsize, PositiveInteger, TryFromUsize};
use multi_ranged::Step;

use crate::errors::builder::edges::EdgesBuilderError;

impl<
    SparseIndex: PositiveInteger + IntoUsize + TryFromUsize,
    RowIndex: Step + PositiveInteger + TryFromUsize + IntoUsize,
    ColumnIndex: Step + PositiveInteger + IntoUsize + TryFromUsize + TryFrom<SparseIndex>,
> Edges for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type Edge = <Self as Matrix>::Coordinates;
    type SourceNodeId = RowIndex;
    type DestinationNodeId = ColumnIndex;
    type EdgeId = SparseIndex;
    type Matrix = Self;

    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

impl<
    SparseIndex: PositiveInteger + 'static,
    RowIndex: Step + PositiveInteger,
    ColumnIndex: Step + PositiveInteger + TryFrom<SparseIndex>,
> GrowableEdges for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type GrowableMatrix = Self;
    type Error = EdgesBuilderError<Self>;

    fn matrix_mut(&mut self) -> &mut Self::GrowableMatrix {
        self
    }

    fn with_capacity(number_of_edges: Self::EdgeId) -> Self {
        <Self as SparseMatrixMut>::with_sparse_capacity(number_of_edges)
    }

    fn with_shape(shape: <Self::GrowableMatrix as SparseMatrixMut>::MinimalShape) -> Self {
        <Self as SparseMatrixMut>::with_sparse_shape(shape)
    }

    fn with_shaped_capacity(
        shape: <Self::GrowableMatrix as SparseMatrixMut>::MinimalShape,
        number_of_edges: Self::EdgeId,
    ) -> Self {
        <Self as SparseMatrixMut>::with_sparse_shaped_capacity(shape, number_of_edges)
    }
}

impl<
    SparseIndex: PositiveInteger + IntoUsize + TryFromUsize,
    RowIndex: Step + PositiveInteger + TryFromUsize + IntoUsize,
    ColumnIndex: Step + PositiveInteger + IntoUsize + TryFrom<SparseIndex> + TryFromUsize,
> Graph for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    fn has_nodes(&self) -> bool {
        self.number_of_rows() > RowIndex::zero() && self.number_of_columns() > ColumnIndex::zero()
    }

    fn has_edges(&self) -> bool {
        self.number_of_defined_values() > SparseIndex::zero()
    }
}

impl<
    SparseIndex: PositiveInteger + IntoUsize + TryFromUsize,
    RowIndex: Step + PositiveInteger + TryFromUsize + IntoUsize,
    ColumnIndex: Step + PositiveInteger + IntoUsize + TryFrom<SparseIndex> + TryFromUsize,
> MonoplexGraph for CSR2D<SparseIndex, RowIndex, ColumnIndex>
{
    type Edge = <Self as Matrix>::Coordinates;
    type Edges = Self;

    fn edges(&self) -> &Self::Edges {
        self
    }
}

impl<
    SparseIndex: PositiveInteger + IntoUsize + TryFromUsize,
    RowIndex: Step
        + PositiveInteger
        + TryFromUsize
        + IntoUsize
        + BidirectionalVocabulary<SourceSymbol = RowIndex, DestinationSymbol = RowIndex>,
    ColumnIndex: Step
        + PositiveInteger
        + IntoUsize
        + TryFrom<SparseIndex>
        + TryFromUsize
        + BidirectionalVocabulary<SourceSymbol = ColumnIndex, DestinationSymbol = ColumnIndex>,
> BipartiteGraph for CSR2D<SparseIndex, RowIndex, ColumnIndex>
where
    Self: Matrix2DRef<RowIndex = RowIndex, ColumnIndex = ColumnIndex>,
{
    type LeftNodeId = RowIndex;
    type RightNodeId = ColumnIndex;
    type LeftNodeSymbol = RowIndex;
    type RightNodeSymbol = ColumnIndex;
    type LeftNodes = RowIndex;
    type RightNodes = ColumnIndex;

    fn left_nodes_vocabulary(&self) -> &Self::LeftNodes {
        self.number_of_rows_ref()
    }

    fn right_nodes_vocabulary(&self) -> &Self::RightNodes {
        self.number_of_columns_ref()
    }
}
