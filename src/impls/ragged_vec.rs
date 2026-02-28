//! Implementations of traits from the graph crate for the
//! [`RaggedVector`] data structure.

use core::fmt::Debug;

use multi_ranged::Step;

use num_traits::AsPrimitive;

use crate::{
    errors::builder::edges::EdgesBuilderError,
    impls::RaggedVector,
    traits::{
        BidirectionalVocabulary, BipartiteGraph, Edges, Graph, GrowableEdges, Matrix2D,
        Matrix2DRef, MonoplexGraph, PositiveInteger, SizedSparseMatrix, SparseMatrix,
        SparseMatrixMut, TryFromUsize,
    },
};

impl<SparseIndex, RowIndex, ColumnIndex> Edges for RaggedVector<SparseIndex, RowIndex, ColumnIndex>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    type Edge = (<Self as Matrix2D>::RowIndex, <Self as Matrix2D>::ColumnIndex);
    type SourceNodeId = <Self as Matrix2D>::RowIndex;
    type DestinationNodeId = <Self as Matrix2D>::ColumnIndex;
    type EdgeId = SparseIndex;
    type Matrix = Self;

    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

impl<SparseIndex, RowIndex, ColumnIndex> GrowableEdges
    for RaggedVector<SparseIndex, RowIndex, ColumnIndex>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger + 'static,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <ColumnIndex as TryFrom<usize>>::Error: Debug,
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

impl<SparseIndex, RowIndex, ColumnIndex> Graph for RaggedVector<SparseIndex, RowIndex, ColumnIndex>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    fn has_nodes(&self) -> bool {
        self.number_of_rows() > <Self as Matrix2D>::RowIndex::zero()
            && self.number_of_columns() > <Self as Matrix2D>::ColumnIndex::zero()
    }

    fn has_edges(&self) -> bool {
        self.number_of_defined_values() > <Self as SparseMatrix>::SparseIndex::zero()
    }
}

impl<SparseIndex, RowIndex, ColumnIndex> MonoplexGraph
    for RaggedVector<SparseIndex, RowIndex, ColumnIndex>
where
    RowIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    ColumnIndex: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    type Edge = (<Self as Matrix2D>::RowIndex, <Self as Matrix2D>::ColumnIndex);
    type Edges = Self;

    fn edges(&self) -> &Self::Edges {
        self
    }
}

impl<SparseIndex, RowIndex, ColumnIndex> BipartiteGraph
    for RaggedVector<SparseIndex, RowIndex, ColumnIndex>
where
    RowIndex: Step
        + PositiveInteger
        + AsPrimitive<usize>
        + TryFromUsize
        + BidirectionalVocabulary<
            SourceSymbol = <Self as Matrix2D>::RowIndex,
            DestinationSymbol = <Self as Matrix2D>::RowIndex,
        >,
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    ColumnIndex: Step
        + PositiveInteger
        + AsPrimitive<usize>
        + TryFromUsize
        + BidirectionalVocabulary<
            SourceSymbol = <Self as Matrix2D>::ColumnIndex,
            DestinationSymbol = <Self as Matrix2D>::ColumnIndex,
        >,
    <RowIndex as TryFrom<usize>>::Error: Debug,
    <ColumnIndex as TryFrom<usize>>::Error: Debug,
{
    type LeftNodeId = <Self as Matrix2D>::RowIndex;
    type RightNodeId = <Self as Matrix2D>::ColumnIndex;
    type LeftNodeSymbol = <Self as Matrix2D>::RowIndex;
    type RightNodeSymbol = <Self as Matrix2D>::ColumnIndex;
    type LeftNodes = <Self as Matrix2D>::RowIndex;
    type RightNodes = <Self as Matrix2D>::ColumnIndex;

    fn left_nodes_vocabulary(&self) -> &Self::LeftNodes {
        self.number_of_rows_ref()
    }

    fn right_nodes_vocabulary(&self) -> &Self::RightNodes {
        self.number_of_columns_ref()
    }
}
