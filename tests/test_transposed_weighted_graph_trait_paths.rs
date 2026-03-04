//! Coverage tests for transposed weighted graph default trait methods.

use geometric_traits::{
    impls::ValuedCSR2D,
    naive_structs::GenericEdgesBuilder,
    traits::{
        BiMatrix2D, Edges, EdgesBuilder, Graph, Matrix, Matrix2D, MonoplexGraph,
        RankSelectSparseMatrix, SizedRowsSparseMatrix2D, SizedSparseMatrix, SizedSparseMatrix2D,
        SparseMatrix, SparseMatrix2D, SparseValuedMatrix, SparseValuedMatrix2D,
        TransposedWeightedEdges, TransposedWeightedMonoplexGraph, ValuedBiMatrix2D, ValuedMatrix,
        ValuedMatrix2D, ValuedSizedSparseBiMatrix2D,
    },
};

type TestValue = f64;
type TestValuedCSR = ValuedCSR2D<usize, usize, usize, TestValue>;

fn build_valued_csr(
    mut edges: Vec<(usize, usize, TestValue)>,
    shape: (usize, usize),
) -> TestValuedCSR {
    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    GenericEdgesBuilder::<_, TestValuedCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(shape)
        .edges(edges.into_iter())
        .build()
        .expect("build valued csr")
}

struct TestWeightedBiMatrix {
    matrix: TestValuedCSR,
    transposed: TestValuedCSR,
}

impl TestWeightedBiMatrix {
    fn new(edges: Vec<(usize, usize, TestValue)>, shape: (usize, usize)) -> Self {
        let matrix = build_valued_csr(edges.clone(), shape);
        let transposed_edges: Vec<(usize, usize, TestValue)> =
            edges.into_iter().map(|(r, c, w)| (c, r, w)).collect();
        let transposed = build_valued_csr(transposed_edges, (shape.1, shape.0));
        Self { matrix, transposed }
    }
}

impl Matrix for TestWeightedBiMatrix {
    type Coordinates = (usize, usize);

    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows(), self.number_of_columns()]
    }
}

impl Matrix2D for TestWeightedBiMatrix {
    type RowIndex = usize;
    type ColumnIndex = usize;

    fn number_of_rows(&self) -> Self::RowIndex {
        self.matrix.number_of_rows()
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.matrix.number_of_columns()
    }
}

impl SparseMatrix for TestWeightedBiMatrix {
    type SparseIndex = usize;
    type SparseCoordinates<'a>
        = <TestValuedCSR as SparseMatrix>::SparseCoordinates<'a>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        SparseMatrix::sparse_coordinates(&self.matrix)
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.matrix.last_sparse_coordinates()
    }

    fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
}

impl SizedSparseMatrix for TestWeightedBiMatrix {
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        self.matrix.number_of_defined_values()
    }
}

impl RankSelectSparseMatrix for TestWeightedBiMatrix {
    fn rank(&self, coordinates: &Self::Coordinates) -> Self::SparseIndex {
        self.matrix.rank(coordinates)
    }

    fn select(&self, sparse_index: Self::SparseIndex) -> Self::Coordinates {
        self.matrix.select(sparse_index)
    }
}

impl SparseMatrix2D for TestWeightedBiMatrix {
    type SparseRow<'a>
        = <TestValuedCSR as SparseMatrix2D>::SparseRow<'a>
    where
        Self: 'a;
    type SparseColumns<'a>
        = <TestValuedCSR as SparseMatrix2D>::SparseColumns<'a>
    where
        Self: 'a;
    type SparseRows<'a>
        = <TestValuedCSR as SparseMatrix2D>::SparseRows<'a>
    where
        Self: 'a;

    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        self.matrix.sparse_row(row)
    }

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

impl SizedRowsSparseMatrix2D for TestWeightedBiMatrix {
    type SparseRowSizes<'a>
        = <TestValuedCSR as SizedRowsSparseMatrix2D>::SparseRowSizes<'a>
    where
        Self: 'a;

    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        self.matrix.sparse_row_sizes()
    }

    fn number_of_defined_values_in_row(&self, row: Self::RowIndex) -> Self::ColumnIndex {
        self.matrix.number_of_defined_values_in_row(row)
    }
}

impl SizedSparseMatrix2D for TestWeightedBiMatrix {
    fn rank_row(&self, row: Self::RowIndex) -> Self::SparseIndex {
        self.matrix.rank_row(row)
    }

    fn select_row(&self, sparse_index: Self::SparseIndex) -> Self::RowIndex {
        self.matrix.select_row(sparse_index)
    }

    fn select_column(&self, sparse_index: Self::SparseIndex) -> Self::ColumnIndex {
        self.matrix.select_column(sparse_index)
    }
}

impl BiMatrix2D for TestWeightedBiMatrix {
    type Matrix = TestValuedCSR;
    type TransposedMatrix = TestValuedCSR;

    fn matrix(&self) -> &Self::Matrix {
        &self.matrix
    }

    fn transposed(&self) -> &Self::TransposedMatrix {
        &self.transposed
    }
}

impl ValuedMatrix for TestWeightedBiMatrix {
    type Value = TestValue;
}

impl ValuedMatrix2D for TestWeightedBiMatrix {}

impl SparseValuedMatrix for TestWeightedBiMatrix {
    type SparseValues<'a>
        = <TestValuedCSR as SparseValuedMatrix>::SparseValues<'a>
    where
        Self: 'a;

    fn sparse_values(&self) -> Self::SparseValues<'_> {
        self.matrix.sparse_values()
    }
}

impl SparseValuedMatrix2D for TestWeightedBiMatrix {
    type SparseRowValues<'a>
        = <TestValuedCSR as SparseValuedMatrix2D>::SparseRowValues<'a>
    where
        Self: 'a;

    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        self.matrix.sparse_row_values(row)
    }
}

impl ValuedBiMatrix2D for TestWeightedBiMatrix {
    type ValuedMatrix = TestValuedCSR;
    type ValuedTransposedMatrix = TestValuedCSR;
}

impl Edges for TestWeightedBiMatrix {
    type Edge = (usize, usize, TestValue);
    type SourceNodeId = usize;
    type DestinationNodeId = usize;
    type EdgeId = usize;
    type Matrix = Self;

    fn matrix(&self) -> &Self::Matrix {
        self
    }
}

struct TestWeightedGraph {
    edges: TestWeightedBiMatrix,
}

impl Graph for TestWeightedGraph {
    fn has_nodes(&self) -> bool {
        self.edges.number_of_rows() > 0 && self.edges.number_of_columns() > 0
    }

    fn has_edges(&self) -> bool {
        self.edges.number_of_defined_values() > 0
    }
}

impl MonoplexGraph for TestWeightedGraph {
    type Edge = (usize, usize, TestValue);
    type Edges = TestWeightedBiMatrix;

    fn edges(&self) -> &Self::Edges {
        &self.edges
    }
}

#[test]
fn test_transposed_weighted_edges_and_column_value_helpers() {
    let matrix =
        TestWeightedBiMatrix::new(vec![(0, 2, 5.0), (1, 2, 7.0), (1, 3, 2.0), (3, 2, 1.0)], (4, 4));

    let predecessor_weights: Vec<f64> =
        TransposedWeightedEdges::predecessor_weights(&matrix, 2).collect();
    assert_eq!(predecessor_weights, vec![5.0, 7.0, 1.0]);
    assert_eq!(TransposedWeightedEdges::max_predecessor_weight(&matrix, 2), Some(7.0));
    assert_eq!(TransposedWeightedEdges::min_predecessor_weight(&matrix, 2), Some(1.0));

    let sparse_column_values: Vec<f64> =
        ValuedSizedSparseBiMatrix2D::sparse_column_values(&matrix, 2).collect();
    assert_eq!(sparse_column_values, vec![5.0, 7.0, 1.0]);
    assert_eq!(ValuedSizedSparseBiMatrix2D::sparse_column_max_value(&matrix, 2), Some(7.0));
    assert_eq!(ValuedSizedSparseBiMatrix2D::sparse_column_min_value(&matrix, 2), Some(1.0));
}

#[test]
fn test_transposed_weighted_monoplex_graph_helpers() {
    let graph = TestWeightedGraph {
        edges: TestWeightedBiMatrix::new(
            vec![(0, 2, 5.0), (1, 2, 7.0), (1, 3, 2.0), (3, 2, 1.0)],
            (4, 4),
        ),
    };

    let predecessor_weights: Vec<f64> =
        TransposedWeightedMonoplexGraph::predecessor_weights(&graph, 2).collect();
    assert_eq!(predecessor_weights, vec![5.0, 7.0, 1.0]);
    assert_eq!(TransposedWeightedMonoplexGraph::max_predecessor_weight(&graph, 2), Some(7.0));
    assert_eq!(TransposedWeightedMonoplexGraph::min_predecessor_weight(&graph, 2), Some(1.0));
}
