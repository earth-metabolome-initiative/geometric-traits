//! Tests for GenericImplicitValuedMatrix2D: all trait delegations and
//! implicit value functionality.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, GenericImplicitValuedMatrix2D},
    traits::{
        BipartiteGraph, Edges, EmptyRows, Graph, ImplicitValuedMatrix, Matrix2D, Matrix2DRef,
        MatrixMut, MonoplexGraph, RankSelectSparseMatrix, SizedRowsSparseMatrix2D,
        SizedSparseMatrix, SizedSparseMatrix2D, SizedSparseValuedMatrix, SparseMatrix,
        SparseMatrix2D, SparseMatrixMut, SparseValuedMatrix, SparseValuedMatrix2D,
    },
};

type TestCSR = CSR2D<usize, usize, usize>;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("test indices should fit in u32"))
}

fn build_implicit(
    entries: Vec<(usize, usize)>,
    shape: (usize, usize),
) -> GenericImplicitValuedMatrix2D<TestCSR, impl Fn((usize, usize)) -> f64, f64> {
    let mut inner: TestCSR = SparseMatrixMut::with_sparse_shaped_capacity(shape, entries.len());
    for (r, c) in entries {
        MatrixMut::add(&mut inner, (r, c)).unwrap();
    }
    GenericImplicitValuedMatrix2D::new(inner, |(r, c): (usize, usize)| usize_to_f64(r * 10 + c))
}

// ============================================================================
// Matrix / Matrix2D / Matrix2DRef
// ============================================================================

#[test]
fn test_implicit_number_of_rows() {
    let m = build_implicit(vec![(0, 0), (1, 1)], (3, 4));
    assert_eq!(m.number_of_rows(), 3);
    assert_eq!(m.number_of_columns(), 4);
}

#[test]
fn test_implicit_matrix2d_ref() {
    let m = build_implicit(vec![(0, 0)], (3, 4));
    assert_eq!(*m.number_of_rows_ref(), 3);
    assert_eq!(*m.number_of_columns_ref(), 4);
}

// ============================================================================
// SparseMatrix
// ============================================================================

#[test]
fn test_implicit_sparse_coordinates() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).collect();
    assert_eq!(coords, vec![(0, 0), (0, 1), (1, 2)]);
}

#[test]
fn test_implicit_last_sparse_coordinates() {
    let m = build_implicit(vec![(0, 0), (1, 2)], (2, 3));
    assert_eq!(m.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_implicit_is_empty() {
    let empty = build_implicit(vec![], (0, 0));
    assert!(empty.is_empty());
    let non_empty = build_implicit(vec![(0, 0)], (1, 1));
    assert!(!non_empty.is_empty());
}

// ============================================================================
// SizedSparseMatrix
// ============================================================================

#[test]
fn test_implicit_number_of_defined_values() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 0)], (2, 2));
    assert_eq!(m.number_of_defined_values(), 3);
}

// ============================================================================
// RankSelectSparseMatrix
// ============================================================================

#[test]
fn test_implicit_rank_select() {
    // Use one entry per row so select aligns with row boundaries
    let m = build_implicit(vec![(0, 0), (1, 1)], (2, 2));
    assert_eq!(m.rank(&(0, 0)), 0);
    assert_eq!(m.rank(&(1, 1)), 1);
    assert_eq!(m.select(0), (0, 0));
    assert_eq!(m.select(1), (1, 1));
}

// ============================================================================
// SparseMatrix2D
// ============================================================================

#[test]
fn test_implicit_sparse_row() {
    let m = build_implicit(vec![(0, 0), (0, 2), (1, 1)], (2, 3));
    let row0: Vec<usize> = m.sparse_row(0).collect();
    assert_eq!(row0, vec![0, 2]);
    let row1: Vec<usize> = m.sparse_row(1).collect();
    assert_eq!(row1, vec![1]);
}

#[test]
fn test_implicit_has_entry() {
    let m = build_implicit(vec![(0, 1), (1, 0)], (2, 2));
    assert!(!m.has_entry(0, 0));
    assert!(m.has_entry(0, 1));
    assert!(m.has_entry(1, 0));
    assert!(!m.has_entry(1, 1));
}

#[test]
fn test_implicit_sparse_columns() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 1)], (2, 2));
    let cols: Vec<usize> = m.sparse_columns().collect();
    assert_eq!(cols, vec![0, 1, 1]);
}

#[test]
fn test_implicit_sparse_rows() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 1)], (2, 2));
    let rows: Vec<usize> = m.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 1]);
}

// ============================================================================
// EmptyRows
// ============================================================================

#[test]
fn test_implicit_empty_rows() {
    let m = build_implicit(vec![(0, 0)], (3, 3));
    assert_eq!(m.number_of_empty_rows(), 2);
    assert_eq!(m.number_of_non_empty_rows(), 1);
    let empty: Vec<usize> = m.empty_row_indices().collect();
    assert_eq!(empty, vec![1, 2]);
    let non_empty: Vec<usize> = m.non_empty_row_indices().collect();
    assert_eq!(non_empty, vec![0]);
}

// ============================================================================
// SizedRowsSparseMatrix2D
// ============================================================================

#[test]
fn test_implicit_sized_rows() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 0)], (2, 2));
    assert_eq!(m.number_of_defined_values_in_row(0), 2);
    assert_eq!(m.number_of_defined_values_in_row(1), 1);
    let sizes: Vec<usize> = m.sparse_row_sizes().collect();
    assert_eq!(sizes, vec![2, 1]);
}

// ============================================================================
// SizedSparseMatrix2D
// ============================================================================

#[test]
fn test_implicit_sized_sparse_matrix2d() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 0)], (2, 2));
    assert_eq!(m.rank_row(0), 0);
    assert_eq!(m.rank_row(1), 2);
    assert_eq!(m.select_row(0), 0);
    assert_eq!(m.select_row(2), 1);
    assert_eq!(m.select_column(0), 0);
    assert_eq!(m.select_column(1), 1);
    assert_eq!(m.select_column(2), 0);
}

// ============================================================================
// ImplicitValuedMatrix
// ============================================================================

#[test]
fn test_implicit_value() {
    let m = build_implicit(vec![(0, 0), (1, 2)], (2, 3));
    assert!((m.implicit_value(&(0, 0)) - 0.0).abs() < f64::EPSILON); // 0*10+0
    assert!((m.implicit_value(&(1, 2)) - 12.0).abs() < f64::EPSILON); // 1*10+2
    assert!((m.implicit_value(&(2, 3)) - 23.0).abs() < f64::EPSILON); // 2*10+3
}

// ============================================================================
// SparseValuedMatrix
// ============================================================================

#[test]
fn test_implicit_sparse_values() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let values: Vec<f64> = m.sparse_values().collect();
    assert_eq!(values, vec![0.0, 1.0, 12.0]); // (0,0)=0, (0,1)=1, (1,2)=12
}

// ============================================================================
// SizedSparseValuedMatrix
// ============================================================================

#[test]
fn test_implicit_select_value() {
    // Use one entry per row so select aligns with row boundaries
    let m = build_implicit(vec![(0, 0), (1, 2)], (2, 3));
    assert!((m.select_value(0) - 0.0).abs() < f64::EPSILON); // (0,0) -> 0*10+0
    assert!((m.select_value(1) - 12.0).abs() < f64::EPSILON); // (1,2) -> 1*10+2
}

// ============================================================================
// SparseValuedMatrix2D
// ============================================================================

#[test]
fn test_implicit_sparse_row_values() {
    let m = build_implicit(vec![(0, 0), (0, 1), (1, 2)], (2, 3));
    let row0_values: Vec<f64> = m.sparse_row_values(0).collect();
    assert_eq!(row0_values, vec![0.0, 1.0]);
    let row1_values: Vec<f64> = m.sparse_row_values(1).collect();
    assert_eq!(row1_values, vec![12.0]);
}

// ============================================================================
// Graph traits: Edges, Graph, MonoplexGraph, BipartiteGraph
// ============================================================================

#[test]
fn test_implicit_edges_matrix() {
    let m = build_implicit(vec![(0, 0), (1, 1)], (2, 2));
    let _matrix = Edges::matrix(&m);
}

#[test]
fn test_implicit_graph_has_nodes() {
    let m = build_implicit(vec![(0, 0)], (2, 3));
    assert!(Graph::has_nodes(&m));
    assert!(Graph::has_edges(&m));
}

#[test]
fn test_implicit_graph_no_edges() {
    let m = build_implicit(vec![], (2, 3));
    assert!(Graph::has_nodes(&m));
    assert!(!Graph::has_edges(&m));
}

#[test]
fn test_implicit_graph_no_nodes() {
    let m = build_implicit(vec![], (0, 0));
    assert!(!Graph::has_nodes(&m));
}

#[test]
fn test_implicit_monoplex_graph() {
    let m = build_implicit(vec![(0, 0)], (2, 2));
    let _e = MonoplexGraph::edges(&m);
}

#[test]
fn test_implicit_bipartite_graph() {
    let m = build_implicit(vec![(0, 0)], (2, 3));
    let _left = BipartiteGraph::left_nodes_vocabulary(&m);
    let _right = BipartiteGraph::right_nodes_vocabulary(&m);
}

// ============================================================================
// Clone / Debug
// ============================================================================

// Debug requires Map: Debug, which closures don't implement.
// Skip debug test.
