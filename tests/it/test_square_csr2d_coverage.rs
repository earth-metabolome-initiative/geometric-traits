//! Tests for SquareCSR2D: AsRef, add with shape expansion, diagonal tracking,
//! SortedVec GrowableVocabulary errors, and additional graph algorithm paths.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, MatrixMut, SparseMatrix2D, SparseMatrixMut, SparseSquareMatrix, SquareMatrix,
        VocabularyBuilder, algorithms::connected_components::ConnectedComponents,
    },
};

// ============================================================================
// SquareCSR2D: AsRef, order, diagonal tracking
// ============================================================================

#[test]
fn test_square_csr2d_as_ref() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (1, 2)).unwrap();
    let inner: &CSR2D<usize, usize, usize> = sq.as_ref();
    assert_eq!(inner.number_of_rows(), 3);
}

#[test]
fn test_square_csr2d_order() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(4);
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    MatrixMut::add(&mut sq, (2, 3)).unwrap();
    assert_eq!(sq.order(), 4);
}

#[test]
fn test_square_csr2d_diagonal_count() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap(); // diagonal
    MatrixMut::add(&mut sq, (0, 1)).unwrap(); // off-diagonal
    MatrixMut::add(&mut sq, (1, 1)).unwrap(); // diagonal
    MatrixMut::add(&mut sq, (2, 0)).unwrap(); // off-diagonal
    assert_eq!(sq.number_of_defined_diagonal_values(), 2);
}

#[test]
fn test_square_csr2d_add_expands_shape() {
    // Adding entry at (2, 3) should expand to at least 4x4
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(1);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    assert_eq!(sq.order(), 1);
    // Now add beyond current shape
    MatrixMut::add(&mut sq, (2, 3)).unwrap();
    assert_eq!(sq.order(), 4); // max(3, 4) = 4
    assert_eq!(sq.number_of_rows(), 4);
    assert_eq!(sq.number_of_columns(), 4);
}

#[test]
fn test_square_csr2d_number_of_defined_values_in_row() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 0)).unwrap();
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    MatrixMut::add(&mut sq, (0, 2)).unwrap();
    MatrixMut::add(&mut sq, (1, 0)).unwrap();
    assert_eq!(sq.number_of_defined_values_in_row(0), 3);
    assert_eq!(sq.number_of_defined_values_in_row(1), 1);
}

// ============================================================================
// SquareCSR2D via DiEdgesBuilder
// ============================================================================

#[test]
fn test_square_csr2d_builder() {
    let sq: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(3)
        .expected_shape(4)
        .edges(vec![(0, 1), (1, 2), (2, 3)].into_iter())
        .build()
        .unwrap();
    assert_eq!(sq.order(), 4);
    assert!(sq.has_entry(0, 1));
    assert!(sq.has_entry(1, 2));
    assert!(sq.has_entry(2, 3));
    assert!(!sq.has_entry(0, 0));
}

// ============================================================================
// SortedVec: GrowableVocabulary::add error paths
// ============================================================================

#[test]
fn test_sorted_vec_is_sorted() {
    let sv: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![10, 20, 30].into_iter().enumerate())
        .build()
        .unwrap();
    assert!(sv.is_sorted());
}

#[test]
fn test_sorted_vec_binary_search_by() {
    let sv: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(4)
        .symbols(vec![10, 20, 30, 40].into_iter().enumerate())
        .build()
        .unwrap();
    assert_eq!(sv.binary_search_by(|v| v.cmp(&20)), Ok(1));
    assert_eq!(sv.binary_search_by(|v| v.cmp(&25)), Err(2));
}

// ============================================================================
// Connected Components: From conversion, TooManyComponents with small marker
// ============================================================================

fn build_undigraph_simple(
    node_list: Vec<usize>,
    edge_list: Vec<(usize, usize)>,
) -> UndiGraph<usize> {
    let num_nodes = node_list.len();
    let num_edges = edge_list.len();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(node_list.into_iter().enumerate())
        .build()
        .unwrap();
    let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(nodes.len())
        .edges(edge_list.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

#[test]
fn test_cc_chain_single_component() {
    let graph = build_undigraph_simple(vec![0, 1, 2, 3], vec![(0, 1), (1, 2), (2, 3)]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 4);
    assert_eq!(cc.smallest_component_size(), 4);
}

#[test]
fn test_cc_three_components() {
    // Three disconnected components: {0,1}, {2}, {3,4,5}
    let graph = build_undigraph_simple(vec![0, 1, 2, 3, 4, 5], vec![(0, 1), (3, 4), (4, 5)]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 3);
    assert_eq!(cc.largest_component_size(), 3);
    assert_eq!(cc.smallest_component_size(), 1);
}

#[test]
fn test_cc_dense_graph() {
    // Triangle: all connected (edges must be sorted for CSR construction)
    let graph = build_undigraph_simple(vec![0, 1, 2], vec![(0, 1), (0, 2), (1, 2)]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 1);
}
