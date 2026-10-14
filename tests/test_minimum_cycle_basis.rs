//! Tests for the graph-level minimum-cycle-basis trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

fn build_undigraph(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    let mut normalized_edges = edges
        .iter()
        .copied()
        .map(|[left, right]| if left <= right { (left, right) } else { (right, left) })
        .collect::<Vec<_>>();
    normalized_edges.sort_unstable();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

#[test]
fn test_minimum_cycle_basis_tree_is_empty() {
    let graph = build_undigraph(5, &[[0, 1], [1, 2], [2, 3], [3, 4]]);
    let basis = graph.minimum_cycle_basis().unwrap();

    assert_eq!(basis.cycle_rank(), 0);
    assert_eq!(basis.total_weight(), 0);
    assert!(basis.is_empty());
}

#[test]
fn test_minimum_cycle_basis_triangle() {
    let graph = build_undigraph(3, &[[0, 1], [1, 2], [0, 2]]);
    let basis = graph.minimum_cycle_basis().unwrap();

    assert_eq!(basis.cycle_rank(), 1);
    assert_eq!(basis.total_weight(), 3);
    assert_eq!(basis.minimum_cycle_basis().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
}

#[test]
fn test_minimum_cycle_basis_square_with_diagonal() {
    let graph = build_undigraph(4, &[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]);
    let basis = graph.minimum_cycle_basis().unwrap();

    assert_eq!(basis.cycle_rank(), 2);
    assert_eq!(basis.total_weight(), 6);
    assert_eq!(
        basis.minimum_cycle_basis().cloned().collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![0, 2, 3]]
    );
}

#[test]
fn test_minimum_cycle_basis_across_articulation() {
    let graph = build_undigraph(5, &[[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [2, 4]]);
    let basis = graph.minimum_cycle_basis().unwrap();

    assert_eq!(basis.cycle_rank(), 2);
    assert_eq!(basis.total_weight(), 6);
    assert_eq!(
        basis.minimum_cycle_basis().cloned().collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![2, 3, 4]]
    );
}

#[test]
fn test_minimum_cycle_basis_cubane_weight_only() {
    let graph = build_undigraph(
        8,
        &[
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
    );
    let basis = graph.minimum_cycle_basis().unwrap();

    assert_eq!(basis.cycle_rank(), 5);
    assert_eq!(basis.len(), 5);
    assert_eq!(basis.total_weight(), 20);
}

#[test]
fn test_minimum_cycle_basis_blanket_impl_on_reference() {
    let graph = build_undigraph(3, &[[0, 1], [1, 2], [0, 2]]);
    let basis = <UndiGraph<usize> as MinimumCycleBasis>::minimum_cycle_basis(&graph).unwrap();

    assert_eq!(basis.minimum_cycle_basis().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
}

#[test]
fn test_minimum_cycle_basis_rejects_self_loops() {
    let graph = build_undigraph(3, &[[0, 0], [0, 1], [1, 2]]);

    assert_eq!(graph.minimum_cycle_basis(), Err(MinimumCycleBasisError::SelfLoopsUnsupported));
}
