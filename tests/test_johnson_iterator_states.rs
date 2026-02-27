//! Manual stepping tests for Johnson's iterator state transitions.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::*,
    traits::{EdgesBuilder, Johnson},
};

fn build_square_csr(
    node_count: usize,
    mut edges: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .expect("failed to build square CSR matrix")
}

#[test]
fn test_johnson_manual_next_on_dag_is_immediately_none() {
    let matrix = build_square_csr(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let mut iterator = matrix.johnson();

    assert_eq!(iterator.next(), None);
    assert_eq!(iterator.next(), None);
}

#[test]
fn test_johnson_manual_next_across_multiple_sccs_then_exhaustion() {
    // SCC #1: 0 <-> 1, SCC #2: 2 <-> 3, bridge 1 -> 2.
    let matrix = build_square_csr(4, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);
    let mut iterator = matrix.johnson();

    let first = iterator.next().expect("expected first cycle");
    let second = iterator.next().expect("expected second cycle");
    let third = iterator.next();
    let fourth = iterator.next();

    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 2);
    assert!(first.contains(&0) || first.contains(&2));
    assert!(second.contains(&0) || second.contains(&2));
    assert_eq!(third, None);
    assert_eq!(fourth, None);
}

#[test]
fn test_johnson_manual_next_with_isolated_nodes_stops_after_single_cycle() {
    // One 2-cycle and two isolated nodes.
    let matrix = build_square_csr(4, vec![(0, 1), (1, 0)]);
    let mut iterator = matrix.johnson();

    let cycle = iterator.next().expect("expected one cycle");
    assert_eq!(cycle.len(), 2);
    assert!(cycle.contains(&0));
    assert!(cycle.contains(&1));
    assert_eq!(iterator.next(), None);
    assert_eq!(iterator.next(), None);
}
