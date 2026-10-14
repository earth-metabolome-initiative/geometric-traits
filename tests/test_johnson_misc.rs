//! Additional Johnson cycle-enumeration coverage by algorithm domain.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SquareCSR2D},
    prelude::*,
    traits::EdgesBuilder,
};

fn build_sq(n: usize, mut edges: Vec<(usize, usize)>) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_johnson_large_scc_with_bypass() {
    let m = build_sq(4, vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 3);
}

#[test]
fn test_johnson_dense_with_blocking() {
    let m = build_sq(
        4,
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
        ],
    );
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 5);
}

#[test]
fn test_johnson_multiple_components_varied() {
    let m =
        build_sq(8, vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (5, 6), (5, 7), (6, 7), (7, 5)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 4);
}

#[test]
fn test_johnson_chain_of_twocycles() {
    let m = build_sq(4, vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 3);
}

#[test]
fn test_johnson_figure_eight() {
    let m = build_sq(5, vec![(0, 1), (0, 3), (1, 2), (2, 0), (3, 4), (4, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_deeply_nested_blocking() {
    let m = build_sq(4, vec![(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 4);
}

#[test]
fn test_johnson_multiple_sccs_with_tails() {
    let m = build_sq(5, vec![(0, 1), (1, 0), (2, 3), (3, 4), (4, 3)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_k5_complete() {
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                edges.push((i, j));
            }
        }
    }
    let m = build_sq(5, edges);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() > 10);
}
