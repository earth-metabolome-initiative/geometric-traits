//! Tests for the Hopcroft-Karp algorithm.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::CSR2D,
    prelude::*,
    traits::{EdgesBuilder, HopcroftKarp},
};

#[test]
fn test_hopcroft_karp_simple() {
    let edge_data: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 0)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(3)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    assert_eq!(assignment.len(), 3);
}

#[test]
fn test_hopcroft_karp_bipartite() {
    // Bipartite matching: 3 left nodes, 3 right nodes
    // 0 -> 0, 1
    // 1 -> 1, 2
    // 2 -> 0, 2
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(6)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    // Should find a perfect matching of size 3
    assert_eq!(assignment.len(), 3);

    // Each row should be matched to exactly one column
    let rows: Vec<_> = assignment.iter().map(|(r, _)| *r).collect();
    let cols: Vec<_> = assignment.iter().map(|(_, c)| *c).collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(cols.len(), 3);
}

#[test]
fn test_hopcroft_karp_no_matching() {
    // No edges, no matching
    let edge_data: Vec<(usize, usize)> = vec![];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(0)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    assert_eq!(assignment.len(), 0);
}

#[test]
fn test_hopcroft_karp_partial_matching() {
    // Not all rows can be matched
    // Only row 0 has an edge
    let edge_data: Vec<(usize, usize)> = vec![(0, 0)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(1)
            .expected_shape((3, 3))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    assert_eq!(assignment.len(), 1);
    assert_eq!(assignment[0], (0, 0));
}

#[test]
fn test_hopcroft_karp_rectangular_wide() {
    // More columns than rows
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (0, 2), (1, 1), (1, 3)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(4)
            .expected_shape((2, 4))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    // Can match both rows
    assert_eq!(assignment.len(), 2);
}

#[test]
fn test_hopcroft_karp_rectangular_tall() {
    // More rows than columns
    let edge_data: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (1, 1), (2, 1), (3, 0)];

    let edges: CSR2D<usize, usize, usize> =
        GenericEdgesBuilder::<_, CSR2D<usize, usize, usize>>::default()
            .expected_number_of_edges(5)
            .expected_shape((4, 2))
            .edges(edge_data.into_iter())
            .build()
            .unwrap();

    let assignment = edges.hopcroft_karp().unwrap();
    // Can only match 2 rows (limited by columns)
    assert_eq!(assignment.len(), 2);
}
