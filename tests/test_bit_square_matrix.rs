//! Integration tests for [`BitSquareMatrix`].

use geometric_traits::{prelude::*, traits::SparseMatrix};

#[test]
fn test_k4_properties() {
    let m = BitSquareMatrix::from_symmetric_edges(
        4,
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    );

    assert_eq!(m.order(), 4);
    assert_eq!(m.number_of_defined_values(), 12); // 4*3 directed edges
    assert_eq!(m.number_of_defined_diagonal_values(), 0);

    // Each vertex has degree 3
    for i in 0..4 {
        assert_eq!(m.number_of_defined_values_in_row(i), 3);
    }

    // Any two vertices share exactly 2 common neighbors
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_eq!(m.neighbor_intersection_count(i, j), 2);
        }
    }
}

#[test]
fn test_sparse_row_exact_size() {
    let m = BitSquareMatrix::from_edges(5, vec![(0, 1), (0, 3), (0, 4)]);
    let row = m.sparse_row(0);
    assert_eq!(row.len(), 3);
}

#[test]
fn test_sparse_coordinates_round_trip() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let m = BitSquareMatrix::from_edges(4, edges.clone());
    let collected: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).collect();
    assert_eq!(collected, edges);
}

#[test]
fn test_transpose_directed() {
    let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 2)]);
    let t = m.transpose();
    // Transpose swaps (0,1) -> (1,0) and (1,2) -> (2,1)
    assert!(t.has_entry(1, 0));
    assert!(t.has_entry(2, 1));
    assert!(!t.has_entry(0, 1));
    assert!(!t.has_entry(1, 2));
    assert_eq!(t.number_of_defined_values(), 2);
}

#[test]
fn test_density_full_graph() {
    let m = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    // 6 directed edges out of 9 possible cells
    let d = m.density();
    assert!((d - 6.0 / 9.0).abs() < 1e-10);
}

#[test]
fn test_single_node_self_loop() {
    let m = BitSquareMatrix::from_edges(1, vec![(0, 0)]);
    assert_eq!(m.number_of_defined_values(), 1);
    assert_eq!(m.number_of_defined_diagonal_values(), 1);
    assert!(m.has_entry(0, 0));
    let coords: Vec<_> = SparseMatrix::sparse_coordinates(&m).collect();
    assert_eq!(coords, vec![(0, 0)]);
}

#[test]
fn test_row_indices_and_column_indices() {
    let m = BitSquareMatrix::new(3);
    let rows: Vec<usize> = m.row_indices().collect();
    assert_eq!(rows, vec![0, 1, 2]);
    let cols: Vec<usize> = m.column_indices().collect();
    assert_eq!(cols, vec![0, 1, 2]);
}

#[test]
fn test_sparse_rows_contract() {
    // Verify sparse_rows() yields row indices repeated per entry
    let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (0, 2), (2, 0)]);
    let rows: Vec<usize> = m.sparse_rows().collect();
    assert_eq!(rows, vec![0, 0, 2]);
}

#[test]
fn test_row_and_count() {
    use bitvec::vec::BitVec;
    let m = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2)]);
    let mut mask = BitVec::repeat(true, 4);
    // Node 0 has neighbors {1, 2}, all in mask
    assert_eq!(m.row_and_count(0, &mask), 2);
    // Exclude node 2
    mask.set(2, false);
    assert_eq!(m.row_and_count(0, &mask), 1);
}

#[cfg(feature = "arbitrary")]
mod with_arbitrary {
    use geometric_traits::{prelude::*, test_utils::check_sparse_matrix_invariants};

    #[test]
    fn test_invariants_empty() {
        let m = BitSquareMatrix::new(5);
        check_sparse_matrix_invariants(&m);
    }

    #[test]
    fn test_invariants_directed() {
        let m = BitSquareMatrix::from_edges(4, vec![(0, 1), (0, 3), (1, 2), (3, 0)]);
        check_sparse_matrix_invariants(&m);
    }

    #[test]
    fn test_invariants_symmetric() {
        let m = BitSquareMatrix::from_symmetric_edges(
            4,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        );
        check_sparse_matrix_invariants(&m);
    }
}
