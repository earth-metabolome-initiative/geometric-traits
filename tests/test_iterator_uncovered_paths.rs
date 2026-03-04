//! Targeted coverage tests for iterator edge branches.

use geometric_traits::{
    impls::{CSR2D, GenericImplicitValuedMatrix2D},
    naive_structs::GenericEdgesBuilder,
    traits::{EdgesBuilder, EmptyRows, ImplicitValuedSparseMatrix, SizedRowsSparseMatrix2D},
};

type TestCSR = CSR2D<usize, usize, usize>;

fn build_csr(mut edges: Vec<(usize, usize)>, shape: (usize, usize)) -> TestCSR {
    edges.sort_unstable();
    GenericEdgesBuilder::<_, TestCSR>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(shape)
        .edges(edges.into_iter())
        .build()
        .expect("build csr")
}

#[test]
fn test_empty_row_indices_next_back_reaches_none_path() {
    let csr = build_csr(vec![], (1, 1));
    let mut empty_rows = csr.empty_row_indices();

    assert_eq!(empty_rows.next_back(), Some(0));
    assert_eq!(empty_rows.next_back(), None);
}

#[test]
fn test_non_empty_row_indices_next_back_reaches_none_path() {
    let csr = build_csr(vec![(0, 0)], (1, 1));
    let mut non_empty_rows = csr.non_empty_row_indices();

    assert_eq!(non_empty_rows.next_back(), Some(0));
    assert_eq!(non_empty_rows.next_back(), None);
}

#[test]
fn test_sparse_row_sizes_single_row_next_back_sets_exhausted() {
    let csr = build_csr(vec![], (1, 1));
    let mut row_sizes = csr.sparse_row_sizes();

    assert_eq!(row_sizes.next_back(), Some(0));
    assert_eq!(row_sizes.next_back(), None);
}

#[test]
fn test_implicit_sparse_values_len_and_next_back_paths() {
    let csr = build_csr(vec![(0, 1), (1, 2)], (2, 3));
    let matrix = GenericImplicitValuedMatrix2D::new(csr, |(row, column)| {
        f64::from(u16::try_from(row * 10 + column).expect("small matrix index should fit in u16"))
    });
    let mut values = matrix.sparse_implicit_values();

    assert_eq!(values.len(), 2);
    assert_eq!(values.next_back(), Some(12.0));
    assert_eq!(values.next(), Some(1.0));
    assert_eq!(values.next_back(), None);
}
