//! Test for a specific InChI failure case.
#![cfg(feature = "std")]

use geometric_traits::{
    errors::builder::edges::EdgesBuilderError, prelude::UndiEdgesBuilder, traits::EdgesBuilder,
};

#[test]
/// Test function for InChI failure case.
#[inline]
pub fn test_inchi_failure() {
    let vocabulary_size = 20;
    let edges: Vec<(usize, usize)> = vec![
        (0, 1),
        (1, 4),
        (1, 7),
        (2, 3),
        (2, 8),
        (3, 5),
        (3, 9),
        (4, 2),
        (5, 10),
        (5, 11),
        (6, 0),
        (11, 4),
    ];
    let error = UndiEdgesBuilder::default()
        .expected_shape(vocabulary_size)
        .expected_number_of_edges(edges.len())
        .edges(edges.into_iter())
        .build();

    assert_eq!(
        error,
        Err(EdgesBuilderError::MatrixError(geometric_traits::impls::MutabilityError::OutOfBounds(
            (4, 2),
            (20, 20),
            "In an upper triangular matrix, row indices must be less than or equal to column indices."
        )))
    );
}
