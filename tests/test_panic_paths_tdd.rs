//! TDD tests for panic-path hardening in LAP wrappers and padded-diagonal
//! wrapper.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{GenericMatrix2DWithPaddedDiagonal, ValuedCSR2D},
    prelude::{Jaqaman, LAPError, SparseMatrix2D, SparseValuedMatrix2D},
    traits::{
        Matrix, Matrix2D, SparseMatrix, SparseMatrixMut, SparseValuedMatrix,
        SparseValuedMatrix2D as SparseValuedMatrix2DTrait, ValuedMatrix, ValuedMatrix2D,
    },
};

#[test]
fn test_jaqaman_non_empty_matrix_with_no_sparse_edges_returns_empty_assignment() {
    let matrix = NoEdgeButNonEmptySparseValuedMatrix;

    let result = matrix.jaqaman(900.0, 1000.0);
    assert_eq!(result, Ok(Vec::new()));
}

#[test]
fn test_jaqaman_malformed_sparse_input_returns_typed_error() {
    let malformed = DuplicateEdgeSparseValuedMatrix;

    let result = malformed.jaqaman(900.0, 1000.0);
    assert_eq!(result, Err(LAPError::ExpandedMatrixBuildFailed));
}

#[test]
fn test_padded_diagonal_sparse_row_when_row_cannot_convert_to_column_returns_empty() {
    let inner: ValuedCSR2D<usize, u16, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10u16, 10u8), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(inner, |_: u16| 0.0)
        .expect("construction should succeed");

    assert!(padded.sparse_row(300u16).collect::<Vec<u8>>().is_empty());
    assert!(padded.sparse_row_values(300u16).collect::<Vec<f64>>().is_empty());
}

struct DuplicateEdgeSparseValuedMatrix;
struct NoEdgeButNonEmptySparseValuedMatrix;

impl Matrix for DuplicateEdgeSparseValuedMatrix {
    type Coordinates = (u8, u8);

    fn shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}

impl Matrix2D for DuplicateEdgeSparseValuedMatrix {
    type RowIndex = u8;
    type ColumnIndex = u8;

    fn number_of_rows(&self) -> Self::RowIndex {
        1
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        1
    }
}

impl SparseMatrix for DuplicateEdgeSparseValuedMatrix {
    type SparseIndex = u16;
    type SparseCoordinates<'a>
        = std::vec::IntoIter<(u8, u8)>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        vec![(0, 0), (0, 0)].into_iter()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        Some((0, 0))
    }

    fn is_empty(&self) -> bool {
        false
    }
}

impl SparseMatrix2D for DuplicateEdgeSparseValuedMatrix {
    type SparseRow<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;

    type SparseColumns<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;

    type SparseRows<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;

    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        if row == 0 { vec![0, 0].into_iter() } else { Vec::new().into_iter() }
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        vec![0, 0].into_iter()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        vec![0, 0].into_iter()
    }
}

impl ValuedMatrix for DuplicateEdgeSparseValuedMatrix {
    type Value = f64;
}

impl ValuedMatrix2D for DuplicateEdgeSparseValuedMatrix {}

impl SparseValuedMatrix for DuplicateEdgeSparseValuedMatrix {
    type SparseValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_values(&self) -> Self::SparseValues<'_> {
        vec![1.0, 2.0].into_iter()
    }
}

impl SparseValuedMatrix2DTrait for DuplicateEdgeSparseValuedMatrix {
    type SparseRowValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        if row == 0 { vec![1.0, 2.0].into_iter() } else { Vec::new().into_iter() }
    }
}

impl Matrix for NoEdgeButNonEmptySparseValuedMatrix {
    type Coordinates = (u8, u8);

    fn shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}

impl Matrix2D for NoEdgeButNonEmptySparseValuedMatrix {
    type RowIndex = u8;
    type ColumnIndex = u8;

    fn number_of_rows(&self) -> Self::RowIndex {
        1
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        1
    }
}

impl SparseMatrix for NoEdgeButNonEmptySparseValuedMatrix {
    type SparseIndex = u16;
    type SparseCoordinates<'a>
        = std::vec::IntoIter<(u8, u8)>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        Vec::new().into_iter()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        None
    }

    fn is_empty(&self) -> bool {
        // Intentionally inconsistent to exercise wrapper panic path.
        false
    }
}

impl SparseMatrix2D for NoEdgeButNonEmptySparseValuedMatrix {
    type SparseRow<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;
    type SparseColumns<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;
    type SparseRows<'a>
        = std::vec::IntoIter<u8>
    where
        Self: 'a;

    fn sparse_row(&self, _row: Self::RowIndex) -> Self::SparseRow<'_> {
        Vec::new().into_iter()
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        Vec::new().into_iter()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        Vec::new().into_iter()
    }
}

impl ValuedMatrix for NoEdgeButNonEmptySparseValuedMatrix {
    type Value = f64;
}

impl ValuedMatrix2D for NoEdgeButNonEmptySparseValuedMatrix {}

impl SparseValuedMatrix for NoEdgeButNonEmptySparseValuedMatrix {
    type SparseValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_values(&self) -> Self::SparseValues<'_> {
        Vec::new().into_iter()
    }
}

impl SparseValuedMatrix2DTrait for NoEdgeButNonEmptySparseValuedMatrix {
    type SparseRowValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_row_values(&self, _row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        Vec::new().into_iter()
    }
}
