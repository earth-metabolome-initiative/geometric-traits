//! Coverage-oriented tests for `test_utils` invariant checkers on malformed
//! sparse matrix implementations.
#![cfg(all(feature = "std", feature = "arbitrary"))]

use geometric_traits::{
    prelude::*,
    test_utils::{check_sparse_matrix_invariants, check_valued_matrix_invariants},
    traits::{
        Matrix, Matrix2D, SparseMatrix, SparseMatrix2D, SparseValuedMatrix, SparseValuedMatrix2D,
        ValuedMatrix, ValuedMatrix2D,
    },
};

struct FakeSparseMatrix {
    rows: u8,
    columns: u8,
    sparse_rows: Vec<Vec<u8>>,
    sparse_coordinates: Vec<(u8, u8)>,
}

impl Matrix for FakeSparseMatrix {
    type Coordinates = (u8, u8);

    fn shape(&self) -> Vec<usize> {
        vec![usize::from(self.rows), usize::from(self.columns)]
    }
}

impl Matrix2D for FakeSparseMatrix {
    type RowIndex = u8;
    type ColumnIndex = u8;

    fn number_of_rows(&self) -> Self::RowIndex {
        self.rows
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.columns
    }
}

impl SparseMatrix for FakeSparseMatrix {
    type SparseIndex = u16;
    type SparseCoordinates<'a>
        = std::vec::IntoIter<(u8, u8)>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.sparse_coordinates.clone().into_iter()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.sparse_coordinates.last().copied()
    }

    fn is_empty(&self) -> bool {
        self.sparse_coordinates.is_empty()
    }
}

impl SparseMatrix2D for FakeSparseMatrix {
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
        self.sparse_rows[usize::from(row)].clone().into_iter()
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.sparse_coordinates
            .iter()
            .map(|(_, column)| *column)
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        (0..self.rows).collect::<Vec<_>>().into_iter()
    }
}

struct FakeSparseValuedMatrix {
    matrix: FakeSparseMatrix,
    sparse_row_values: Vec<Vec<f64>>,
}

impl Matrix for FakeSparseValuedMatrix {
    type Coordinates = (u8, u8);

    fn shape(&self) -> Vec<usize> {
        self.matrix.shape()
    }
}

impl Matrix2D for FakeSparseValuedMatrix {
    type RowIndex = u8;
    type ColumnIndex = u8;

    fn number_of_rows(&self) -> Self::RowIndex {
        self.matrix.number_of_rows()
    }

    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.matrix.number_of_columns()
    }
}

impl SparseMatrix for FakeSparseValuedMatrix {
    type SparseIndex = u16;
    type SparseCoordinates<'a>
        = std::vec::IntoIter<(u8, u8)>
    where
        Self: 'a;

    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        self.matrix.sparse_coordinates()
    }

    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.matrix.last_sparse_coordinates()
    }

    fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
}

impl SparseMatrix2D for FakeSparseValuedMatrix {
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
        self.matrix.sparse_row(row)
    }

    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.matrix.sparse_columns()
    }

    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.matrix.sparse_rows()
    }
}

impl ValuedMatrix for FakeSparseValuedMatrix {
    type Value = f64;
}

impl ValuedMatrix2D for FakeSparseValuedMatrix {}

impl SparseValuedMatrix for FakeSparseValuedMatrix {
    type SparseValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_values(&self) -> Self::SparseValues<'_> {
        self.sparse_row_values
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl SparseValuedMatrix2D for FakeSparseValuedMatrix {
    type SparseRowValues<'a>
        = std::vec::IntoIter<f64>
    where
        Self: 'a;

    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        self.sparse_row_values[usize::from(row)].clone().into_iter()
    }
}

#[test]
#[should_panic(expected = "not sorted")]
fn test_check_sparse_matrix_invariants_panics_when_coordinates_are_unsorted() {
    let matrix = FakeSparseMatrix {
        rows: 2,
        columns: 3,
        sparse_rows: vec![vec![0, 2], vec![1]],
        sparse_coordinates: vec![(0, 2), (0, 0), (1, 1)],
    };
    check_sparse_matrix_invariants(&matrix);
}

#[test]
#[should_panic(expected = "is not sorted")]
fn test_check_sparse_matrix_invariants_panics_when_row_is_not_sorted() {
    let matrix = FakeSparseMatrix {
        rows: 2,
        columns: 3,
        sparse_rows: vec![vec![2, 0], vec![1]],
        sparse_coordinates: vec![(0, 0), (0, 2), (1, 1)],
    };
    check_sparse_matrix_invariants(&matrix);
}

#[test]
#[should_panic(expected = "has duplicates")]
fn test_check_sparse_matrix_invariants_panics_when_row_has_duplicates() {
    let matrix = FakeSparseMatrix {
        rows: 2,
        columns: 2,
        sparse_rows: vec![vec![0, 0], vec![1]],
        sparse_coordinates: vec![(0, 0), (1, 1)],
    };
    check_sparse_matrix_invariants(&matrix);
}

#[test]
#[should_panic(expected = "have duplicates")]
fn test_check_sparse_matrix_invariants_panics_when_coordinates_have_duplicates() {
    let matrix = FakeSparseMatrix {
        rows: 2,
        columns: 2,
        sparse_rows: vec![vec![0], vec![1]],
        sparse_coordinates: vec![(0, 0), (0, 0), (1, 1)],
    };
    check_sparse_matrix_invariants(&matrix);
}

#[test]
#[should_panic(expected = "different lengths for column indices and values")]
fn test_check_valued_matrix_invariants_panics_when_row_lengths_mismatch() {
    let matrix = FakeSparseValuedMatrix {
        matrix: FakeSparseMatrix {
            rows: 2,
            columns: 2,
            sparse_rows: vec![vec![0, 1], vec![0]],
            sparse_coordinates: vec![(0, 0), (0, 1), (1, 0)],
        },
        sparse_row_values: vec![vec![10.0], vec![20.0]],
    };
    check_valued_matrix_invariants(&matrix);
}
