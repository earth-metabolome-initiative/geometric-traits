//! A dense square adjacency matrix backed by [`BitVec`] rows.
//!
//! [`BitSquareMatrix`] stores one [`BitVec`] per row, providing O(1)
//! `has_entry` and efficient bitwise neighbor-set operations (AND +
//! popcount) needed by max-clique branch-and-bound algorithms.

use alloc::vec::Vec;
use core::iter::RepeatN;

use bitvec::{
    order::Lsb0,
    slice::{BitSlice, IterOnes},
    vec::BitVec,
};

use crate::prelude::*;

// ============================================================================
// Struct
// ============================================================================

/// A dense square adjacency matrix backed by bitvec rows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitSquareMatrix {
    rows: Vec<BitVec>,
    number_of_edges: usize,
    number_of_diagonal_values: usize,
}

// ============================================================================
// Inherent methods
// ============================================================================

impl BitSquareMatrix {
    /// Creates an empty square matrix of the given order.
    #[inline]
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            rows: (0..order).map(|_| BitVec::repeat(false, order)).collect(),
            number_of_edges: 0,
            number_of_diagonal_values: 0,
        }
    }

    /// Sets the bit at `(row, col)`, updating the edge count.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize) {
        if !self.rows[row][col] {
            self.rows[row].set(col, true);
            self.number_of_edges += 1;
            if row == col {
                self.number_of_diagonal_values += 1;
            }
        }
    }

    /// Sets both `(row, col)` and `(col, row)`.
    #[inline]
    pub fn set_symmetric(&mut self, row: usize, col: usize) {
        self.set(row, col);
        self.set(col, row);
    }

    /// Clears the bit at `(row, col)`, updating the edge count.
    #[inline]
    pub fn clear(&mut self, row: usize, col: usize) {
        if self.rows[row][col] {
            self.rows[row].set(col, false);
            self.number_of_edges -= 1;
            if row == col {
                self.number_of_diagonal_values -= 1;
            }
        }
    }

    /// Returns the raw bit-slice for a row.
    #[inline]
    #[must_use]
    pub fn row_bitslice(&self, row: usize) -> &BitSlice {
        &self.rows[row]
    }

    /// Returns `|N(i) ∩ N(j)|`: the number of common neighbors of `i`
    /// and `j`, computed as word-level AND + popcount with zero allocation.
    #[inline]
    #[must_use]
    pub fn neighbor_intersection_count(&self, i: usize, j: usize) -> usize {
        self.rows[i]
            .as_raw_slice()
            .iter()
            .zip(self.rows[j].as_raw_slice())
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// Returns `|(row's neighbors) ∩ mask|`: the number of neighbors of
    /// `row` that are set in `mask`, computed without allocation.
    ///
    /// This is the core operation for max-clique branch-and-bound:
    /// `P ∩ N(v)` cardinality.
    #[inline]
    #[must_use]
    pub fn row_and_count(&self, row: usize, mask: &BitVec) -> usize {
        self.rows[row]
            .as_raw_slice()
            .iter()
            .zip(mask.as_raw_slice())
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// Builds a directed matrix from an iterator of `(row, col)` edges.
    #[inline]
    #[must_use]
    pub fn from_edges(order: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut m = Self::new(order);
        for (r, c) in edges {
            m.set(r, c);
        }
        m
    }

    /// Builds a symmetric matrix from an iterator of `(row, col)` edges.
    #[inline]
    #[must_use]
    pub fn from_symmetric_edges(
        order: usize,
        edges: impl IntoIterator<Item = (usize, usize)>,
    ) -> Self {
        let mut m = Self::new(order);
        for (r, c) in edges {
            m.set_symmetric(r, c);
        }
        m
    }
}

// ============================================================================
// Matrix trait hierarchy
// ============================================================================

impl Matrix for BitSquareMatrix {
    type Coordinates = (usize, usize);

    #[inline]
    fn shape(&self) -> Vec<usize> {
        let n = self.rows.len();
        vec![n, n]
    }
}

impl Matrix2D for BitSquareMatrix {
    type RowIndex = usize;
    type ColumnIndex = usize;

    #[inline]
    fn number_of_rows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn number_of_columns(&self) -> usize {
        self.rows.len()
    }
}

impl SquareMatrix for BitSquareMatrix {
    type Index = usize;

    #[inline]
    fn order(&self) -> usize {
        self.rows.len()
    }
}

// ============================================================================
// Custom iterators
// ============================================================================

/// Iterates all set-bit `(row, col)` pairs.
///
/// Collects coordinates up-front so that `DoubleEndedIterator`
/// interleaving is trivially correct.
#[derive(Clone)]
pub struct BitSquareMatrixSparseCoordinates<'a> {
    storage: Vec<(usize, usize)>,
    front: usize,
    back: usize,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> BitSquareMatrixSparseCoordinates<'a> {
    fn new(rows: &'a [BitVec]) -> Self {
        let storage: Vec<(usize, usize)> = rows
            .iter()
            .enumerate()
            .flat_map(|(r, bv)| bv.iter_ones().map(move |c| (r, c)))
            .collect();
        let len = storage.len();
        Self { storage, front: 0, back: len, _marker: core::marker::PhantomData }
    }
}

impl Iterator for BitSquareMatrixSparseCoordinates<'_> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.front >= self.back {
            return None;
        }
        let item = self.storage[self.front];
        self.front += 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitSquareMatrixSparseCoordinates<'_> {}

impl DoubleEndedIterator for BitSquareMatrixSparseCoordinates<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, usize)> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.storage[self.back])
    }
}

/// Iterates all column indices across all rows (flattened).
#[derive(Clone)]
pub struct BitSquareMatrixSparseColumns<'a> {
    inner: BitSquareMatrixSparseCoordinates<'a>,
}

impl Iterator for BitSquareMatrixSparseColumns<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.inner.next().map(|(_, col)| col)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for BitSquareMatrixSparseColumns<'_> {}

impl DoubleEndedIterator for BitSquareMatrixSparseColumns<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.inner.next_back().map(|(_, col)| col)
    }
}

/// Iterates row indices, repeating each row index once per defined
/// value in that row (matching the `SparseMatrix2D::sparse_rows()`
/// contract).
#[derive(Clone)]
pub struct BitSquareMatrixSparseRows<'a> {
    rows: &'a [BitVec],
    /// Current row (forward).
    front_row: usize,
    /// Current backward row (inclusive).
    back_row: usize,
    /// Remaining repeats for the current front row.
    front_repeat: RepeatN<usize>,
    /// Remaining repeats for the current back row.
    back_repeat: RepeatN<usize>,
    /// Whether front and back have merged onto the same row.
    merged: bool,
}

impl<'a> BitSquareMatrixSparseRows<'a> {
    fn new(rows: &'a [BitVec]) -> Self {
        let n = rows.len();
        if n == 0 {
            return Self {
                rows,
                front_row: 0,
                back_row: 0,
                front_repeat: core::iter::repeat_n(0, 0),
                back_repeat: core::iter::repeat_n(0, 0),
                merged: true,
            };
        }
        let front_count = rows[0].count_ones();
        let back_row = n - 1;
        let merged = n == 1;
        let back_repeat = if merged {
            core::iter::repeat_n(0, 0)
        } else {
            core::iter::repeat_n(back_row, rows[back_row].count_ones())
        };
        Self {
            rows,
            front_row: 0,
            back_row,
            front_repeat: core::iter::repeat_n(0, front_count),
            back_repeat,
            merged,
        }
    }
}

impl Iterator for BitSquareMatrixSparseRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if let Some(row) = self.front_repeat.next() {
                return Some(row);
            }
            if self.merged {
                return self.back_repeat.next();
            }
            self.front_row += 1;
            match self.front_row.cmp(&self.back_row) {
                core::cmp::Ordering::Equal => {
                    self.merged = true;
                    self.front_repeat = self.back_repeat.clone();
                    self.back_repeat = core::iter::repeat_n(0, 0);
                    // Loop back to try front_repeat
                }
                core::cmp::Ordering::Greater => {
                    self.merged = true;
                    return None;
                }
                core::cmp::Ordering::Less => {
                    self.front_repeat = core::iter::repeat_n(
                        self.front_row,
                        self.rows[self.front_row].count_ones(),
                    );
                }
            }
        }
    }
}

impl DoubleEndedIterator for BitSquareMatrixSparseRows<'_> {
    fn next_back(&mut self) -> Option<usize> {
        loop {
            if self.merged {
                return self.front_repeat.next_back();
            }
            if let Some(row) = self.back_repeat.next_back() {
                return Some(row);
            }
            if self.back_row <= self.front_row {
                self.merged = true;
                // Loop back to use front_repeat
            } else {
                self.back_row -= 1;
                if self.back_row == self.front_row {
                    self.merged = true;
                    // Loop back to use front_repeat
                } else {
                    self.back_repeat =
                        core::iter::repeat_n(self.back_row, self.rows[self.back_row].count_ones());
                }
            }
        }
    }
}

/// Iterates `count_ones()` per row.
#[derive(Clone)]
pub struct BitSquareMatrixSparseRowSizes<'a> {
    rows: &'a [BitVec],
    front: usize,
    back: usize,
}

impl<'a> BitSquareMatrixSparseRowSizes<'a> {
    fn new(rows: &'a [BitVec]) -> Self {
        Self { rows, front: 0, back: rows.len() }
    }
}

impl Iterator for BitSquareMatrixSparseRowSizes<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.front >= self.back {
            return None;
        }
        let row = self.front;
        self.front += 1;
        Some(self.rows[row].count_ones())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitSquareMatrixSparseRowSizes<'_> {}

impl DoubleEndedIterator for BitSquareMatrixSparseRowSizes<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(self.rows[self.back].count_ones())
    }
}

// ============================================================================
// Sparse trait hierarchy
// ============================================================================

impl SparseMatrix for BitSquareMatrix {
    type SparseIndex = usize;
    type SparseCoordinates<'a> = BitSquareMatrixSparseCoordinates<'a>;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        BitSquareMatrixSparseCoordinates::new(&self.rows)
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        for row in (0..self.rows.len()).rev() {
            if let Some(col) = self.rows[row].last_one() {
                return Some((row, col));
            }
        }
        None
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.number_of_edges == 0
    }
}

impl SizedSparseMatrix for BitSquareMatrix {
    #[inline]
    fn number_of_defined_values(&self) -> usize {
        self.number_of_edges
    }
}

impl SparseSquareMatrix for BitSquareMatrix {
    #[inline]
    fn number_of_defined_diagonal_values(&self) -> usize {
        self.number_of_diagonal_values
    }
}

impl SparseMatrix2D for BitSquareMatrix {
    type SparseRow<'a> = IterOnes<'a, usize, Lsb0>;
    type SparseColumns<'a> = BitSquareMatrixSparseColumns<'a>;
    type SparseRows<'a> = BitSquareMatrixSparseRows<'a>;

    #[inline]
    fn sparse_row(&self, row: usize) -> Self::SparseRow<'_> {
        self.rows[row].iter_ones()
    }

    #[inline]
    fn has_entry(&self, row: usize, column: usize) -> bool {
        self.rows[row][column]
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        BitSquareMatrixSparseColumns { inner: BitSquareMatrixSparseCoordinates::new(&self.rows) }
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        BitSquareMatrixSparseRows::new(&self.rows)
    }
}

impl SizedRowsSparseMatrix2D for BitSquareMatrix {
    type SparseRowSizes<'a> = BitSquareMatrixSparseRowSizes<'a>;

    #[inline]
    fn sparse_row_sizes(&self) -> Self::SparseRowSizes<'_> {
        BitSquareMatrixSparseRowSizes::new(&self.rows)
    }

    #[inline]
    fn number_of_defined_values_in_row(&self, row: usize) -> usize {
        self.rows[row].count_ones()
    }
}

// ============================================================================
// TransposableMatrix2D
// ============================================================================

impl TransposableMatrix2D<Self> for BitSquareMatrix {
    #[inline]
    fn transpose(&self) -> Self {
        let n = self.rows.len();
        let mut t = Self::new(n);
        for (r, bv) in self.rows.iter().enumerate() {
            for c in bv.iter_ones() {
                t.set(c, r);
            }
        }
        t
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let m = BitSquareMatrix::new(4);
        assert_eq!(m.order(), 4);
        assert_eq!(m.number_of_edges, 0);
        assert!(m.is_empty());
        assert_eq!(m.shape(), vec![4, 4]);
    }

    #[test]
    fn test_set_and_has_entry() {
        let mut m = BitSquareMatrix::new(4);
        m.set(0, 1);
        assert!(m.has_entry(0, 1));
        assert!(!m.has_entry(1, 0));
        assert_eq!(m.number_of_defined_values(), 1);
    }

    #[test]
    fn test_set_symmetric() {
        let mut m = BitSquareMatrix::new(4);
        m.set_symmetric(0, 1);
        assert!(m.has_entry(0, 1));
        assert!(m.has_entry(1, 0));
        assert_eq!(m.number_of_defined_values(), 2);
    }

    #[test]
    fn test_set_idempotent() {
        let mut m = BitSquareMatrix::new(3);
        m.set(1, 2);
        m.set(1, 2);
        assert_eq!(m.number_of_defined_values(), 1);
    }

    #[test]
    fn test_clear() {
        let mut m = BitSquareMatrix::new(3);
        m.set(0, 1);
        m.clear(0, 1);
        assert!(!m.has_entry(0, 1));
        assert_eq!(m.number_of_defined_values(), 0);
    }

    #[test]
    fn test_clear_idempotent() {
        let mut m = BitSquareMatrix::new(3);
        m.clear(0, 1);
        assert_eq!(m.number_of_defined_values(), 0);
    }

    #[test]
    fn test_diagonal_count() {
        let mut m = BitSquareMatrix::new(3);
        m.set(0, 0);
        m.set(1, 1);
        m.set(0, 1);
        assert_eq!(m.number_of_defined_diagonal_values(), 2);
        m.clear(0, 0);
        assert_eq!(m.number_of_defined_diagonal_values(), 1);
    }

    #[test]
    fn test_sparse_row() {
        let m = BitSquareMatrix::from_edges(4, vec![(0, 1), (0, 3), (1, 2)]);
        let row0: Vec<usize> = m.sparse_row(0).collect();
        assert_eq!(row0, vec![1, 3]);
        let row1: Vec<usize> = m.sparse_row(1).collect();
        assert_eq!(row1, vec![2]);
        let row2: Vec<usize> = m.sparse_row(2).collect();
        assert!(row2.is_empty());
    }

    #[test]
    fn test_sparse_row_double_ended() {
        let m = BitSquareMatrix::from_edges(5, vec![(0, 1), (0, 2), (0, 4)]);
        let row0_rev: Vec<usize> = m.sparse_row(0).rev().collect();
        assert_eq!(row0_rev, vec![4, 2, 1]);
    }

    #[test]
    fn test_sparse_coordinates() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 2), (2, 0)]);
        let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).collect();
        assert_eq!(coords, vec![(0, 1), (1, 2), (2, 0)]);
    }

    #[test]
    fn test_sparse_coordinates_double_ended() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 2), (2, 0)]);
        let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).rev().collect();
        assert_eq!(coords, vec![(2, 0), (1, 2), (0, 1)]);
    }

    #[test]
    fn test_sparse_coordinates_exact_size() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 2), (2, 0)]);
        let iter = SparseMatrix::sparse_coordinates(&m);
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_sparse_rows_repeats_per_entry() {
        // Row 0 has 2 entries, row 1 has 0, row 2 has 1
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (0, 2), (2, 0)]);
        let rows: Vec<usize> = m.sparse_rows().collect();
        assert_eq!(rows, vec![0, 0, 2]);
    }

    #[test]
    fn test_sparse_rows_double_ended() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (0, 2), (2, 0)]);
        let rows: Vec<usize> = m.sparse_rows().rev().collect();
        assert_eq!(rows, vec![2, 0, 0]);
    }

    #[test]
    fn test_sparse_rows_interleaved() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (0, 2), (2, 0)]);
        let mut iter = m.sparse_rows();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(2));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_sparse_rows_single_row() {
        let m = BitSquareMatrix::from_edges(1, vec![(0, 0)]);
        let rows: Vec<usize> = m.sparse_rows().collect();
        assert_eq!(rows, vec![0]);
    }

    #[test]
    fn test_sparse_rows_empty() {
        let m = BitSquareMatrix::new(3);
        let rows: Vec<usize> = m.sparse_rows().collect();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_sparse_columns() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 2), (1, 0)]);
        let cols: Vec<usize> = m.sparse_columns().collect();
        assert_eq!(cols, vec![2, 0]);
    }

    #[test]
    fn test_sparse_row_sizes() {
        let m = BitSquareMatrix::from_edges(4, vec![(0, 1), (0, 2), (2, 3)]);
        let sizes: Vec<usize> = m.sparse_row_sizes().collect();
        assert_eq!(sizes, vec![2, 0, 1, 0]);
    }

    #[test]
    fn test_sparse_row_sizes_double_ended() {
        let m = BitSquareMatrix::from_edges(4, vec![(0, 1), (0, 2), (2, 3)]);
        let sizes: Vec<usize> = m.sparse_row_sizes().rev().collect();
        assert_eq!(sizes, vec![0, 1, 0, 2]);
    }

    #[test]
    fn test_sparse_row_sizes_exact_size() {
        let m = BitSquareMatrix::from_edges(4, vec![(0, 1), (0, 2), (2, 3)]);
        let sizes = m.sparse_row_sizes();
        assert_eq!(sizes.len(), 4);
    }

    #[test]
    fn test_number_of_defined_values_in_row() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 0), (0, 1), (0, 2)]);
        assert_eq!(m.number_of_defined_values_in_row(0), 3);
        assert_eq!(m.number_of_defined_values_in_row(1), 0);
    }

    #[test]
    fn test_last_sparse_coordinates() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (2, 0)]);
        assert_eq!(m.last_sparse_coordinates(), Some((2, 0)));

        let empty = BitSquareMatrix::new(3);
        assert_eq!(empty.last_sparse_coordinates(), None);
    }

    #[test]
    fn test_neighbor_intersection_count() {
        let m = BitSquareMatrix::from_symmetric_edges(
            4,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        );
        // In K4, every pair shares 2 common neighbors
        assert_eq!(m.neighbor_intersection_count(0, 1), 2);
        assert_eq!(m.neighbor_intersection_count(1, 2), 2);
    }

    #[test]
    fn test_row_and_count() {
        let m = BitSquareMatrix::from_symmetric_edges(
            4,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        );
        // All 4 nodes are neighbors of 0 except 0 itself
        let mut mask = bitvec::vec::BitVec::repeat(true, 4);
        assert_eq!(m.row_and_count(0, &mask), 3);
        // Exclude node 1 from the mask
        mask.set(1, false);
        assert_eq!(m.row_and_count(0, &mask), 2);
    }

    #[test]
    fn test_from_symmetric_edges() {
        let m = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
        assert!(m.has_entry(0, 1));
        assert!(m.has_entry(1, 0));
        assert!(m.has_entry(1, 2));
        assert!(m.has_entry(2, 1));
        assert!(!m.has_entry(0, 2));
        assert_eq!(m.number_of_defined_values(), 4);
    }

    #[test]
    fn test_row_bitslice() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 0), (0, 2)]);
        let bits = m.row_bitslice(0);
        assert!(bits[0]);
        assert!(!bits[1]);
        assert!(bits[2]);
    }

    #[test]
    fn test_matrix_dimensions() {
        let m = BitSquareMatrix::new(5);
        assert_eq!(m.number_of_rows(), 5);
        assert_eq!(m.number_of_columns(), 5);
        assert_eq!(m.order(), 5);
        assert_eq!(BitSquareMatrix::dimensions(), 2);
    }

    #[test]
    fn test_transpose_directed() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 2)]);
        let t = m.transpose();
        assert!(t.has_entry(1, 0));
        assert!(t.has_entry(2, 1));
        assert!(!t.has_entry(0, 1));
        assert!(!t.has_entry(1, 2));
    }

    #[test]
    fn test_transpose_symmetric_is_identity() {
        let m = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
        let t = m.transpose();
        assert_eq!(m, t);
    }

    #[test]
    fn test_zero_order() {
        let m = BitSquareMatrix::new(0);
        assert_eq!(m.order(), 0);
        assert!(m.is_empty());
        assert_eq!(m.last_sparse_coordinates(), None);
        let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&m).collect();
        assert!(coords.is_empty());
        let rows: Vec<usize> = m.sparse_rows().collect();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_sparse_coordinates_mixed_forward_backward() {
        let m = BitSquareMatrix::from_edges(3, vec![(0, 0), (0, 2), (1, 1), (2, 0)]);
        let mut iter = SparseMatrix::sparse_coordinates(&m);
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next_back(), Some((2, 0)));
        assert_eq!(iter.next_back(), Some((1, 1)));
        assert_eq!(iter.next(), Some((0, 2)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_density() {
        let m = BitSquareMatrix::from_edges(2, vec![(0, 0), (0, 1)]);
        let d = m.density();
        assert!((d - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_density_zero() {
        let m = BitSquareMatrix::new(0);
        assert!((m.density() - 0.0).abs() < f64::EPSILON);
    }
}
