//! Compactification of sparse matrices by remapping row and column indices to
//! contiguous 0..n ranges, eliminating rows and columns with no entries.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::{
    impls::ValuedCSR2D,
    traits::{MatrixMut, SparseMatrixMut, SparseValuedMatrix2D, TryFromUsize},
};

/// A sparse matrix whose row/column indices have been compacted to contiguous
/// 0-based ranges, along with reverse mappings back to the original indices.
pub struct CompactMatrix<RowIndex, ColumnIndex> {
    /// The compacted matrix with indices in 0..n_unique_rows ×
    /// 0..n_unique_cols.
    pub matrix: ValuedCSR2D<usize, usize, usize, f64>,
    /// Maps compact row index → original row index.
    pub row_map: Vec<RowIndex>,
    /// Maps compact column index → original column index.
    pub col_map: Vec<ColumnIndex>,
}

/// Compactifies a sparse valued matrix by collecting only the rows and columns
/// that have at least one entry, remapping their indices to 0..n_unique_rows
/// and 0..n_unique_cols respectively.
///
/// The returned [`CompactMatrix`] contains:
/// - A dense-indexed `ValuedCSR2D<usize, usize, usize, f64>` with remapped
///   coordinates and f64 values copied from the original matrix.
/// - `row_map`: compact row → original row index.
/// - `col_map`: compact column → original column index.
///
/// This is useful for reducing a large sparse matrix (with many empty
/// rows/cols) down to only the rows and columns that participate in edges,
/// before feeding it to a dense solver.
#[inline]
pub fn compactify<M>(matrix: &M) -> CompactMatrix<M::RowIndex, M::ColumnIndex>
where
    M: SparseValuedMatrix2D,
    M::Value: Into<f64>,
    M::RowIndex: AsPrimitive<usize> + TryFromUsize + Ord,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize + Ord,
{
    // Collect unique rows that have at least one sparse entry.
    let mut row_map: Vec<M::RowIndex> = Vec::new();
    let mut col_set: Vec<M::ColumnIndex> = Vec::new();

    for row in matrix.row_indices() {
        let mut has_entry = false;
        for col in matrix.sparse_row(row) {
            has_entry = true;
            col_set.push(col);
        }
        if has_entry {
            row_map.push(row);
        }
    }

    // Deduplicate and sort columns.
    col_set.sort_unstable();
    col_set.dedup();
    let col_map = col_set;

    let n_rows = row_map.len();
    let n_cols = col_map.len();

    if n_rows == 0 || n_cols == 0 {
        return CompactMatrix {
            matrix: SparseMatrixMut::with_sparse_shape((0, 0)),
            row_map,
            col_map,
        };
    }

    // Build column index → compact column index lookup.
    // Since col_map is sorted, we use binary search.
    let col_to_compact = |col: M::ColumnIndex| -> usize {
        col_map.binary_search(&col).expect("column must be in col_map")
    };

    // Count total edges for capacity.
    let n_edges: usize = row_map.iter().map(|&row| matrix.sparse_row(row).count()).sum();

    let mut compact: ValuedCSR2D<usize, usize, usize, f64> =
        SparseMatrixMut::with_sparse_shaped_capacity((n_rows, n_cols), n_edges);

    for (compact_row, &orig_row) in row_map.iter().enumerate() {
        for (col, value) in matrix.sparse_row(orig_row).zip(matrix.sparse_row_values(orig_row)) {
            let compact_col = col_to_compact(col);
            compact
                .add((compact_row, compact_col, value.into()))
                .expect("Failed to add entry to compact matrix");
        }
    }

    CompactMatrix { matrix: compact, row_map, col_map }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::traits::{Matrix2D, SparseMatrix, SparseMatrix2D, SparseValuedMatrix2D};

    #[test]
    fn test_compactify_basic() {
        // 5×5 sparse matrix with entries only in rows 1,3 and cols 0,4.
        let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
            SparseMatrixMut::with_sparse_shaped_capacity((5, 5), 3);
        csr.add((1, 0, 0.5)).unwrap();
        csr.add((1, 4, 0.3)).unwrap();
        csr.add((3, 4, 0.1)).unwrap();

        let compact = compactify(&csr);

        // Should have 2 unique rows and 2 unique columns.
        assert_eq!(compact.matrix.number_of_rows(), 2);
        assert_eq!(compact.matrix.number_of_columns(), 2);
        assert_eq!(compact.row_map, vec![1u8, 3u8]);
        assert_eq!(compact.col_map, vec![0u8, 4u8]);

        // 3 edges preserved.
        assert_eq!(compact.matrix.sparse_row(0).count(), 2);
        assert_eq!(compact.matrix.sparse_row(1).count(), 1);
    }

    #[test]
    fn test_compactify_empty() {
        let csr: ValuedCSR2D<u8, u8, u8, f64> =
            SparseMatrixMut::with_sparse_shaped_capacity((3, 3), 0);

        let compact = compactify(&csr);
        assert!(compact.row_map.is_empty());
        assert!(compact.col_map.is_empty());
        assert!(compact.matrix.is_empty());
    }

    #[test]
    fn test_compactify_preserves_values() {
        let mut csr: ValuedCSR2D<u8, u8, u8, f64> =
            SparseMatrixMut::with_sparse_shaped_capacity((3, 3), 2);
        csr.add((0, 2, 1.5)).unwrap();
        csr.add((2, 0, 2.5)).unwrap();

        let compact = compactify(&csr);
        assert_eq!(compact.row_map, vec![0u8, 2u8]);
        assert_eq!(compact.col_map, vec![0u8, 2u8]);

        // Check values via sparse_row_values.
        let vals: Vec<f64> = compact.matrix.sparse_row_values(0).collect();
        assert_eq!(vals, vec![1.5]);
        let vals: Vec<f64> = compact.matrix.sparse_row_values(1).collect();
        assert_eq!(vals, vec![2.5]);
    }

    #[test]
    fn test_compactify_already_compact() {
        // 2×2 dense matrix — should remain 2×2.
        let csr: ValuedCSR2D<u8, u8, u8, f64> =
            ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0]]).unwrap();

        let compact = compactify(&csr);
        assert_eq!(compact.matrix.number_of_rows(), 2);
        assert_eq!(compact.matrix.number_of_columns(), 2);
        assert_eq!(compact.row_map, vec![0u8, 1u8]);
        assert_eq!(compact.col_map, vec![0u8, 1u8]);
    }
}
