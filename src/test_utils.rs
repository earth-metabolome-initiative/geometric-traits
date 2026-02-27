//! Testing utilities for constructing type instances from raw bytes and
//! replaying fuzz corpus or crash files, and shared invariant-checking
//! functions used by both fuzz targets and regression tests.
//!
//! This module is available when the `arbitrary` feature is enabled. It
//! provides helpers used by both fuzz targets and regression tests, so
//! crash files produced by fuzzing can be directly replayed as unit tests.

use alloc::vec::Vec;
use core::fmt::Debug;

use arbitrary::{Arbitrary, Unstructured};

use crate::{
    prelude::*,
    traits::{
        EdgesBuilder, IntoUsize, SparseMatrix, SparseMatrix2D, SparseValuedMatrix,
        SparseValuedMatrix2D,
    },
};

// ============================================================================
// Deserialization helpers
// ============================================================================

/// Construct a value of type `T` from raw bytes using the [`Arbitrary`] trait.
///
/// Returns `None` if the bytes are insufficient or do not produce a valid
/// instance.
pub fn from_bytes<T: for<'a> Arbitrary<'a>>(bytes: &[u8]) -> Option<T> {
    let mut u = Unstructured::new(bytes);
    T::arbitrary(&mut u).ok()
}

/// Load all files from a directory and construct instances of `T` from each
/// file's raw bytes.
///
/// Files that fail to produce valid instances are silently skipped.
/// Returns an empty vector if the directory does not exist or is unreadable.
pub fn replay_dir<T: for<'a> Arbitrary<'a>>(dir: &std::path::Path) -> Vec<T> {
    let mut results = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return results;
    };
    for entry in entries.flatten() {
        if entry.path().is_file() {
            if let Ok(bytes) = std::fs::read(entry.path()) {
                if let Some(instance) = from_bytes::<T>(&bytes) {
                    results.push(instance);
                }
            }
        }
    }
    results
}

// ============================================================================
// CSR2D invariants (from fuzz/fuzz_targets/csr2d.rs)
// ============================================================================

/// Check that a sparse matrix has sorted, unique columns per row and sorted,
/// unique global coordinates.
///
/// # Panics
///
/// Panics if any row has unsorted or duplicate column indices, or if the
/// global sparse coordinates are unsorted or contain duplicates.
pub fn check_sparse_matrix_invariants<M>(csr: &M)
where
    M: SparseMatrix2D,
    M::ColumnIndex: Ord + Clone + Debug,
    M::RowIndex: Debug,
    M::Coordinates: Ord + Clone + Debug + PartialEq,
{
    for row_index in csr.row_indices() {
        let column_indices: Vec<M::ColumnIndex> = csr.sparse_row(row_index).collect();
        let mut sorted_column_indices = column_indices.clone();
        sorted_column_indices.sort_unstable();
        assert_eq!(
            column_indices, sorted_column_indices,
            "The row {row_index:?} is not sorted"
        );
        sorted_column_indices.dedup();
        assert_eq!(
            column_indices, sorted_column_indices,
            "The row {row_index:?} has duplicates"
        );
    }

    let sparse_coordinates: Vec<M::Coordinates> =
        SparseMatrix::sparse_coordinates(csr).collect();
    let mut clone_sparse_coordinates = sparse_coordinates.clone();
    clone_sparse_coordinates.sort_unstable();
    assert_eq!(
        sparse_coordinates, clone_sparse_coordinates,
        "The sparse coordinates are not sorted"
    );
    clone_sparse_coordinates.dedup();
    assert_eq!(
        sparse_coordinates, clone_sparse_coordinates,
        "The sparse coordinates have duplicates"
    );
}

// ============================================================================
// ValuedCSR2D invariants (from fuzz/fuzz_targets/valued_csr2d.rs)
// ============================================================================

/// Check that each row of a valued sparse matrix has the same number of
/// column indices and values.
///
/// # Panics
///
/// Panics if any row has a different count of column indices vs. values.
pub fn check_valued_matrix_invariants<M>(csr: &M)
where
    M: SparseValuedMatrix2D,
    M::RowIndex: Debug,
{
    for row_index in csr.row_indices() {
        let column_indices: Vec<M::ColumnIndex> = csr.sparse_row(row_index).collect();
        let column_values: Vec<M::Value> = csr.sparse_row_values(row_index).collect();
        assert_eq!(
            column_indices.len(),
            column_values.len(),
            "The row {row_index:?} has different lengths for column indices and values"
        );
    }
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal invariants
// (from fuzz/fuzz_targets/generic_matrix2d_with_padded_diagonal.rs)
// ============================================================================

/// Type alias for the padded diagonal type used by fuzz targets.
pub type FuzzPaddedDiag =
    GenericMatrix2DWithPaddedDiagonal<ValuedCSR2D<u16, u8, u8, f64>, fn(u8) -> f64>;

/// Check invariants of a [`GenericMatrix2DWithPaddedDiagonal`]: the matrix
/// must be square, every row must contain its diagonal element, column/value
/// counts must match, and imputation flags must be consistent with the
/// underlying matrix.
///
/// # Panics
///
/// Panics if any invariant is violated.
pub fn check_padded_diagonal_invariants(
    padded_csr: &FuzzPaddedDiag,
) {
    assert_eq!(
        padded_csr.number_of_rows(),
        padded_csr.number_of_columns(),
        "The number of rows and columns should be equal",
    );

    for row_index in padded_csr.row_indices() {
        // Check that the diagonal of the row is imputed.
        let mut sparse_column_indices = padded_csr.sparse_row(row_index);
        sparse_column_indices.find(|column_index| *column_index == row_index).expect(
            "The diagonal of the row should always be imputed but was not found in the sparse row",
        );

        // Check that the number of sparse column indices and values are equal.
        let number_of_sparse_column_indices = padded_csr.sparse_row(row_index).count();
        let number_of_sparse_column_values = padded_csr.sparse_row_values(row_index).count();

        assert_eq!(
            number_of_sparse_column_indices, number_of_sparse_column_values,
            "The number of sparse column indices and values should be equal"
        );

        // Check that the `is_diagonal_imputed` method works as expected.
        let underlying_matrix = padded_csr.matrix();
        let has_diagonal = if row_index < underlying_matrix.number_of_rows() {
            underlying_matrix
                .sparse_row(row_index)
                .any(|column_index| column_index == row_index)
        } else {
            false
        };
        let is_diagonal_imputed = padded_csr.is_diagonal_imputed(row_index);
        assert_eq!(
            has_diagonal, !is_diagonal_imputed,
            "The inner diagonal state was `{has_diagonal}` but the `is_diagonal_imputed` method returned `{is_diagonal_imputed}`"
        );

        // Check that the number of elements is consistent.
        let expected_number_of_elements = if row_index < underlying_matrix.number_of_rows() {
            let number_of_inner_sparse_column_indices =
                underlying_matrix.sparse_row(row_index).count();
            if has_diagonal {
                number_of_inner_sparse_column_indices
            } else {
                number_of_inner_sparse_column_indices + 1
            }
        } else {
            1
        };

        assert_eq!(
            number_of_sparse_column_indices, expected_number_of_elements,
            "The number of elements in the padded sparse row should be equal to the number of \
             elements in the inner sparse row plus the diagonal element if it has been imputed"
        );
    }
}

// ============================================================================
// PaddedMatrix2D invariants (from fuzz/fuzz_targets/padded_matrix2d.rs)
// ============================================================================

/// Check invariants of a [`PaddedMatrix2D`]: all values from the underlying
/// CSR matrix must appear in the padded matrix, and imputed values must use
/// the padding value.
///
/// # Panics
///
/// Panics if any invariant is violated.
pub fn check_padded_matrix2d_invariants(
    csr: &ValuedCSR2D<u16, u8, u8, u8>,
) {
    let Ok(padded_matrix) = PaddedMatrix2D::new(csr, |_| 1) else {
        return;
    };
    let padded_number_of_rows = padded_matrix.number_of_rows();
    let padded_number_of_columns = padded_matrix.number_of_columns();
    let csr_number_of_rows = csr.number_of_rows();
    let csr_number_of_columns = csr.number_of_columns();
    let mut last_tuple = None;

    for row_index in csr.row_indices() {
        let csr_column_values: Vec<(u8, u8)> = csr
            .sparse_row(row_index)
            .zip(csr.sparse_row_values(row_index))
            .collect();
        let padded_column_values: Vec<(u8, u8)> = padded_matrix
            .column_indices()
            .zip(padded_matrix.row_values(row_index))
            .collect();

        for &(column_index, value) in &csr_column_values {
            assert!(
                padded_column_values.contains(&(column_index, value)),
                "The padded matrix does not contain the value {value} (last tuple was \
                 {last_tuple:?}) at column index {column_index}/{padded_number_of_columns} \
                 ({csr_number_of_columns}) for row index {row_index}/{padded_number_of_rows} \
                 ({csr_number_of_rows}). {csr:?}"
            );
            last_tuple = Some((column_index, value));
        }

        for (column_index, value) in padded_column_values {
            if padded_matrix.is_imputed((row_index, column_index)) {
                assert_eq!(value, 1);
            } else {
                assert!(
                    csr_column_values.contains(&(column_index, value)),
                    "The csr matrix does not contain the value {value} at column index \
                     {column_index} for row index {row_index}"
                );
            }
        }
    }
}

// ============================================================================
// Kahn ordering (from fuzz/fuzz_targets/kahn.rs)
// ============================================================================

/// Check that a Kahn topological ordering is valid: for every edge (u, v),
/// the position of u must be <= the position of v. Also verifies that
/// the ordering can produce a valid upper triangular matrix.
///
/// Does nothing if the matrix has more than `max_size` rows (to avoid slow
/// tests) or if the matrix contains a cycle.
///
/// # Panics
///
/// Panics if the ordering violates the topological invariant.
pub fn check_kahn_ordering(
    matrix: &SquareCSR2D<CSR2D<u16, u8, u8>>,
    max_size: usize,
) {
    if matrix.number_of_rows().into_usize() > max_size
        || matrix.number_of_columns().into_usize() > max_size
    {
        return;
    }

    let Ok(ordering) = matrix.kahn() else {
        return;
    };

    matrix.row_indices().for_each(|row_id| {
        let resorted_row_id = ordering[row_id.into_usize()];
        matrix.sparse_row(row_id).for_each(|successor_id| {
            let resorted_successor_id = ordering[successor_id.into_usize()];
            assert!(
                resorted_row_id <= resorted_successor_id,
                "The ordering {ordering:?} is not valid: {resorted_row_id} ({row_id}) > \
                 {resorted_successor_id} ({successor_id})",
            );
        });
    });

    // If the ordering is valid, it must be possible to construct an
    // upper triangular matrix from the ordering.
    let mut coordinates: Vec<(u8, u8)> = SparseMatrix::sparse_coordinates(matrix)
        .map(|(i, j)| (ordering[i.into_usize()], ordering[j.into_usize()]))
        .collect();
    coordinates.sort_unstable();

    let _triangular: UpperTriangularCSR2D<CSR2D<u16, u8, u8>> =
        UpperTriangularCSR2D::from_entries(coordinates)
            .expect("The ordering should be valid");
}

// ============================================================================
// Similarity invariants (from fuzz/fuzz_targets/wu_palmer.rs, lin.rs)
// ============================================================================

/// Check that a [`ScalarSimilarity`] implementation satisfies basic
/// invariants: self-similarity > 0.99, symmetry, and bounds [0, 1].
///
/// Only the first `max_outer` source nodes are tested (all destinations
/// are always tested).
///
/// # Panics
///
/// Panics if any invariant is violated.
pub fn check_similarity_invariants<S, N>(
    similarity: &S,
    node_ids: &[N],
    max_outer: usize,
) where
    S: ScalarSimilarity<N, N, Similarity = f64>,
    N: Copy + Eq + Debug,
{
    for &src in node_ids.iter().take(max_outer) {
        for &dst in node_ids {
            let sim = similarity.similarity(&src, &dst);
            if src == dst {
                assert!(
                    sim > 0.99,
                    "Expected self-similarity of {src:?} > 0.99, got {sim}"
                );
            } else {
                let symmetric_similarity = similarity.similarity(&dst, &src);
                assert!(
                    (symmetric_similarity - sim).abs() < f64::EPSILON,
                    "Expected sim({src:?}, {dst:?}) == sim({dst:?}, {src:?}) got \
                     {sim}!={symmetric_similarity}"
                );
            }
            assert!(
                sim <= 1.0,
                "Expected sim({src:?},{dst:?}) = {sim} <= 1"
            );
            assert!(
                sim >= 0.0,
                "Expected sim({src:?},{dst:?}) = {sim} >= 0"
            );
        }
    }
}

// ============================================================================
// LAP assignment validation (from fuzz/fuzz_targets/lap.rs)
// ============================================================================

/// Validate that a LAP assignment is valid: no duplicate rows or columns,
/// all edges exist, and indices are within bounds.
///
/// # Panics
///
/// Panics if the assignment is invalid.
pub fn validate_lap_assignment(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
    assignment: &[(u8, u8)],
    label: &str,
) {
    let number_of_rows = csr.number_of_rows().into_usize();
    let number_of_columns = csr.number_of_columns().into_usize();
    let mut seen_rows = vec![false; number_of_rows];
    let mut seen_columns = vec![false; number_of_columns];

    for &(row, column) in assignment {
        let row_index = row.into_usize();
        let column_index = column.into_usize();

        assert!(
            row_index < number_of_rows,
            "{label}: row index out of bounds ({row_index} >= {number_of_rows})"
        );
        assert!(
            column_index < number_of_columns,
            "{label}: column index out of bounds ({column_index} >= {number_of_columns})"
        );
        assert!(
            csr.has_entry(row, column),
            "{label}: assignment includes non-existing edge ({row}, {column})"
        );
        assert!(
            !seen_rows[row_index],
            "{label}: duplicate row in assignment ({row})"
        );
        assert!(
            !seen_columns[column_index],
            "{label}: duplicate column in assignment ({column})"
        );

        seen_rows[row_index] = true;
        seen_columns[column_index] = true;
    }
}

/// Returns `true` when edge weights span a numerically stable range,
/// avoiding extreme floating-point regimes.
pub fn lap_values_are_numerically_stable(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) -> bool {
    let mut minimum_value = f64::INFINITY;
    let mut maximum_value = 0.0_f64;

    for row in csr.row_indices() {
        for value in csr.sparse_row_values(row) {
            if value > 0.0 && value.is_finite() {
                minimum_value = minimum_value.min(value);
                maximum_value = maximum_value.max(value);
            }
        }
    }

    if minimum_value.is_infinite() {
        return true;
    }

    minimum_value >= f64::MIN_POSITIVE
        && maximum_value <= 1e150
        && (maximum_value / minimum_value) <= 1e12
}

/// Compute the total cost of an assignment.
pub fn lap_assignment_cost(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
    assignment: &[(u8, u8)],
) -> f64 {
    assignment
        .iter()
        .map(|&(row, column)| {
            csr.sparse_row(row)
                .zip(csr.sparse_row_values(row))
                .find_map(|(c, v)| (c == column).then_some(v))
                .unwrap_or_else(|| {
                    panic!("Assignment includes non-existing edge ({row}, {column})")
                })
        })
        .sum()
}

/// Check full LAP sparse-wrapper invariants: both `sparse_lapmod` and
/// `sparse_lapjv` should agree on results when the weight range is
/// numerically stable.
///
/// # Panics
///
/// Panics if the wrappers disagree when they should agree.
pub fn check_lap_sparse_wrapper_invariants(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) {
    let numerically_stable = lap_values_are_numerically_stable(csr);
    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    let padding_value = (maximum_value + 1.0) * 2.0;
    let maximal_cost = (padding_value + 1.0) * 2.0;

    if !padding_value.is_finite()
        || !maximal_cost.is_finite()
        || padding_value <= 0.0
        || maximal_cost <= padding_value
    {
        return;
    }

    let sparse_lapmod = csr.sparse_lapmod(padding_value, maximal_cost);
    let sparse_lapjv = csr.sparse_lapjv(padding_value, maximal_cost);

    match (&sparse_lapmod, &sparse_lapjv) {
        (Ok(lapmod_assignment), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, lapmod_assignment, "SparseLAPMOD");
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");

            if numerically_stable {
                assert_eq!(
                    lapmod_assignment.len(),
                    lapjv_assignment.len(),
                    "SparseLAPMOD/SparseLAPJV cardinality mismatch: {csr:?}"
                );

                let lapmod_cost = lap_assignment_cost(csr, lapmod_assignment);
                let lapjv_cost = lap_assignment_cost(csr, lapjv_assignment);
                assert!(
                    (lapmod_cost - lapjv_cost).abs() < 1e-9,
                    "SparseLAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs \
                     {lapjv_cost}): {csr:?}"
                );
                if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
                    assert_eq!(
                        lapmod_assignment.len(),
                        hopcroft_karp_assignment.len(),
                        "SparseLAPMOD/Hopcroft-Karp cardinality mismatch: {csr:?}"
                    );
                }
            }
        }
        (Err(lapmod_error), Err(lapjv_error)) => {
            if numerically_stable {
                assert_eq!(
                    lapmod_error, lapjv_error,
                    "Sparse wrapper error mismatch: SparseLAPMOD={lapmod_error:?} \
                     SparseLAPJV={lapjv_error:?} matrix={csr:?}"
                );
            }
        }
        (Ok(lapmod_assignment), Err(lapjv_error)) => {
            validate_lap_assignment(csr, lapmod_assignment, "SparseLAPMOD");
            if numerically_stable {
                panic!(
                    "Sparse wrapper mismatch: SparseLAPMOD returned assignment of len {} \
                     but SparseLAPJV failed with {lapjv_error:?}: {csr:?}",
                    lapmod_assignment.len(),
                );
            }
        }
        (Err(lapmod_error), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");
            if numerically_stable {
                panic!(
                    "Sparse wrapper mismatch: SparseLAPMOD failed with {lapmod_error:?} \
                     but SparseLAPJV returned assignment of len {}: {csr:?}",
                    lapjv_assignment.len(),
                );
            }
        }
    }
}

/// Check full LAP invariants on square matrices: core `lapmod` should agree
/// with the sparse wrappers.
///
/// # Panics
///
/// Panics if results are inconsistent.
pub fn check_lap_square_invariants(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) {
    if csr.number_of_rows().into_usize() != csr.number_of_columns().into_usize() {
        return;
    }

    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    let max_cost = (maximum_value + 1.0) * 2.0;
    if !max_cost.is_finite() || max_cost <= 0.0 {
        return;
    }

    let Ok(lapmod_assignment) = csr.lapmod(max_cost) else {
        return;
    };
    validate_lap_assignment(csr, &lapmod_assignment, "LAPMOD");

    let numerically_stable = lap_values_are_numerically_stable(csr);
    if numerically_stable {
        if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
            assert_eq!(
                lapmod_assignment.len(),
                hopcroft_karp_assignment.len(),
                "LAPMOD/Hopcroft-Karp cardinality mismatch: {csr:?}"
            );
        }
    }

    let padding_value = (maximum_value + 1.0) * 4.0;
    let maximal_cost = (padding_value + 1.0) * 2.0;
    if !padding_value.is_finite()
        || !maximal_cost.is_finite()
        || maximal_cost <= padding_value
    {
        return;
    }

    let sparse_lapmod_assignment = csr.sparse_lapmod(padding_value, maximal_cost);
    let sparse_lapjv_assignment = csr.sparse_lapjv(padding_value, maximal_cost);

    if !numerically_stable {
        if let Ok(assignment) = &sparse_lapmod_assignment {
            validate_lap_assignment(csr, assignment, "SparseLAPMOD");
        }
        if let Ok(assignment) = &sparse_lapjv_assignment {
            validate_lap_assignment(csr, assignment, "SparseLAPJV");
        }
        return;
    }

    let sparse_lapmod_assignment = sparse_lapmod_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPMOD failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });
    let sparse_lapjv_assignment = sparse_lapjv_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPJV failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });

    validate_lap_assignment(csr, &sparse_lapmod_assignment, "SparseLAPMOD");
    validate_lap_assignment(csr, &sparse_lapjv_assignment, "SparseLAPJV");

    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapmod_assignment.len(),
        "LAPMOD/SparseLAPMOD cardinality mismatch: {csr:?}"
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapjv_assignment.len(),
        "LAPMOD/SparseLAPJV cardinality mismatch: {csr:?}"
    );

    let lapmod_cost = lap_assignment_cost(csr, &lapmod_assignment);
    let sparse_lapmod_cost = lap_assignment_cost(csr, &sparse_lapmod_assignment);
    let sparse_lapjv_cost = lap_assignment_cost(csr, &sparse_lapjv_assignment);

    assert!(
        (lapmod_cost - sparse_lapmod_cost).abs() < 1e-9,
        "LAPMOD/SparseLAPMOD objective mismatch ({lapmod_cost} vs \
         {sparse_lapmod_cost}): {csr:?}",
    );
    assert!(
        (lapmod_cost - sparse_lapjv_cost).abs() < 1e-9,
        "LAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs \
         {sparse_lapjv_cost}): {csr:?}",
    );
}

// ============================================================================
// Louvain invariants (from fuzz/fuzz_targets/louvain.rs)
// ============================================================================

/// Returns `true` when edge weights are in a numerically stable range for
/// Louvain modularity comparisons.
pub fn louvain_weights_are_numerically_stable(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) -> bool {
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0_f64;

    for row in csr.row_indices() {
        for val in csr.sparse_row_values(row) {
            if val > 0.0 && val.is_finite() && val.is_normal() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    if min_val.is_infinite() {
        return true;
    }

    min_val >= f64::MIN_POSITIVE && max_val <= 1e150 && (max_val / min_val) <= 1e12
}

/// Check Louvain invariants on arbitrary input (should never panic) and,
/// when possible, on a symmetrized version of the matrix (partition length,
/// modularity bounds, determinism).
///
/// # Panics
///
/// Panics if Louvain fails on valid symmetric input or produces invalid
/// results.
pub fn check_louvain_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    // Louvain must never panic on arbitrary input.
    let _: Result<LouvainResult<usize>, _> = csr.louvain(&LouvainConfig::default());

    // Skip symmetric invariant checking for extreme weight ranges.
    if !louvain_weights_are_numerically_stable(csr) {
        return;
    }

    let rows = csr.number_of_rows().into_usize();
    let cols = csr.number_of_columns().into_usize();
    if rows != cols || rows == 0 || rows > u8::MAX as usize {
        return;
    }

    let n = rows as u8;

    // Extract upper-triangle edges with finite positive weights, then mirror.
    let mut edges: Vec<(u8, u8, f64)> = Vec::new();
    for row in csr.row_indices() {
        let r = row.into_usize();
        if r >= rows {
            continue;
        }
        for (col, val) in csr.sparse_row(row).zip(csr.sparse_row_values(row)) {
            let c = col.into_usize();
            if r <= c && val.is_finite() && val.is_normal() && val > 0.0 {
                let r8 = r as u8;
                let c8 = c as u8;
                edges.push((r8, c8, val));
                if r8 != c8 {
                    edges.push((c8, r8, val));
                }
            }
        }
    }

    if edges.is_empty() {
        return;
    }

    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    edges.dedup_by(|(r1, c1, _), (r2, c2, _)| (*r1, *c1) == (*r2, *c2));

    let Ok(edge_count) = u8::try_from(edges.len()) else {
        return;
    };

    let sym_csr: ValuedCSR2D<u8, u8, u8, f64> =
        match GenericEdgesBuilder::default()
            .expected_number_of_edges(edge_count)
            .expected_shape((n, n))
            .edges(edges.into_iter())
            .build()
        {
            Ok(m) => m,
            Err(_) => return,
        };

    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&sym_csr, &config)
        .expect("Louvain must not fail on a valid symmetric graph");

    let n = n as usize;
    assert_eq!(
        result.final_partition().len(),
        n,
        "partition length must equal node count"
    );
    let modularity = result.final_modularity();
    assert!(
        modularity >= -0.5 - 1e-9 && modularity <= 1.0 + 1e-9,
        "modularity {modularity} out of [-0.5, 1.0] (with FP tolerance)"
    );

    // Determinism check.
    let result2 = Louvain::<usize>::louvain(&sym_csr, &config).unwrap();
    assert_eq!(
        result.final_partition(),
        result2.final_partition(),
        "Louvain must be deterministic for the same seed"
    );
    assert!(
        (result.final_modularity() - result2.final_modularity()).abs() <= 1.0e-12,
        "modularity must be deterministic"
    );
}
