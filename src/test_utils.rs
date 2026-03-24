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
use bitvec::vec::BitVec;
use num_traits::AsPrimitive;

use crate::{
    prelude::*,
    traits::{
        DenseValuedMatrix2D, EdgesBuilder, SparseMatrix, SparseMatrix2D, SparseSquareMatrix,
        SparseValuedMatrix, SparseValuedMatrix2D,
    },
};

// ============================================================================
// Deserialization helpers
// ============================================================================

/// Construct a value of type `T` from raw bytes using the [`Arbitrary`] trait.
///
/// Returns `None` if the bytes are insufficient or do not produce a valid
/// instance.
#[must_use]
#[inline]
pub fn from_bytes<T: for<'a> Arbitrary<'a>>(bytes: &[u8]) -> Option<T> {
    let mut u = Unstructured::new(bytes);
    T::arbitrary(&mut u).ok()
}

/// Load all files from a directory and construct instances of `T` from each
/// file's raw bytes.
///
/// Files that fail to produce valid instances are silently skipped.
/// Returns an empty vector if the directory does not exist or is unreadable.
#[must_use]
#[inline]
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
#[inline]
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
        assert_eq!(column_indices, sorted_column_indices, "The row {row_index:?} is not sorted");
        sorted_column_indices.dedup();
        assert_eq!(column_indices, sorted_column_indices, "The row {row_index:?} has duplicates");
    }

    let sparse_coordinates: Vec<M::Coordinates> = SparseMatrix::sparse_coordinates(csr).collect();
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
#[inline]
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
#[inline]
pub fn check_padded_diagonal_invariants(padded_csr: &FuzzPaddedDiag) {
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
            underlying_matrix.sparse_row(row_index).any(|column_index| column_index == row_index)
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
#[inline]
pub fn check_padded_matrix2d_invariants(csr: &ValuedCSR2D<u16, u8, u8, u8>) {
    let Ok(padded_matrix) = PaddedMatrix2D::new(csr, |_| 1) else {
        return;
    };
    let padded_number_of_rows = padded_matrix.number_of_rows();
    let padded_number_of_columns = padded_matrix.number_of_columns();
    let csr_number_of_rows = csr.number_of_rows();
    let csr_number_of_columns = csr.number_of_columns();
    let mut last_tuple = None;

    for row_index in csr.row_indices() {
        let csr_column_values: Vec<(u8, u8)> =
            csr.sparse_row(row_index).zip(csr.sparse_row_values(row_index)).collect();
        let padded_column_values: Vec<(u8, u8)> =
            padded_matrix.column_indices().zip(padded_matrix.row_values(row_index)).collect();

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
// Gabow 1976 invariants (from fuzz/fuzz_targets/gabow_1976.rs)
// ============================================================================

/// Check structural validity and exact-size agreement for Gabow's 1976
/// maximum matching implementation against the existing blossom solver.
///
/// # Panics
///
/// Panics if Gabow 1976 returns an invalid matching, violates maximality, or
/// disagrees with `blossom()` on matching size.
#[inline]
pub fn check_gabow_1976_invariants<M>(csr: &M)
where
    M: SparseSquareMatrix + Blossom + Gabow1976,
    M::Index: Debug,
{
    let n = csr.order().as_();
    let gabow_matching = csr.gabow_1976();
    let blossom_matching = csr.blossom();

    assert_eq!(
        gabow_matching.len(),
        blossom_matching.len(),
        "Gabow1976 and Blossom disagree on matching size (n={n})"
    );
    assert!(gabow_matching.len() <= n / 2);

    let mut matched = vec![false; n];
    for &(u, v) in &gabow_matching {
        let ui = u.as_();
        let vi = v.as_();
        assert!(u < v);
        assert!(!matched[ui], "vertex {u:?} matched twice");
        assert!(!matched[vi], "vertex {v:?} matched twice");
        matched[ui] = true;
        matched[vi] = true;
        assert!(csr.has_entry(u, v));
    }

    for u in csr.row_indices() {
        if matched[u.as_()] {
            continue;
        }
        for w in csr.sparse_row(u) {
            assert!(w == u || matched[w.as_()], "edge ({u:?}, {w:?}) has both endpoints unmatched");
        }
    }
}

// ============================================================================
// Karp-Sipser matching invariants
// ============================================================================

fn check_matching_valid<M>(graph: &M, matching: &[(M::Index, M::Index)])
where
    M: SparseSquareMatrix,
    M::Index: AsPrimitive<usize> + Ord + Copy + Debug,
{
    let mut matched = vec![false; graph.order().as_()];
    for &(left, right) in matching {
        assert!(left < right, "matching pair must satisfy u < v");
        assert!(graph.has_entry(left, right), "matching contains a non-edge");

        let left_index = left.as_();
        let right_index = right.as_();
        assert!(!matched[left_index], "left endpoint is reused");
        assert!(!matched[right_index], "right endpoint is reused");
        matched[left_index] = true;
        matched[right_index] = true;
    }
}

fn assert_karp_sipser_kernel_irreducible<M>(graph: &M, rules: KarpSipserRules)
where
    M: SparseSquareMatrix,
    M::Index: AsPrimitive<usize> + Copy + Debug,
{
    for row in graph.row_indices() {
        let row_index = row.as_();
        let degree = graph.sparse_row(row).filter(|&column| column.as_() != row_index).count();

        match rules {
            KarpSipserRules::Degree1 => {
                assert_ne!(degree, 1, "degree-1 kernel still contains a degree-1 vertex");
            }
            KarpSipserRules::Degree1And2 => {
                assert!(
                    degree == 0 || degree >= 3,
                    "degree-1/2 kernel still contains a reducible vertex of degree {degree}",
                );
            }
        }
    }
}

/// Check that exact Karp-Sipser preprocessing preserves matching cardinality
/// and produces valid recovered matchings for all exact wrapper variants.
///
/// # Panics
///
/// Panics if any Karp-Sipser wrapper returns an invalid matching or if it
/// disagrees in size with the baseline blossom result.
#[inline]
pub fn check_karp_sipser_invariants<M>(graph: &M)
where
    M: SparseSquareMatrix + Blossom + Blum + KarpSipser + MicaliVazirani,
    M::Index: AsPrimitive<usize> + Ord + Copy + Debug,
{
    let blossom_matching = graph.blossom();
    check_matching_valid(graph, &blossom_matching);
    let expected_size = blossom_matching.len();

    let plain_blum_matching = graph.blum();
    check_matching_valid(graph, &plain_blum_matching);
    assert_eq!(
        plain_blum_matching.len(),
        expected_size,
        "plain Blum disagrees with Blossom before Karp-Sipser is applied",
    );

    for rules in [KarpSipserRules::Degree1, KarpSipserRules::Degree1And2] {
        let kernel = graph.karp_sipser_kernel(rules);
        assert_karp_sipser_kernel_irreducible(kernel.graph(), rules);

        let recovered_blossom = kernel.solve_with(Blossom::blossom);
        check_matching_valid(graph, &recovered_blossom);
        assert_eq!(
            recovered_blossom.len(),
            expected_size,
            "Karp-Sipser blossom wrapper changed the matching size",
        );

        let explicit_recover = {
            let kernel = graph.karp_sipser_kernel(rules);
            let kernel_matching = kernel.graph().blossom();
            kernel.recover(kernel_matching)
        };
        check_matching_valid(graph, &explicit_recover);
        assert_eq!(
            explicit_recover.len(),
            expected_size,
            "explicit kernel recover changed the matching size",
        );

        let mv_matching = graph.micali_vazirani_with_karp_sipser(rules);
        check_matching_valid(graph, &mv_matching);
        assert_eq!(
            mv_matching.len(),
            expected_size,
            "Karp-Sipser Micali-Vazirani wrapper changed the matching size",
        );

        let blum_matching = graph.blum_with_karp_sipser(rules);
        check_matching_valid(graph, &blum_matching);
        assert_eq!(
            blum_matching.len(),
            expected_size,
            "Karp-Sipser Blum wrapper changed the matching size",
        );
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
#[inline]
pub fn check_kahn_ordering(matrix: &SquareCSR2D<CSR2D<u16, u8, u8>>, max_size: usize) {
    let number_of_rows: usize = matrix.number_of_rows().as_();
    let number_of_columns: usize = matrix.number_of_columns().as_();
    if number_of_rows > max_size || number_of_columns > max_size {
        return;
    }

    let Ok(ordering) = matrix.kahn() else {
        return;
    };

    matrix.row_indices().for_each(|row_id| {
        let row_index = usize::from(row_id);
        let resorted_row_id = ordering[row_index];
        matrix.sparse_row(row_id).for_each(|successor_id| {
            let successor_index = usize::from(successor_id);
            let resorted_successor_id = ordering[successor_index];
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
        .map(|(i, j)| (ordering[usize::from(i)], ordering[usize::from(j)]))
        .collect();
    coordinates.sort_unstable();

    let _triangular: UpperTriangularCSR2D<CSR2D<u16, u8, u8>> =
        UpperTriangularCSR2D::from_entries(coordinates).expect("The ordering should be valid");
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
#[inline]
pub fn check_similarity_invariants<S, N>(similarity: &S, node_ids: &[N], max_outer: usize)
where
    S: ScalarSimilarity<N, N, Similarity = f64>,
    N: Copy + Eq + Debug,
{
    for &src in node_ids.iter().take(max_outer) {
        for &dst in node_ids {
            let sim = similarity.similarity(&src, &dst);
            if src == dst {
                assert!(sim > 0.99, "Expected self-similarity of {src:?} > 0.99, got {sim}");
            } else {
                let symmetric_similarity = similarity.similarity(&dst, &src);
                assert!(
                    (symmetric_similarity - sim).abs() < f64::EPSILON,
                    "Expected sim({src:?}, {dst:?}) == sim({dst:?}, {src:?}) got \
                     {sim}!={symmetric_similarity}"
                );
            }
            assert!(sim <= 1.0, "Expected sim({src:?},{dst:?}) = {sim} <= 1");
            assert!(sim >= 0.0, "Expected sim({src:?},{dst:?}) = {sim} >= 0");
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
#[inline]
pub fn validate_lap_assignment(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
    assignment: &[(u8, u8)],
    label: &str,
) {
    let number_of_rows: usize = csr.number_of_rows().as_();
    let number_of_columns: usize = csr.number_of_columns().as_();
    let mut seen_rows = vec![false; number_of_rows];
    let mut seen_columns = vec![false; number_of_columns];

    for &(row, column) in assignment {
        let row_index: usize = row.as_();
        let column_index: usize = column.as_();

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
        assert!(!seen_rows[row_index], "{label}: duplicate row in assignment ({row})");
        assert!(!seen_columns[column_index], "{label}: duplicate column in assignment ({column})");

        seen_rows[row_index] = true;
        seen_columns[column_index] = true;
    }
}

/// Returns `true` when edge weights span a numerically stable range,
/// avoiding extreme floating-point regimes.
#[must_use]
#[inline]
pub fn lap_values_are_numerically_stable(csr: &ValuedCSR2D<u16, u8, u8, f64>) -> bool {
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
#[must_use]
#[inline]
pub fn lap_assignment_cost(csr: &ValuedCSR2D<u16, u8, u8, f64>, assignment: &[(u8, u8)]) -> f64 {
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

/// Check full LAP sparse-wrapper invariants: both `jaqaman` and
/// `sparse_lapjv` should agree on results when the weight range is
/// numerically stable.
///
/// # Panics
///
/// Panics if the wrappers disagree when they should agree.
#[inline]
pub fn check_lap_sparse_wrapper_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let numerically_stable = lap_values_are_numerically_stable(csr);
    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    // Use multiplicative scaling so that η/2 = 1.05 × max > max at any
    // magnitude.  The old additive formula (max + 1.0) * 2.0 failed when
    // max + 1.0 == max in floating point (values above ~1e16).
    let padding_value = maximum_value * 2.1;
    let maximal_cost = padding_value * 2.0;

    if !padding_value.is_finite()
        || !maximal_cost.is_finite()
        || padding_value <= 0.0
        || maximal_cost <= padding_value
    {
        return;
    }

    let jaqaman_result = csr.jaqaman(padding_value, maximal_cost);
    let sparse_lapjv = csr.sparse_lapjv(padding_value, maximal_cost);
    let sparse_hungarian = csr.sparse_hungarian(padding_value, maximal_cost);

    match (&jaqaman_result, &sparse_lapjv) {
        (Ok(jaqaman_assignment), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, jaqaman_assignment, "Jaqaman");
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");

            if numerically_stable {
                assert_eq!(
                    jaqaman_assignment.len(),
                    lapjv_assignment.len(),
                    "Jaqaman/SparseLAPJV cardinality mismatch: {csr:?}"
                );

                let jaqaman_cost = lap_assignment_cost(csr, jaqaman_assignment);
                let lapjv_cost = lap_assignment_cost(csr, lapjv_assignment);
                let denom = jaqaman_cost.abs().max(lapjv_cost.abs()).max(1e-30);
                assert!(
                    (jaqaman_cost - lapjv_cost).abs() / denom < 1e-9,
                    "Jaqaman/SparseLAPJV objective mismatch ({jaqaman_cost} vs \
                     {lapjv_cost}): {csr:?}"
                );
                if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
                    assert_eq!(
                        jaqaman_assignment.len(),
                        hopcroft_karp_assignment.len(),
                        "Jaqaman/Hopcroft-Karp cardinality mismatch: {csr:?}"
                    );
                }
            }
        }
        (Err(jaqaman_error), Err(lapjv_error)) => {
            if numerically_stable {
                assert_eq!(
                    jaqaman_error, lapjv_error,
                    "Sparse wrapper error mismatch: Jaqaman={jaqaman_error:?} \
                     SparseLAPJV={lapjv_error:?} matrix={csr:?}"
                );
            }
        }
        (Ok(jaqaman_assignment), Err(lapjv_error)) => {
            validate_lap_assignment(csr, jaqaman_assignment, "Jaqaman");
            assert!(
                !numerically_stable,
                "Sparse wrapper mismatch: Jaqaman returned assignment of len {} \
                 but SparseLAPJV failed with {lapjv_error:?}: {csr:?}",
                jaqaman_assignment.len(),
            );
        }
        (Err(jaqaman_error), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");
            assert!(
                !numerically_stable,
                "Sparse wrapper mismatch: Jaqaman failed with {jaqaman_error:?} \
                 but SparseLAPJV returned assignment of len {}: {csr:?}",
                lapjv_assignment.len(),
            );
        }
    }

    // Cross-validate SparseHungarian against the other solvers.
    if let Ok(hungarian_assignment) = &sparse_hungarian {
        validate_lap_assignment(csr, hungarian_assignment, "SparseHungarian");

        if numerically_stable {
            if let Ok(jaqaman_assignment) = &jaqaman_result {
                assert_eq!(
                    hungarian_assignment.len(),
                    jaqaman_assignment.len(),
                    "SparseHungarian/Jaqaman cardinality mismatch: {csr:?}"
                );

                let hungarian_cost = lap_assignment_cost(csr, hungarian_assignment);
                let jaqaman_cost = lap_assignment_cost(csr, jaqaman_assignment);
                let denom = hungarian_cost.abs().max(jaqaman_cost.abs()).max(1e-30);
                assert!(
                    (hungarian_cost - jaqaman_cost).abs() / denom < 1e-9,
                    "SparseHungarian/Jaqaman objective mismatch ({hungarian_cost} vs \
                     {jaqaman_cost}): {csr:?}"
                );
            }
        }
    } else if numerically_stable && jaqaman_result.is_ok() {
        panic!(
            "SparseHungarian failed but Jaqaman succeeded on numerically stable matrix: {csr:?}"
        );
    }
}

/// Check full LAP invariants on square matrices: core `lapmod` should agree
/// with the sparse wrappers.
///
/// # Panics
///
/// Panics if results are inconsistent.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_lap_square_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let number_of_rows: usize = csr.number_of_rows().as_();
    let number_of_columns: usize = csr.number_of_columns().as_();
    if number_of_rows != number_of_columns {
        return;
    }

    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    let max_cost = maximum_value * 2.1;
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

    let padding_value = maximum_value * 4.2;
    let maximal_cost = padding_value * 2.0;
    if !padding_value.is_finite() || !maximal_cost.is_finite() || maximal_cost <= padding_value {
        return;
    }

    let jaqaman_assignment = csr.jaqaman(padding_value, maximal_cost);
    let sparse_lapjv_assignment = csr.sparse_lapjv(padding_value, maximal_cost);
    let sparse_hungarian_assignment = csr.sparse_hungarian(padding_value, maximal_cost);

    if !numerically_stable {
        if let Ok(assignment) = &jaqaman_assignment {
            validate_lap_assignment(csr, assignment, "Jaqaman");
        }
        if let Ok(assignment) = &sparse_lapjv_assignment {
            validate_lap_assignment(csr, assignment, "SparseLAPJV");
        }
        if let Ok(assignment) = &sparse_hungarian_assignment {
            validate_lap_assignment(csr, assignment, "SparseHungarian");
        }
        return;
    }

    let jaqaman_assignment = jaqaman_assignment.unwrap_or_else(|error| {
        panic!(
            "Jaqaman failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });
    let sparse_lapjv_assignment = sparse_lapjv_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPJV failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });
    let sparse_hungarian_assignment = sparse_hungarian_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseHungarian failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });

    validate_lap_assignment(csr, &jaqaman_assignment, "Jaqaman");
    validate_lap_assignment(csr, &sparse_lapjv_assignment, "SparseLAPJV");
    validate_lap_assignment(csr, &sparse_hungarian_assignment, "SparseHungarian");

    assert_eq!(
        lapmod_assignment.len(),
        jaqaman_assignment.len(),
        "LAPMOD/Jaqaman cardinality mismatch: {csr:?}"
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapjv_assignment.len(),
        "LAPMOD/SparseLAPJV cardinality mismatch: {csr:?}"
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_hungarian_assignment.len(),
        "LAPMOD/SparseHungarian cardinality mismatch: {csr:?}"
    );

    let lapmod_cost = lap_assignment_cost(csr, &lapmod_assignment);
    let jaqaman_cost = lap_assignment_cost(csr, &jaqaman_assignment);
    let sparse_lapjv_cost = lap_assignment_cost(csr, &sparse_lapjv_assignment);
    let sparse_hungarian_cost = lap_assignment_cost(csr, &sparse_hungarian_assignment);

    let denom1 = lapmod_cost.abs().max(jaqaman_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - jaqaman_cost).abs() / denom1 < 1e-9,
        "LAPMOD/Jaqaman objective mismatch ({lapmod_cost} vs \
         {jaqaman_cost}): {csr:?}",
    );
    let denom2 = lapmod_cost.abs().max(sparse_lapjv_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - sparse_lapjv_cost).abs() / denom2 < 1e-9,
        "LAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs \
         {sparse_lapjv_cost}): {csr:?}",
    );
    let denom3 = lapmod_cost.abs().max(sparse_hungarian_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - sparse_hungarian_cost).abs() / denom3 < 1e-9,
        "LAPMOD/SparseHungarian objective mismatch ({lapmod_cost} vs \
         {sparse_hungarian_cost}): {csr:?}",
    );
}

// ============================================================================
// Louvain/Leiden invariants (from fuzz/fuzz_targets/{louvain,leiden}.rs)
// ============================================================================

/// Returns `true` when edge weights are in a numerically stable range for
/// modularity comparisons.
#[must_use]
#[inline]
pub fn louvain_weights_are_numerically_stable(csr: &ValuedCSR2D<u16, u8, u8, f64>) -> bool {
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

fn symmetrized_positive_graph(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) -> Option<ValuedCSR2D<u8, u8, u8, f64>> {
    let rows: usize = csr.number_of_rows().as_();
    let cols: usize = csr.number_of_columns().as_();
    if rows != cols || rows == 0 || rows > u8::MAX as usize {
        return None;
    }

    let Ok(n) = u8::try_from(rows) else {
        return None;
    };

    // Extract upper-triangle edges with finite positive weights, then mirror.
    let mut edges: Vec<(u8, u8, f64)> = Vec::new();
    for row in csr.row_indices() {
        let r: usize = row.as_();
        if r >= rows {
            continue;
        }
        for (col, val) in csr.sparse_row(row).zip(csr.sparse_row_values(row)) {
            let c: usize = col.as_();
            if r <= c && val.is_finite() && val.is_normal() && val > 0.0 {
                let Ok(r8) = u8::try_from(r) else {
                    continue;
                };
                let Ok(c8) = u8::try_from(c) else {
                    continue;
                };
                edges.push((r8, c8, val));
                if r8 != c8 {
                    edges.push((c8, r8, val));
                }
            }
        }
    }

    if edges.is_empty() {
        return None;
    }

    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    edges.dedup_by(|(r1, c1, _), (r2, c2, _)| (*r1, *c1) == (*r2, *c2));

    let Ok(edge_count) = u8::try_from(edges.len()) else {
        return None;
    };

    GenericEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .ok()
}

fn partition_communities_are_connected(
    csr: &ValuedCSR2D<u8, u8, u8, f64>,
    partition: &[usize],
) -> bool {
    let node_count: usize = csr.number_of_rows().as_();
    if node_count == 0 || partition.len() != node_count {
        return false;
    }

    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    let mut nodes_per_community: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    for (node, community) in partition.iter().copied().enumerate() {
        nodes_per_community[community].push(node);
    }

    let mut queue: Vec<usize> = Vec::new();
    let mut is_member = vec![false; node_count];
    let mut visited = vec![false; node_count];

    for nodes in nodes_per_community {
        if nodes.len() <= 1 {
            continue;
        }

        for node in &nodes {
            is_member[*node] = true;
        }

        queue.clear();
        let start = nodes[0];
        queue.push(start);
        visited[start] = true;

        let mut visited_count = 0usize;
        while let Some(node) = queue.pop() {
            visited_count += 1;

            let Ok(row) = u8::try_from(node) else {
                return false;
            };
            for destination in csr.sparse_row(row) {
                let destination: usize = destination.as_();
                if destination < node_count && is_member[destination] && !visited[destination] {
                    visited[destination] = true;
                    queue.push(destination);
                }
            }
        }

        if visited_count != nodes.len() {
            return false;
        }

        for node in &nodes {
            is_member[*node] = false;
            visited[*node] = false;
        }
    }

    true
}

/// Check Louvain invariants on arbitrary input (should never panic) and,
/// when possible, on a symmetrized version of the matrix (partition length,
/// modularity bounds, determinism).
///
/// # Panics
///
/// Panics if Louvain fails on valid symmetric input or produces invalid
/// results.
#[inline]
pub fn check_louvain_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    // Louvain must never panic on arbitrary input.
    let _: Result<LouvainResult<usize>, _> = csr.louvain(&LouvainConfig::default());

    // Skip symmetric invariant checking for extreme weight ranges.
    if !louvain_weights_are_numerically_stable(csr) {
        return;
    }

    let Some(sym_csr) = symmetrized_positive_graph(csr) else {
        return;
    };

    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&sym_csr, &config)
        .expect("Louvain must not fail on a valid symmetric graph");

    let n: usize = sym_csr.number_of_rows().as_();
    assert_eq!(result.final_partition().len(), n, "partition length must equal node count");
    let modularity = result.final_modularity();
    assert!(
        (-0.5 - 1e-9..=1.0 + 1e-9).contains(&modularity),
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

/// Check Leiden invariants on arbitrary input (should never panic) and,
/// when possible, on a symmetrized version of the matrix (partition length,
/// modularity bounds, determinism, community connectedness).
///
/// # Panics
///
/// Panics if Leiden fails on valid symmetric input or produces invalid
/// results.
#[inline]
pub fn check_leiden_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    // Leiden must never panic on arbitrary input.
    let _: Result<LeidenResult<usize>, _> = csr.leiden(&LeidenConfig::default());

    // Skip symmetric invariant checking for extreme weight ranges.
    if !louvain_weights_are_numerically_stable(csr) {
        return;
    }

    let Some(sym_csr) = symmetrized_positive_graph(csr) else {
        return;
    };

    let config = LeidenConfig::default();
    let result = Leiden::<usize>::leiden(&sym_csr, &config)
        .expect("Leiden must not fail on a valid symmetric graph");

    let n: usize = sym_csr.number_of_rows().as_();
    let final_partition = result.final_partition();
    assert_eq!(final_partition.len(), n, "partition length must equal node count");
    let modularity = result.final_modularity();
    assert!(
        (-0.5 - 1e-9..=1.0 + 1e-9).contains(&modularity),
        "modularity {modularity} out of [-0.5, 1.0] (with FP tolerance)"
    );
    assert!(
        partition_communities_are_connected(&sym_csr, final_partition),
        "Leiden communities must induce connected subgraphs"
    );

    // Determinism check.
    let result2 = Leiden::<usize>::leiden(&sym_csr, &config).unwrap();
    assert_eq!(
        result.final_partition(),
        result2.final_partition(),
        "Leiden must be deterministic for the same seed"
    );
    assert!(
        (result.final_modularity() - result2.final_modularity()).abs() <= 1.0e-12,
        "modularity must be deterministic"
    );
}

// ============================================================================
// Jacobi eigenvalue decomposition invariants (from fuzz/fuzz_targets/jacobi.rs)
// ============================================================================

/// Check Jacobi eigenvalue decomposition invariants on arbitrary input.
///
/// Wraps the sparse CSR in a [`PaddedMatrix2D`] (padding with 0.0) and, when
/// the resulting matrix is symmetric, square, finite, and small enough
/// (n ≤ 32), verifies:
/// - eigenvalues are sorted descending and all finite
/// - eigenvectors are orthonormal (VᵀV ≈ I)
/// - reconstruction (A ≈ VΛVᵀ)
/// - determinism (same input → same output)
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_jacobi_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let Ok(padded) = PaddedMatrix2D::new(csr, |_| 0.0) else {
        return;
    };
    let rows: usize = padded.number_of_rows().as_();
    let cols: usize = padded.number_of_columns().as_();

    // Must not panic on any input.
    let result = padded.jacobi(&JacobiConfig::default());

    if rows != cols || rows == 0 || rows > 32 {
        return;
    }

    // Read dense values and check for finiteness / symmetry.
    let n = rows;
    let mut a_flat = Vec::with_capacity(n * n);
    let mut all_finite = true;
    for row_idx in padded.row_indices() {
        for val in padded.row_values(row_idx) {
            if !val.is_finite() {
                all_finite = false;
            }
            a_flat.push(val);
        }
    }
    if !all_finite {
        return;
    }

    let mut is_symmetric = true;
    for i in 0..n {
        for j in (i + 1)..n {
            let scale = a_flat[i * n + j].abs().max(a_flat[j * n + i].abs()).max(1.0);
            if (a_flat[i * n + j] - a_flat[j * n + i]).abs() > 16.0 * f64::EPSILON * scale {
                is_symmetric = false;
            }
        }
    }
    if !is_symmetric {
        return;
    }

    // Skip detailed numerical invariants for extreme value ranges.
    // Jacobi rotations square values internally; if max_abs ≈ 1e155,
    // then max_abs² ≈ 1e310 which is near f64::MAX ≈ 1.8e308.
    let max_abs = a_flat.iter().copied().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_abs > 1e150 {
        return;
    }

    let result = result.expect("Jacobi should succeed on a finite symmetric square matrix");

    // Eigenvalues sorted descending.
    for w in result.eigenvalues().windows(2) {
        assert!(w[0] >= w[1], "eigenvalues not sorted descending: {:?}", result.eigenvalues());
    }

    // All eigenvalues finite.
    for &ev in result.eigenvalues() {
        assert!(ev.is_finite(), "non-finite eigenvalue: {ev}");
    }

    // Orthonormality: VᵀV ≈ I.
    for k in 0..n {
        for l in 0..n {
            let dot: f64 =
                (0..n).map(|i| result.eigenvector(k)[i] * result.eigenvector(l)[i]).sum();
            let expected = if k == l { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-6, "VᵀV[{k},{l}] = {dot}, expected {expected}");
        }
    }

    // Reconstruction: A ≈ VΛVᵀ.
    for i in 0..n {
        for j in 0..n {
            let mut reconstructed = 0.0;
            for k in 0..n {
                reconstructed +=
                    result.eigenvalues()[k] * result.eigenvector(k)[i] * result.eigenvector(k)[j];
            }
            let expected = a_flat[i * n + j];
            let scale = expected.abs().max(1.0);
            assert!(
                (reconstructed - expected).abs() < 1e-6 * scale,
                "Reconstruction failed at ({i}, {j}): expected {expected}, got {reconstructed}"
            );
        }
    }

    // Determinism.
    let result2 = padded.jacobi(&JacobiConfig::default()).unwrap();
    assert_eq!(
        result.eigenvalues(),
        result2.eigenvalues(),
        "Jacobi must be deterministic for the same input"
    );
}

// ============================================================================
// Classical MDS invariants (from fuzz/fuzz_targets/mds.rs)
// ============================================================================

/// Check classical MDS invariants on arbitrary input.
///
/// Wraps the sparse CSR in a [`PaddedMatrix2D`] (padding with 0.0) and, when
/// the resulting matrix forms a valid distance matrix (square, finite,
/// non-negative, zero diagonal, symmetric, and small enough: n ≤ 32), verifies:
/// - coordinates are all finite
/// - eigenvalues are all finite and sorted descending
/// - stress is finite and non-negative
/// - determinism (same input → same output)
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_mds_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let Ok(padded) = PaddedMatrix2D::new(csr, |_| 0.0) else {
        return;
    };
    let rows: usize = padded.number_of_rows().as_();
    let cols: usize = padded.number_of_columns().as_();

    // Must not panic on any input.
    let config = MdsConfig::default();
    let result = padded.classical_mds(&config);

    if rows != cols || rows <= 1 || rows > 32 {
        return;
    }

    // Read dense values and check for valid distance matrix properties.
    let n = rows;
    let mut d_flat = Vec::with_capacity(n * n);
    let mut all_valid = true;
    for row_idx in padded.row_indices() {
        for val in padded.row_values(row_idx) {
            if !val.is_finite() || val < 0.0 {
                all_valid = false;
            }
            d_flat.push(val);
        }
    }
    if !all_valid {
        return;
    }

    // Check diagonal is zero.
    for i in 0..n {
        if d_flat[i * n + i] != 0.0 {
            return;
        }
    }

    // Check symmetry.
    for i in 0..n {
        for j in (i + 1)..n {
            let scale = d_flat[i * n + j].abs().max(d_flat[j * n + i].abs()).max(1.0);
            if (d_flat[i * n + j] - d_flat[j * n + i]).abs() > 16.0 * f64::EPSILON * scale {
                return;
            }
        }
    }

    // Skip detailed numerical invariants for extreme value ranges.
    let max_abs = d_flat.iter().copied().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_abs > 1e150 {
        return;
    }

    let result = result.expect("MDS should succeed on a valid distance matrix");

    // Coordinates are all finite.
    for &c in result.coordinates_flat() {
        assert!(c.is_finite(), "non-finite coordinate: {c}");
    }

    // Eigenvalues are finite.
    for &ev in result.eigenvalues() {
        assert!(ev.is_finite(), "non-finite eigenvalue: {ev}");
    }

    // Eigenvalues sorted descending.
    for w in result.eigenvalues().windows(2) {
        assert!(w[0] >= w[1], "eigenvalues not sorted descending: {:?}", result.eigenvalues());
    }

    // Stress is finite and non-negative.
    assert!(result.stress().is_finite(), "non-finite stress: {}", result.stress());
    assert!(result.stress() >= 0.0, "negative stress: {}", result.stress());

    // Determinism.
    let result2 = padded.classical_mds(&config).unwrap();
    assert_eq!(
        result.coordinates_flat(),
        result2.coordinates_flat(),
        "MDS must be deterministic for the same input"
    );
    assert_eq!(
        result.eigenvalues(),
        result2.eigenvalues(),
        "MDS eigenvalues must be deterministic"
    );
    assert!(
        (result.stress() - result2.stress()).abs() <= f64::EPSILON,
        "MDS stress must be deterministic"
    );
}

// ============================================================================
// Floyd-Warshall invariants (from fuzz/fuzz_targets/floyd_warshall.rs)
// ============================================================================

fn bellman_ford_all_pairs(
    order: usize,
    edges: &[(usize, usize, f64)],
) -> Result<Vec<Option<f64>>, usize> {
    let mut all_pairs = vec![None; order * order];

    for source_id in 0..order {
        let mut distances = vec![f64::INFINITY; order];
        distances[source_id] = 0.0;

        for _ in 0..order.saturating_sub(1) {
            let mut updated = false;
            for &(from, to, weight) in edges {
                if !distances[from].is_finite() {
                    continue;
                }
                let candidate = distances[from] + weight;
                if candidate < distances[to] {
                    distances[to] = candidate;
                    updated = true;
                }
            }
            if !updated {
                break;
            }
        }

        for &(from, to, weight) in edges {
            if distances[from].is_finite() && distances[from] + weight < distances[to] {
                return Err(to);
            }
        }

        for (destination_id, distance) in distances.into_iter().enumerate() {
            if distance.is_finite() {
                all_pairs[source_id * order + destination_id] = Some(distance);
            }
        }
    }

    Ok(all_pairs)
}

/// Check Floyd-Warshall invariants on arbitrary weighted sparse input.
///
/// For arbitrary matrices this helper verifies that the algorithm never
/// panics. For square matrices with finite weights, order ≤ 24, and a
/// moderate dynamic range, it cross-checks the result against a slower
/// Bellman-Ford reference implementation, validates zero diagonal and triangle
/// inequality, and checks determinism.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_floyd_warshall_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let result = csr.floyd_warshall();
    let rows = csr.number_of_rows().as_();
    let columns = csr.number_of_columns().as_();

    if rows != columns {
        assert!(
            matches!(
                result,
                Err(FloydWarshallError::NonSquareMatrix {
                    rows: found_rows,
                    columns: found_columns,
                }) if found_rows == rows && found_columns == columns
            ),
            "non-square matrix should return NonSquareMatrix, got {result:?}"
        );
        return;
    }

    if rows == 0 {
        let distances = result.expect("Floyd-Warshall should succeed on an empty matrix");
        assert_eq!(distances.shape(), vec![0, 0]);
        return;
    }

    let mut edges = Vec::new();
    let mut max_abs = 0.0_f64;
    for row_id in csr.row_indices() {
        let source_id = row_id.as_();
        for (column_id, weight) in csr.sparse_row(row_id).zip(csr.sparse_row_values(row_id)) {
            let destination_id = column_id.as_();
            if !weight.is_finite() {
                assert!(
                    matches!(
                        result,
                        Err(FloydWarshallError::NonFiniteWeight {
                            source_id: found_source,
                            destination_id: found_destination,
                        }) if found_source == source_id && found_destination == destination_id
                    ),
                    "non-finite input weight should return NonFiniteWeight, got {result:?}"
                );
                return;
            }
            max_abs = max_abs.max(weight.abs());
            edges.push((source_id, destination_id, weight));
        }
    }

    if rows > 24 || max_abs > 1e150 {
        return;
    }

    let reference = bellman_ford_all_pairs(rows, &edges);
    match reference {
        Err(_) => {
            assert!(
                matches!(result, Err(FloydWarshallError::NegativeCycle { .. })),
                "negative cycle should return NegativeCycle, got {result:?}"
            );
        }
        Ok(expected) => {
            let distances = result.expect(
                "Floyd-Warshall should succeed on finite square matrices without negative cycles",
            );

            assert_eq!(distances.shape(), vec![rows, rows]);
            for node_id in 0..rows {
                assert_eq!(
                    distances.value((node_id, node_id)),
                    Some(0.0),
                    "distance from a node to itself must be zero in absence of negative cycles"
                );
            }

            for source_id in 0..rows {
                for destination_id in 0..rows {
                    let actual = distances.value((source_id, destination_id));
                    let expected = expected[source_id * rows + destination_id];
                    match (actual, expected) {
                        (None, None) => {}
                        (Some(actual), Some(expected)) => {
                            let tolerance = expected.abs().max(1.0) * 1e-9;
                            assert!(
                                (actual - expected).abs() <= tolerance,
                                "distance mismatch at ({source_id}, {destination_id}): expected {expected}, got {actual}"
                            );
                        }
                        _ => {
                            panic!(
                                "reachability mismatch at ({source_id}, {destination_id}): expected {expected:?}, got {actual:?}"
                            );
                        }
                    }
                }
            }

            for source_id in 0..rows {
                for pivot_id in 0..rows {
                    let Some(source_to_pivot) = distances.value((source_id, pivot_id)) else {
                        continue;
                    };
                    for destination_id in 0..rows {
                        let Some(pivot_to_destination) =
                            distances.value((pivot_id, destination_id))
                        else {
                            continue;
                        };
                        let Some(source_to_destination) =
                            distances.value((source_id, destination_id))
                        else {
                            continue;
                        };
                        let bound = source_to_pivot + pivot_to_destination;
                        let tolerance =
                            bound.abs().max(source_to_destination.abs()).max(1.0) * 1e-9;
                        assert!(
                            source_to_destination <= bound + tolerance,
                            "triangle inequality violated for ({source_id}, {pivot_id}, {destination_id}): {source_to_destination} > {bound}"
                        );
                    }
                }
            }

            let distances2 =
                csr.floyd_warshall().expect("Floyd-Warshall should be deterministic on replay");
            assert_eq!(distances, distances2, "Floyd-Warshall must be deterministic");
        }
    }
}

/// Check PairwiseBFS invariants on arbitrary unweighted square sparse input.
///
/// This helper verifies that repeated BFS returns a square distance matrix with
/// zero diagonal, is deterministic, and matches Floyd-Warshall exactly when
/// the same graph is interpreted as having implicit unit weights.
///
/// Large matrices are still exercised for PairwiseBFS itself, but the
/// cross-check against Floyd-Warshall is capped to keep fuzzing throughput
/// reasonable.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
pub fn check_pairwise_bfs_matches_unit_floyd_warshall(csr: &SquareCSR2D<CSR2D<u16, u8, u8>>) {
    let distances = csr.pairwise_bfs();
    let order = csr.order().as_();

    assert_eq!(distances.shape(), vec![order, order]);
    for node_id in 0..order {
        assert_eq!(
            distances.value((node_id, node_id)),
            Some(0),
            "distance from a node to itself must be zero"
        );
    }

    let distances2 = csr.pairwise_bfs();
    assert_eq!(distances, distances2, "PairwiseBFS must be deterministic");

    if order > 64 {
        return;
    }

    let floyd_warshall = GenericImplicitValuedMatrix2D::new(csr.clone(), |_| 1usize)
        .floyd_warshall()
        .expect("unit-weight Floyd-Warshall should succeed on square unweighted matrices");
    assert_eq!(
        distances, floyd_warshall,
        "PairwiseBFS must match Floyd-Warshall with implicit unit weights"
    );
}

/// Check PairwiseDijkstra invariants on arbitrary weighted sparse input.
///
/// This helper verifies that repeated Dijkstra returns a square distance matrix
/// with zero diagonal, is deterministic, and matches Floyd-Warshall exactly on
/// finite square matrices with non-negative weights. Large matrices and
/// extreme magnitudes are still exercised for PairwiseDijkstra itself, but the
/// cross-check against Floyd-Warshall is capped to keep fuzzing throughput
/// reasonable.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_pairwise_dijkstra_matches_floyd_warshall(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let rows = csr.number_of_rows().as_();
    let columns = csr.number_of_columns().as_();

    if rows != columns {
        let result = csr.pairwise_dijkstra();
        assert!(
            matches!(
                result,
                Err(PairwiseDijkstraError::NonSquareMatrix {
                    rows: found_rows,
                    columns: found_columns,
                }) if found_rows == rows && found_columns == columns
            ),
            "non-square matrix should return NonSquareMatrix, got {result:?}"
        );
        return;
    }

    if rows == 0 {
        let result = csr.pairwise_dijkstra();
        let distances = result.expect("PairwiseDijkstra should succeed on an empty matrix");
        assert_eq!(distances.shape(), vec![0, 0]);
        return;
    }

    let mut max_abs = 0.0_f64;
    let mut non_finite_edges = Vec::new();
    let mut negative_edges = Vec::new();
    for row_id in csr.row_indices() {
        let source_id = row_id.as_();
        for (column_id, weight) in csr.sparse_row(row_id).zip(csr.sparse_row_values(row_id)) {
            let destination_id = column_id.as_();
            if !weight.is_finite() {
                non_finite_edges.push((source_id, destination_id));
                continue;
            }
            if weight < 0.0 {
                negative_edges.push((source_id, destination_id));
                continue;
            }
            max_abs = max_abs.max(weight.abs());
        }
    }

    if !non_finite_edges.is_empty() || !negative_edges.is_empty() {
        let result = csr.pairwise_dijkstra();
        match result {
            Err(PairwiseDijkstraError::NonFiniteWeight { source_id, destination_id }) => {
                assert!(
                    non_finite_edges.contains(&(source_id, destination_id)),
                    "expected a non-finite edge in {non_finite_edges:?}, got ({source_id}, {destination_id})"
                );
            }
            Err(PairwiseDijkstraError::NegativeWeight { source_id, destination_id }) => {
                assert!(
                    negative_edges.contains(&(source_id, destination_id)),
                    "expected a negative edge in {negative_edges:?}, got ({source_id}, {destination_id})"
                );
            }
            _ => {
                panic!(
                    "invalid weighted input should return NonFiniteWeight or NegativeWeight, got {result:?}"
                );
            }
        }
        return;
    }

    if rows > 32 || max_abs > 1e150 {
        return;
    }

    let result = csr.pairwise_dijkstra();
    let distances = result.expect(
        "PairwiseDijkstra should succeed on finite square matrices without negative weights",
    );
    let floyd_warshall = csr
        .floyd_warshall()
        .expect("Floyd-Warshall should succeed on the same non-negative weighted matrix");

    assert_eq!(distances.shape(), vec![rows, rows]);
    for node_id in 0..rows {
        assert_eq!(
            distances.value((node_id, node_id)),
            Some(0.0),
            "distance from a node to itself must be zero"
        );
    }

    for source_id in 0..rows {
        for destination_id in 0..rows {
            let actual = distances.value((source_id, destination_id));
            let expected = floyd_warshall.value((source_id, destination_id));
            match (actual, expected) {
                (None, None) => {}
                (Some(actual), Some(expected)) => {
                    let tolerance = expected.abs().max(1.0) * 1e-9;
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "distance mismatch at ({source_id}, {destination_id}): expected {expected}, got {actual}"
                    );
                }
                _ => {
                    panic!(
                        "reachability mismatch at ({source_id}, {destination_id}): expected {expected:?}, got {actual:?}"
                    );
                }
            }
        }
    }

    let distances2 =
        csr.pairwise_dijkstra().expect("PairwiseDijkstra should be deterministic on replay");
    assert_eq!(distances, distances2, "PairwiseDijkstra must be deterministic");
}

// ============================================================================
// Line-graph invariants
// ============================================================================

/// Check that the line-graph algorithms satisfy structural invariants on an
/// arbitrary directed graph wrapped in [`GenericGraph`].
///
/// Invariants checked:
/// - **Undirected `line_graph`**: `|E(L(G))| == sum_v C(deg(v), 2)` where
///   `deg(v)` counts edges with `src < dst` incident to `v`. Every edge in L(G)
///   corresponds to two original edges sharing a common endpoint.
/// - **Directed `directed_line_graph`**: `|E(L(G))| == sum_v in_deg(v) *
///   out_deg(v)`. Every edge `(i, j)` in L(G) satisfies `head(original_edge_i)
///   == tail(original_edge_j)`.
/// - **Edge map length** equals the line-graph vertex count.
/// - **Determinism**: repeated calls produce identical results.
///
/// Graphs larger than `max_nodes` are silently skipped to keep fuzzing fast.
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_line_graph_invariants(
    graph: &GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>,
    max_nodes: usize,
) {
    let n: usize = graph.number_of_nodes().into();
    if n > max_nodes {
        return;
    }

    // ── Undirected line graph ──────────────────────────────────────────
    let lg = graph.line_graph();
    let em = lg.edge_map();

    // edge_map length == number of vertices in L(G)
    assert_eq!(lg.number_of_vertices(), em.len());

    // Collect undirected edges (src < dst) and compute degrees.
    let undi_edges: Vec<(u8, u8)> =
        SparseMatrix::sparse_coordinates(graph.edges()).filter(|&(s, d)| s < d).collect();
    assert_eq!(lg.number_of_vertices(), undi_edges.len());

    // Degree per vertex for undirected view.
    let mut deg = vec![0usize; n];
    for &(s, d) in &undi_edges {
        deg[usize::from(s)] += 1;
        deg[usize::from(d)] += 1;
    }
    let expected_undi_lg_edges: usize = deg.iter().map(|&d| d * d.saturating_sub(1) / 2).sum();
    let actual_undi_lg_edges = Edges::number_of_edges(lg.graph()) / 2;
    assert_eq!(
        actual_undi_lg_edges, expected_undi_lg_edges,
        "undirected |E(L(G))| mismatch: got {actual_undi_lg_edges}, expected {expected_undi_lg_edges}"
    );

    // Every edge in L(G) must correspond to original edges sharing an endpoint.
    for (i, j) in Edges::sparse_coordinates(lg.graph()) {
        if i < j {
            let (a1, a2) = em[i];
            let (b1, b2) = em[j];
            assert!(
                a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2,
                "undirected L(G) edge ({i},{j}): originals ({a1},{a2}),({b1},{b2}) share no endpoint"
            );
        }
    }

    // Determinism.
    let lg2 = graph.line_graph();
    assert_eq!(lg.edge_map(), lg2.edge_map(), "undirected line_graph not deterministic");

    // ── Directed line graph ────────────────────────────────────────────
    let dlg = graph.directed_line_graph();
    let dem = dlg.edge_map();

    assert_eq!(dlg.number_of_vertices(), dem.len());
    let total_edges: usize = SparseMatrix::sparse_coordinates(graph.edges()).count();
    assert_eq!(dlg.number_of_vertices(), total_edges);

    // |E(L(G))| == sum_v in_deg(v) * out_deg(v).
    let mut in_deg = vec![0usize; n];
    let mut out_deg = vec![0usize; n];
    for (s, d) in SparseMatrix::sparse_coordinates(graph.edges()) {
        out_deg[usize::from(s)] += 1;
        in_deg[usize::from(d)] += 1;
    }
    let expected_di_lg_edges: usize = (0..n).map(|v| in_deg[v] * out_deg[v]).sum();
    let actual_di_lg_edges: usize = Edges::number_of_edges(dlg.graph());
    assert_eq!(
        actual_di_lg_edges, expected_di_lg_edges,
        "directed |E(L(G))| mismatch: got {actual_di_lg_edges}, expected {expected_di_lg_edges}"
    );

    // Every edge (i,j) in directed L(G): head of original edge i == tail of
    // original edge j.
    for (i, j) in Edges::sparse_coordinates(dlg.graph()) {
        let (_src_i, dst_i) = dem[i];
        let (src_j, _dst_j) = dem[j];
        assert_eq!(
            dst_i, src_j,
            "directed L(G) edge ({i},{j}): head of edge {i} is {dst_i}, tail of edge {j} is {src_j}"
        );
    }

    // Determinism.
    let dlg2 = graph.directed_line_graph();
    assert_eq!(dlg.edge_map(), dlg2.edge_map(), "directed_line_graph not deterministic");
}

// ============================================================================
// BitSquareMatrix invariants
// ============================================================================

/// Comprehensive invariant checks for [`BitSquareMatrix`].
///
/// Validates sparse matrix contracts, transpose roundtrip, `ExactSizeIterator`
/// guarantees, `neighbor_intersection_count`, `row_and_count`, constructor
/// paths, and iterator consistency.
///
/// # Arguments
///
/// * `m` – the matrix to check
/// * `mask_bytes` – arbitrary bytes used to build a `BitVec` mask for
///   `row_and_count` validation
///
/// # Panics
///
/// Panics if any invariant is violated.
#[allow(clippy::too_many_lines)]
pub fn check_bit_square_matrix_invariants(m: &BitSquareMatrix, mask_bytes: &[u8]) {
    let order = m.order();
    let cap = order.min(16);

    // ── Basic sparse matrix invariants ───────────────────────────────────
    check_sparse_matrix_invariants(m);

    // ── Edge count matches sum of row counts ─────────────────────────────
    let actual: usize = (0..order).map(|r| m.sparse_row(r).count()).sum();
    assert_eq!(m.number_of_defined_values(), actual);

    // ── has_entry consistent with sparse_row ─────────────────────────────
    for r in 0..cap {
        let row_cols: Vec<usize> = m.sparse_row(r).collect();
        for c in 0..order {
            assert_eq!(m.has_entry(r, c), row_cols.contains(&c));
        }
    }

    // ── Forward + reverse coordinate iteration ───────────────────────────
    let fwd: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    let mut rev: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).rev().collect();
    rev.reverse();
    assert_eq!(fwd, rev);

    // ── Diagonal count consistency ───────────────────────────────────────
    let diag_count = (0..order).filter(|&i| m.has_entry(i, i)).count();
    assert_eq!(m.number_of_defined_diagonal_values(), diag_count);

    // ── Row sizes consistency ────────────────────────────────────────────
    let sizes: Vec<usize> = m.sparse_row_sizes().collect();
    assert_eq!(sizes.len(), order);
    for (r, &sz) in sizes.iter().enumerate() {
        assert_eq!(sz, m.number_of_defined_values_in_row(r));
    }

    // ── sparse_rows() yields row indices repeated per entry ──────────────
    let sparse_rows: Vec<usize> = m.sparse_rows().collect();
    assert_eq!(sparse_rows.len(), actual);
    let mut expected_rows = Vec::new();
    for r in 0..order {
        let count = m.sparse_row(r).count();
        for _ in 0..count {
            expected_rows.push(r);
        }
    }
    assert_eq!(sparse_rows, expected_rows);

    // ── ExactSizeIterator contracts ──────────────────────────────────────
    let coords = SparseMatrix::sparse_coordinates(m);
    assert_eq!(coords.len(), actual);

    let row_sizes = m.sparse_row_sizes();
    assert_eq!(row_sizes.len(), order);

    let cols = m.sparse_columns();
    assert_eq!(cols.len(), actual);

    // ── last_sparse_coordinates ──────────────────────────────────────────
    let all_coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    if let Some(last) = m.last_sparse_coordinates() {
        assert_eq!(all_coords.last().copied(), Some(last));
    } else {
        assert!(all_coords.is_empty());
    }

    // ── sparse_columns consistency ───────────────────────────────────────
    let from_coords: Vec<usize> = SparseMatrix::sparse_coordinates(m).map(|(_, c)| c).collect();
    let from_iter: Vec<usize> = m.sparse_columns().collect();
    assert_eq!(from_coords, from_iter);

    // ── Transpose roundtrip and semantics ────────────────────────────────
    let t = m.transpose();
    assert_eq!(t.order(), order);
    assert_eq!(t.number_of_defined_values(), m.number_of_defined_values());
    assert_eq!(t.transpose(), *m, "transpose is not involutory");
    for &(r, c) in &all_coords {
        assert!(t.has_entry(c, r), "transpose missing ({c},{r}) for original ({r},{c})");
    }

    // ── neighbor_intersection_count cross-validation (small matrices) ────
    for i in 0..cap {
        for j in i..cap {
            let expected: usize = m.sparse_row(i).filter(|&col| m.has_entry(j, col)).count();
            assert_eq!(
                m.neighbor_intersection_count(i, j),
                expected,
                "neighbor_intersection_count({i},{j}) mismatch"
            );
            // Symmetry: AND + popcount is commutative
            assert_eq!(
                m.neighbor_intersection_count(i, j),
                m.neighbor_intersection_count(j, i),
                "neighbor_intersection_count not symmetric for ({i},{j})"
            );
        }
    }

    // ── row_and_count cross-validation ───────────────────────────────────
    let mask = if order > 0 {
        let mut bv = BitVec::repeat(false, order);
        for (idx, &byte) in mask_bytes.iter().enumerate() {
            if idx >= order {
                break;
            }
            bv.set(idx, byte & 1 != 0);
        }
        bv
    } else {
        BitVec::new()
    };
    for r in 0..cap {
        let expected: usize = m.sparse_row(r).filter(|&col| col < mask.len() && mask[col]).count();
        assert_eq!(m.row_and_count(r, &mask), expected, "row_and_count({r}) mismatch");
    }

    // ── row_bitslice consistency ─────────────────────────────────────────
    for r in 0..cap {
        let bits = m.row_bitslice(r);
        assert_eq!(bits.len(), order);
        for c in 0..order {
            assert_eq!(bits[c], m.has_entry(r, c));
        }
    }

    // ── from_edges constructor roundtrip ─────────────────────────────────
    let edges: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    let rebuilt = BitSquareMatrix::from_edges(order, edges.iter().copied());
    assert_eq!(rebuilt, *m, "from_edges roundtrip mismatch");

    // ── from_symmetric_edges constructor ─────────────────────────────────
    // Collect undirected edges (src <= dst) from original, build symmetric, verify
    let sym_edges: Vec<(usize, usize)> = edges.iter().filter(|&&(r, c)| r <= c).copied().collect();
    let sym = BitSquareMatrix::from_symmetric_edges(order, sym_edges.iter().copied());
    // Every original edge that has its mirror should appear in sym
    for &(r, c) in &sym_edges {
        assert!(sym.has_entry(r, c));
        assert!(sym.has_entry(c, r));
    }
}

#[cfg(all(test, feature = "arbitrary", feature = "std"))]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arbitrary::Arbitrary;

    use super::*;
    use crate::traits::{MatrixMut, ScalarSimilarity};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct NeedsThreeBytes([u8; 3]);

    impl<'a> Arbitrary<'a> for NeedsThreeBytes {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            let bytes = u.bytes(3)?;
            Ok(Self([bytes[0], bytes[1], bytes[2]]))
        }
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos();
            let pid = process::id();
            let path = std::env::temp_dir().join(format!("geometric_traits_{label}_{pid}_{now}"));
            fs::create_dir_all(&path).expect("failed to create temp directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn sample_sparse_csr() -> CSR2D<u16, u8, u8> {
        let mut csr: CSR2D<u16, u8, u8> = CSR2D::with_sparse_shaped_capacity((3, 3), 4);
        MatrixMut::add(&mut csr, (0, 0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 2)).expect("insert (0,2)");
        MatrixMut::add(&mut csr, (1, 1)).expect("insert (1,1)");
        MatrixMut::add(&mut csr, (2, 2)).expect("insert (2,2)");
        csr
    }

    fn sample_valued_csr_f64() -> ValuedCSR2D<u16, u8, u8, f64> {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 4);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 1, 2.0)).expect("insert (0,1)");
        MatrixMut::add(&mut csr, (1, 0, 2.0)).expect("insert (1,0)");
        MatrixMut::add(&mut csr, (1, 1, 1.0)).expect("insert (1,1)");
        csr
    }

    fn sample_valued_csr_u8() -> ValuedCSR2D<u16, u8, u8, u8> {
        let mut csr: ValuedCSR2D<u16, u8, u8, u8> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 3);
        MatrixMut::add(&mut csr, (0, 0, 7)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 1, 3)).expect("insert (0,1)");
        MatrixMut::add(&mut csr, (1, 1, 9)).expect("insert (1,1)");
        csr
    }

    #[test]
    fn test_from_bytes_success_and_failure() {
        assert_eq!(from_bytes::<u8>(&[42]), Some(42));
        assert_eq!(from_bytes::<NeedsThreeBytes>(&[1, 2, 3]), Some(NeedsThreeBytes([1, 2, 3])));
        assert!(from_bytes::<NeedsThreeBytes>(&[7, 8]).is_none());
    }

    #[test]
    fn test_replay_dir_skips_invalid_files() {
        let dir = TempDir::new("replay_dir");
        fs::write(dir.path().join("valid.bin"), [1u8, 2u8, 3u8]).expect("write valid file");
        fs::write(dir.path().join("invalid.bin"), [7u8, 8u8]).expect("write invalid file");
        fs::create_dir_all(dir.path().join("nested")).expect("create nested directory");

        let mut decoded: Vec<NeedsThreeBytes> = replay_dir(dir.path());
        decoded.sort_unstable();
        assert_eq!(decoded, vec![NeedsThreeBytes([1, 2, 3])]);
    }

    #[test]
    fn test_replay_dir_missing_directory_returns_empty() {
        let dir = TempDir::new("replay_missing");
        let missing = dir.path().join("does_not_exist");
        let decoded: Vec<u8> = replay_dir(&missing);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_check_sparse_matrix_invariants_on_valid_matrix() {
        let csr = sample_sparse_csr();
        check_sparse_matrix_invariants(&csr);
    }

    #[test]
    fn test_check_valued_matrix_invariants_on_valid_matrix() {
        let csr = sample_valued_csr_f64();
        check_valued_matrix_invariants(&csr);
    }

    #[test]
    fn test_check_padded_diagonal_invariants_on_valid_matrix() {
        fn one(_: u8) -> f64 {
            1.0
        }

        let mut base: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 3);
        MatrixMut::add(&mut base, (0, 1, 4.0)).expect("insert (0,1)");
        MatrixMut::add(&mut base, (1, 0, 4.0)).expect("insert (1,0)");
        MatrixMut::add(&mut base, (1, 1, 2.0)).expect("insert (1,1)");

        let padded = GenericMatrix2DWithPaddedDiagonal::new(base, one as fn(u8) -> f64)
            .expect("padded diagonal construction");
        check_padded_diagonal_invariants(&padded);
    }

    #[test]
    fn test_check_padded_matrix2d_invariants_on_valid_matrix() {
        let csr = sample_valued_csr_u8();
        check_padded_matrix2d_invariants(&csr);
    }

    #[test]
    fn test_check_kahn_ordering_on_simple_dag() {
        let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
            SquareCSR2D::with_sparse_shaped_capacity(3, 2);
        matrix.extend(vec![(0, 1), (1, 2)]).expect("extend matrix");
        check_kahn_ordering(&matrix, 10);
    }

    struct IdentitySimilarity;

    impl ScalarSimilarity<u8, u8> for IdentitySimilarity {
        type Similarity = f64;

        fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
            if left == right { 1.0 } else { 0.5 }
        }
    }

    struct BadSimilarity;

    impl ScalarSimilarity<u8, u8> for BadSimilarity {
        type Similarity = f64;

        fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
            if left == right { 0.0 } else { 0.4 }
        }
    }

    #[test]
    fn test_check_similarity_invariants_passes_for_valid_similarity() {
        let nodes = [0u8, 1u8, 2u8];
        check_similarity_invariants(&IdentitySimilarity, &nodes, 3);
    }

    #[test]
    #[should_panic(expected = "self-similarity")]
    fn test_check_similarity_invariants_panics_for_invalid_similarity() {
        let nodes = [0u8, 1u8];
        check_similarity_invariants(&BadSimilarity, &nodes, 2);
    }

    #[test]
    fn test_validate_lap_assignment_accepts_valid_assignment() {
        let csr = sample_valued_csr_f64();
        validate_lap_assignment(&csr, &[(0, 0), (1, 1)], "valid");
    }

    #[test]
    #[should_panic(expected = "duplicate row")]
    fn test_validate_lap_assignment_panics_on_duplicate_row() {
        let csr = sample_valued_csr_f64();
        validate_lap_assignment(&csr, &[(0, 0), (0, 1)], "duplicate_row");
    }

    #[test]
    #[should_panic(expected = "non-existing edge")]
    fn test_validate_lap_assignment_panics_on_missing_edge() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (1, 1, 2.0)).expect("insert (1,1)");
        validate_lap_assignment(&csr, &[(0, 1)], "missing");
    }

    #[test]
    fn test_lap_values_are_numerically_stable_true() {
        let csr = sample_valued_csr_f64();
        assert!(lap_values_are_numerically_stable(&csr));
    }

    #[test]
    fn test_lap_values_are_numerically_stable_false_for_large_ratio() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0e-10)).expect("insert small value");
        MatrixMut::add(&mut csr, (1, 1, 1.0e10)).expect("insert large value");
        assert!(!lap_values_are_numerically_stable(&csr));
    }

    #[test]
    fn test_lap_assignment_cost_returns_expected_cost() {
        let csr = sample_valued_csr_f64();
        let cost = lap_assignment_cost(&csr, &[(0, 0), (1, 1)]);
        assert!((cost - 2.0).abs() <= f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "non-existing edge")]
    fn test_lap_assignment_cost_panics_for_missing_edge() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (1, 1, 2.0)).expect("insert (1,1)");
        let _ = lap_assignment_cost(&csr, &[(0, 1)]);
    }

    #[test]
    fn test_check_lap_sparse_wrapper_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_lap_sparse_wrapper_invariants(&csr);
    }

    #[test]
    fn test_check_lap_square_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_lap_square_invariants(&csr);
    }

    #[test]
    fn test_louvain_weights_are_numerically_stable_true() {
        let csr = sample_valued_csr_f64();
        assert!(louvain_weights_are_numerically_stable(&csr));
    }

    #[test]
    fn test_louvain_weights_are_numerically_stable_false_for_large_ratio() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0e-20)).expect("insert small value");
        MatrixMut::add(&mut csr, (1, 1, 1.0e20)).expect("insert large value");
        assert!(!louvain_weights_are_numerically_stable(&csr));
    }

    #[test]
    fn test_check_louvain_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_louvain_invariants(&csr);
    }

    #[test]
    fn test_check_jacobi_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_jacobi_invariants(&csr);
    }

    #[test]
    fn test_check_mds_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_mds_invariants(&csr);
    }

    mod coverage_submodule {
        use super::*;

        struct AsymmetricSimilarity;

        impl ScalarSimilarity<u8, u8> for AsymmetricSimilarity {
            type Similarity = f64;

            fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
                match (*left, *right) {
                    (0, 1) => 0.2,
                    (1, 0) => 0.8,
                    _ if left == right => 1.0,
                    _ => 0.5,
                }
            }
        }

        #[test]
        fn test_check_kahn_ordering_returns_for_size_guard() {
            let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(3, 2);
            matrix.extend(vec![(0, 1), (1, 2)]).expect("extend matrix");
            check_kahn_ordering(&matrix, 2);
        }

        #[test]
        #[should_panic(expected = "sim(0, 1)")]
        fn test_check_similarity_invariants_panics_for_asymmetry() {
            check_similarity_invariants(&AsymmetricSimilarity, &[0u8, 1u8], 2);
        }

        #[test]
        fn test_asymmetric_similarity_default_arm() {
            let similarity = AsymmetricSimilarity;
            assert!((similarity.similarity(&2, &3) - 0.5).abs() <= f64::EPSILON);
        }

        #[test]
        #[should_panic(expected = "row index out of bounds")]
        fn test_validate_lap_assignment_panics_on_row_out_of_bounds() {
            let csr = sample_valued_csr_f64();
            validate_lap_assignment(&csr, &[(2, 0)], "row_oob");
        }

        #[test]
        #[should_panic(expected = "column index out of bounds")]
        fn test_validate_lap_assignment_panics_on_column_out_of_bounds() {
            let csr = sample_valued_csr_f64();
            validate_lap_assignment(&csr, &[(0, 2)], "column_oob");
        }

        #[test]
        fn test_check_louvain_invariants_returns_for_unstable_weights() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
            MatrixMut::add(&mut csr, (0, 0, 1.0e-20)).expect("insert tiny value");
            MatrixMut::add(&mut csr, (1, 1, 1.0e20)).expect("insert huge value");
            check_louvain_invariants(&csr);
        }

        #[test]
        fn test_check_line_graph_invariants_smoke() {
            let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(4, 4);
            matrix.extend(vec![(0, 1), (0, 2), (1, 2), (2, 3)]).expect("extend matrix");
            let graph: GenericGraph<u8, _> = GenericGraph::from((4u8, matrix));
            check_line_graph_invariants(&graph, 32);
        }

        #[test]
        fn test_check_line_graph_invariants_returns_for_size_guard() {
            let matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(4, 0);
            let graph: GenericGraph<u8, _> = GenericGraph::from((4u8, matrix));
            check_line_graph_invariants(&graph, 2);
        }

        #[test]
        fn test_check_louvain_invariants_returns_when_symmetrized_edge_count_overflows_u8() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((17, 17), 17 * 17);

            for row in 0u8..17 {
                for column in 0u8..17 {
                    MatrixMut::add(&mut csr, (row, column, 1.0)).expect("insert dense edge");
                }
            }

            check_louvain_invariants(&csr);
        }
    }
}
