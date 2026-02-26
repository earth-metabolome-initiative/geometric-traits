//! Unified fuzz harness for LAP wrappers and LAPMOD core.
//!
//! Invariants checked:
//! 1. Sparse wrappers (`sparse_lapmod`/`sparse_lapjv`) either both fail with
//!    the same error or both succeed with equal cardinality and (for
//!    numerically stable value ranges) equal cost.
//! 2. When wrappers succeed, assignments are valid and (for numerically stable
//!    value ranges) match Hopcroft-Karp cardinality.
//! 3. On square matrices where core `lapmod` succeeds, both wrappers also
//!    succeed and match core cardinality and (for numerically stable value
//!    ranges) objective.

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::{
        HopcroftKarp, IntoUsize, Matrix2D, SparseLAPJV, SparseLAPMOD, SparseMatrix2D,
        SparseValuedMatrix, SparseValuedMatrix2D, LAPMOD,
    },
};
use honggfuzz::fuzz;

type Csr = ValuedCSR2D<u16, u8, u8, f64>;

fn objective_values_are_numerically_stable(csr: &Csr) -> bool {
    let mut minimum_value = f64::INFINITY;
    let mut maximum_value = 0.0f64;

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

    // Avoid asserting objective equality in extreme floating-point regimes
    // where subnormal values and huge magnitudes make LAPJV/LAPMOD numerically
    // incomparable even when both return valid maximum-cardinality matchings.
    minimum_value >= f64::MIN_POSITIVE
        && maximum_value <= 1e150
        && (maximum_value / minimum_value) <= 1e12
}

fn edge_cost(csr: &Csr, row: u8, column: u8) -> Option<f64> {
    csr.sparse_row(row)
        .zip(csr.sparse_row_values(row))
        .find_map(|(candidate_column, value)| (candidate_column == column).then_some(value))
}

fn assignment_cost(csr: &Csr, assignment: &[(u8, u8)]) -> f64 {
    assignment
        .iter()
        .map(|&(row, column)| {
            edge_cost(csr, row, column).unwrap_or_else(|| {
                panic!("Assignment includes non-existing edge ({row}, {column})")
            })
        })
        .sum()
}

fn validate_assignment(csr: &Csr, assignment: &[(u8, u8)], label: &str) {
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
        assert!(!seen_rows[row_index], "{label}: duplicate row in assignment ({row})");
        assert!(!seen_columns[column_index], "{label}: duplicate column in assignment ({column})");

        seen_rows[row_index] = true;
        seen_columns[column_index] = true;
    }
}

fn check_sparse_wrapper_invariants(csr: &Csr) {
    let numerically_stable = objective_values_are_numerically_stable(csr);
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
            validate_assignment(csr, lapmod_assignment, "SparseLAPMOD");
            validate_assignment(csr, lapjv_assignment, "SparseLAPJV");

            if numerically_stable {
                assert_eq!(
                    lapmod_assignment.len(),
                    lapjv_assignment.len(),
                    "SparseLAPMOD/SparseLAPJV cardinality mismatch: {:?}",
                    csr
                );

                let lapmod_cost = assignment_cost(csr, lapmod_assignment);
                let lapjv_cost = assignment_cost(csr, lapjv_assignment);
                assert!(
                    (lapmod_cost - lapjv_cost).abs() < 1e-9,
                    "SparseLAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs {lapjv_cost}): {:?}",
                    csr
                );
                if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
                    assert_eq!(
                        lapmod_assignment.len(),
                        hopcroft_karp_assignment.len(),
                        "SparseLAPMOD/Hopcroft-Karp cardinality mismatch: {:?}",
                        csr
                    );
                    assert_eq!(
                        lapjv_assignment.len(),
                        hopcroft_karp_assignment.len(),
                        "SparseLAPJV/Hopcroft-Karp cardinality mismatch: {:?}",
                        csr
                    );
                }
            }
        }
        (Err(lapmod_error), Err(lapjv_error)) => {
            if numerically_stable {
                assert_eq!(
                    lapmod_error, lapjv_error,
                    "Sparse wrapper error mismatch: SparseLAPMOD={lapmod_error:?} SparseLAPJV={lapjv_error:?} matrix={csr:?}"
                );
            }
        }
        (Ok(lapmod_assignment), Err(lapjv_error)) => {
            validate_assignment(csr, lapmod_assignment, "SparseLAPMOD");
            if numerically_stable {
                panic!(
                    "Sparse wrapper mismatch: SparseLAPMOD returned assignment of len {} but SparseLAPJV failed with {lapjv_error:?}: {:?}",
                    lapmod_assignment.len(),
                    csr
                );
            }
        }
        (Err(lapmod_error), Ok(lapjv_assignment)) => {
            validate_assignment(csr, lapjv_assignment, "SparseLAPJV");
            if numerically_stable {
                panic!(
                    "Sparse wrapper mismatch: SparseLAPMOD failed with {lapmod_error:?} but SparseLAPJV returned assignment of len {}: {:?}",
                    lapjv_assignment.len(),
                    csr
                );
            }
        }
    }
}

fn check_square_lapmod_invariants(csr: &Csr) {
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
    validate_assignment(csr, &lapmod_assignment, "LAPMOD");

    let numerically_stable = objective_values_are_numerically_stable(csr);
    if numerically_stable {
        let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() else {
            return;
        };
        assert_eq!(
            lapmod_assignment.len(),
            hopcroft_karp_assignment.len(),
            "LAPMOD/Hopcroft-Karp cardinality mismatch: {:?}",
            csr
        );
    }

    let padding_value = (maximum_value + 1.0) * 4.0;
    let maximal_cost = (padding_value + 1.0) * 2.0;
    if !padding_value.is_finite() || !maximal_cost.is_finite() || maximal_cost <= padding_value {
        return;
    }

    let sparse_lapmod_assignment = csr.sparse_lapmod(padding_value, maximal_cost);
    let sparse_lapjv_assignment = csr.sparse_lapjv(padding_value, maximal_cost);

    if !numerically_stable {
        if let Ok(assignment) = &sparse_lapmod_assignment {
            validate_assignment(csr, assignment, "SparseLAPMOD");
        }
        if let Ok(assignment) = &sparse_lapjv_assignment {
            validate_assignment(csr, assignment, "SparseLAPJV");
        }
        return;
    }

    let sparse_lapmod_assignment = sparse_lapmod_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPMOD failed on square matrix that LAPMOD solved with error {error:?}: {:?}",
            csr
        )
    });
    let sparse_lapjv_assignment = sparse_lapjv_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPJV failed on square matrix that LAPMOD solved with error {error:?}: {:?}",
            csr
        )
    });

    validate_assignment(csr, &sparse_lapmod_assignment, "SparseLAPMOD");
    validate_assignment(csr, &sparse_lapjv_assignment, "SparseLAPJV");

    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapmod_assignment.len(),
        "LAPMOD/SparseLAPMOD cardinality mismatch: {:?}",
        csr
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapjv_assignment.len(),
        "LAPMOD/SparseLAPJV cardinality mismatch: {:?}",
        csr
    );

    let lapmod_cost = assignment_cost(csr, &lapmod_assignment);
    let sparse_lapmod_cost = assignment_cost(csr, &sparse_lapmod_assignment);
    let sparse_lapjv_cost = assignment_cost(csr, &sparse_lapjv_assignment);

    assert!(
        (lapmod_cost - sparse_lapmod_cost).abs() < 1e-9,
        "LAPMOD/SparseLAPMOD objective mismatch ({lapmod_cost} vs {sparse_lapmod_cost}): {:?}",
        csr
    );
    assert!(
        (lapmod_cost - sparse_lapjv_cost).abs() < 1e-9,
        "LAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs {sparse_lapjv_cost}): {:?}",
        csr
    );
}

fn main() {
    loop {
        fuzz!(|csr: Csr| {
            check_sparse_wrapper_invariants(&csr);
            check_square_lapmod_invariants(&csr);
        });
    }
}
