//! Integration tests for `VecMatrix2D` dense trait implementations.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::VecMatrix2D,
    prelude::{DenseValuedMatrix, DenseValuedMatrix2D, Hungarian, LAPJV, Matrix2D},
};

#[test]
fn test_value_access() {
    let m = VecMatrix2D::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(m.value((0, 0)), 1);
    assert_eq!(m.value((0, 2)), 3);
    assert_eq!(m.value((1, 0)), 4);
    assert_eq!(m.value((1, 2)), 6);
}

#[test]
fn test_row_values() {
    let m = VecMatrix2D::new(2, 3, vec![10, 20, 30, 40, 50, 60]);
    let row0: Vec<i32> = m.row_values(0).collect();
    let row1: Vec<i32> = m.row_values(1).collect();
    assert_eq!(row0, vec![10, 20, 30]);
    assert_eq!(row1, vec![40, 50, 60]);
}

#[test]
fn test_values_iteration() {
    let m = VecMatrix2D::new(2, 2, vec![1, 2, 3, 4]);
    let all: Vec<i32> = m.values().collect();
    assert_eq!(all, vec![1, 2, 3, 4]);
}

#[test]
fn test_matrix2d_dimensions() {
    let m = VecMatrix2D::new(3, 4, vec![0.0_f64; 12]);
    assert_eq!(m.number_of_rows(), 3);
    assert_eq!(m.number_of_columns(), 4);
}

#[test]
fn test_empty_matrix_dimensions() {
    let m = VecMatrix2D::<f64>::new(0, 0, vec![]);
    assert_eq!(m.number_of_rows(), 0);
    assert_eq!(m.number_of_columns(), 0);
}

#[test]
fn test_lapjv_on_vec_matrix2d() {
    let m = VecMatrix2D::new(3, 3, vec![1.0, 0.5, 10.0, 0.5, 10.0, 20.0, 10.0, 20.0, 0.5]);
    let mut assignment = m.lapjv(100.0).expect("LAPJV failed");
    assignment.sort_unstable();
    assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_hungarian_on_vec_matrix2d() {
    let m = VecMatrix2D::new(3, 3, vec![1.0, 0.5, 10.0, 0.5, 10.0, 20.0, 10.0, 20.0, 0.5]);
    let mut assignment = m.hungarian(100.0).expect("Hungarian failed");
    assignment.sort_unstable();
    assert_eq!(assignment, vec![(0, 1), (1, 0), (2, 2)]);
}

#[test]
fn test_lapjv_hungarian_agreement() {
    let m = VecMatrix2D::new(
        4,
        4,
        vec![5.0, 9.0, 1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 6.0, 1.0, 3.0, 9.0, 8.0, 7.0, 6.0, 2.0],
    );
    let mut lapjv = m.lapjv(100.0).expect("LAPJV failed");
    let mut hung = m.hungarian(100.0).expect("Hungarian failed");
    lapjv.sort_unstable();
    hung.sort_unstable();

    let cost = |assignments: &[(usize, usize)]| -> f64 {
        assignments.iter().map(|&(r, c)| m.value((r, c))).sum()
    };
    let lapjv_cost = cost(&lapjv);
    let hungarian_cost = cost(&hung);
    let denom = lapjv_cost.abs().max(hungarian_cost.abs()).max(1e-30);
    assert!(
        (lapjv_cost - hungarian_cost).abs() / denom < 1e-9,
        "Cost mismatch: lapjv={lapjv_cost} hungarian={hungarian_cost}"
    );
}

#[test]
#[should_panic(expected = "assertion")]
fn test_new_mismatched_dimensions() {
    let _ = VecMatrix2D::new(2, 3, vec![1; 5]);
}
