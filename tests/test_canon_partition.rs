//! Tests for the backtrackable ordered partition used by canonization.
#![cfg(feature = "std")]

use geometric_traits::traits::BacktrackableOrderedPartition;

fn collect_cells(partition: &BacktrackableOrderedPartition) -> Vec<Vec<usize>> {
    partition.cells().map(|cell| cell.elements().to_vec()).collect()
}

fn collect_non_singleton_cells(partition: &BacktrackableOrderedPartition) -> Vec<Vec<usize>> {
    partition.non_singleton_cells().map(|cell| cell.elements().to_vec()).collect()
}

fn collect_cells_with_sorted_members(partition: &BacktrackableOrderedPartition) -> Vec<Vec<usize>> {
    let mut cells = collect_cells(partition);
    for cell in &mut cells {
        cell.sort_unstable();
    }
    cells
}

#[test]
fn test_partition_initializes_empty_and_unit_partitions() {
    let empty = BacktrackableOrderedPartition::new(0);
    assert!(empty.is_empty());
    assert!(empty.is_discrete());
    assert_eq!(empty.number_of_cells(), 0);
    assert_eq!(empty.signature(), Vec::<usize>::new());
    assert!(collect_cells(&empty).is_empty());

    let unit = BacktrackableOrderedPartition::new(4);
    assert_eq!(unit.order(), 4);
    assert!(!unit.is_empty());
    assert!(!unit.is_discrete());
    assert_eq!(unit.number_of_cells(), 1);
    assert_eq!(unit.number_of_discrete_cells(), 0);
    assert_eq!(unit.signature(), vec![4]);
    assert_eq!(collect_cells(&unit), vec![vec![0, 1, 2, 3]]);
    assert_eq!(collect_non_singleton_cells(&unit), vec![vec![0, 1, 2, 3]]);
}

#[test]
fn test_partition_individualize_and_backtrack_round_trip() {
    let mut partition = BacktrackableOrderedPartition::new(4);
    let root = partition.set_backtrack_point();
    let original_cell = partition.cell_of(2);
    let singleton = partition.individualize(original_cell, 2);

    assert_eq!(partition.number_of_cells(), 2);
    assert_eq!(partition.number_of_discrete_cells(), 1);
    assert_eq!(partition.signature(), vec![3]);
    assert_eq!(partition.cell_of(2), singleton);
    assert_eq!(collect_cells(&partition), vec![vec![0, 1, 3], vec![2]]);
    assert_eq!(collect_non_singleton_cells(&partition), vec![vec![0, 1, 3]]);

    partition.goto_backtrack_point(root);

    assert_eq!(partition.number_of_cells(), 1);
    assert_eq!(partition.number_of_discrete_cells(), 0);
    assert_eq!(partition.signature(), vec![4]);
    assert_eq!(collect_cells_with_sorted_members(&partition), vec![vec![0, 1, 2, 3]]);
}

#[test]
fn test_partition_split_cell_by_key_groups_equal_keys_deterministically() {
    let mut partition = BacktrackableOrderedPartition::new(6);
    let root = partition.set_backtrack_point();
    let cell = partition.cell_of(0);
    let groups = partition.split_cell_by_key(cell, |element| [2_u8, 1, 2, 1, 3, 3][element]);

    assert_eq!(groups.len(), 3);
    assert_eq!(partition.signature(), vec![2, 2, 2]);
    assert_eq!(collect_cells(&partition), vec![vec![1, 3], vec![0, 2], vec![4, 5]]);
    assert_eq!(
        groups.iter().map(|&group| partition.cell_elements(group).to_vec()).collect::<Vec<_>>(),
        vec![vec![1, 3], vec![0, 2], vec![4, 5]]
    );
    assert_eq!(partition.cell_of(1), groups[0]);
    assert_eq!(partition.cell_of(0), groups[1]);
    assert_eq!(partition.cell_of(4), groups[2]);

    partition.goto_backtrack_point(root);
    assert_eq!(collect_cells_with_sorted_members(&partition), vec![vec![0, 1, 2, 3, 4, 5]]);
}

#[test]
fn test_partition_split_cell_by_key_preserves_prior_order_within_equal_keys() {
    let mut partition = BacktrackableOrderedPartition::new(5);
    let root = partition.set_backtrack_point();
    let original = partition.cell_of(0);
    let singleton = partition.individualize(original, 0);

    let prior_remainder = partition.cell_elements(partition.cell_of(4)).to_vec();
    assert_eq!(partition.cell_elements(singleton), &[0]);

    let groups = partition.split_cell_by_key(partition.cell_of(4), |element| match element {
        1 | 2 => 0_u8,
        3 | 4 => 1_u8,
        _ => unreachable!("split is only applied to the non-singleton remainder"),
    });
    let expected_zero_group = prior_remainder
        .iter()
        .copied()
        .filter(|&element| matches!(element, 1 | 2))
        .collect::<Vec<_>>();
    let expected_one_group = prior_remainder
        .iter()
        .copied()
        .filter(|&element| matches!(element, 3 | 4))
        .collect::<Vec<_>>();

    assert_eq!(groups.len(), 2);
    assert_eq!(collect_cells(&partition), vec![expected_zero_group, expected_one_group, vec![0]]);

    partition.goto_backtrack_point(root);
    assert_eq!(collect_cells_with_sorted_members(&partition), vec![vec![0, 1, 2, 3, 4]]);
}

#[test]
fn test_partition_supports_nested_splits_and_backtracking() {
    let mut partition = BacktrackableOrderedPartition::new(5);
    let root = partition.set_backtrack_point();
    let first_groups =
        partition.split_cell_by_key(partition.cell_of(0), |element| usize::from(element >= 2));

    assert_eq!(collect_cells(&partition), vec![vec![0, 1], vec![2, 3, 4]]);
    assert_eq!(partition.signature(), vec![2, 3]);

    let nested = partition.set_backtrack_point();
    let singleton = partition.individualize(first_groups[1], 4);

    assert_eq!(partition.cell_of(4), singleton);
    assert_eq!(collect_cells(&partition), vec![vec![0, 1], vec![2, 3], vec![4]]);
    assert_eq!(collect_non_singleton_cells(&partition), vec![vec![0, 1], vec![2, 3]]);
    assert_eq!(partition.signature(), vec![2, 2]);

    partition.goto_backtrack_point(nested);
    assert_eq!(collect_cells(&partition), vec![vec![0, 1], vec![2, 3, 4]]);
    assert_eq!(partition.signature(), vec![2, 3]);

    partition.goto_backtrack_point(root);
    assert_eq!(collect_cells_with_sorted_members(&partition), vec![vec![0, 1, 2, 3, 4]]);
    assert_eq!(partition.signature(), vec![5]);
}
