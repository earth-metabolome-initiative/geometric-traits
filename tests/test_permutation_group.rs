//! Integration tests for [`PermutationGroup`].
#![cfg(feature = "alloc")]

use geometric_traits::prelude::*;

fn identity(degree: usize) -> Vec<usize> {
    (0..degree).collect()
}

/// The permutation of `0..degree` that maps `cycle[i]` to `cycle[i + 1]`.
fn cycle(degree: usize, cycle: &[usize]) -> Vec<usize> {
    let mut permutation = identity(degree);
    for (index, &point) in cycle.iter().enumerate() {
        permutation[point] = cycle[(index + 1) % cycle.len()];
    }
    permutation
}

/// Applies `first`, then `second`.
fn compose(first: &[usize], second: &[usize]) -> Vec<usize> {
    first.iter().map(|&point| second[point]).collect()
}

fn sorted_image(element: &[usize], set: &[usize]) -> Vec<usize> {
    let mut image: Vec<usize> = set.iter().map(|&point| element[point]).collect();
    image.sort_unstable();
    image
}

fn symmetric(degree: usize) -> PermutationGroup {
    let all: Vec<usize> = identity(degree);
    PermutationGroup::new(degree, [cycle(degree, &[0, 1]), cycle(degree, &all)]).unwrap()
}

fn dihedral(degree: usize) -> PermutationGroup {
    let rotation = cycle(degree, &identity(degree));
    let reflection: Vec<usize> = (0..degree).map(|point| (degree - point) % degree).collect();
    PermutationGroup::new(degree, [rotation, reflection]).unwrap()
}

/// `(0 1)(2 3)…`, one generator per pair.
fn disjoint_transpositions(pairs: usize) -> Vec<Vec<usize>> {
    (0..pairs).map(|pair| cycle(2 * pairs, &[2 * pair, 2 * pair + 1])).collect()
}

fn factorial(n: u128) -> u128 {
    (1..=n).product()
}

#[test]
fn test_orders_of_known_groups() {
    assert_eq!(symmetric(5).order(), Some(120));
    assert_eq!(dihedral(4).order(), Some(8));
    assert_eq!(dihedral(5).order(), Some(10));
    let alternating = PermutationGroup::new(
        5,
        [cycle(5, &[0, 1, 2]), cycle(5, &[1, 2, 3]), cycle(5, &[2, 3, 4])],
    )
    .unwrap();
    assert_eq!(alternating.order(), Some(60));
    for pairs in [0, 1, 2, 7, 32, 64] {
        let group = PermutationGroup::new(2 * pairs, disjoint_transpositions(pairs)).unwrap();
        assert_eq!(group.order(), Some(1_u128 << pairs), "{pairs} pairs");
    }
}

#[test]
fn test_order_reports_overflow_as_none() {
    assert_eq!(symmetric(34).order(), Some(factorial(34)));
    assert_eq!(symmetric(35).order(), None);
    let pairs = 128;
    let group = PermutationGroup::new(2 * pairs, disjoint_transpositions(pairs)).unwrap();
    assert_eq!(group.order(), None);
}

#[test]
fn test_empty_generator_list_gives_the_trivial_group() {
    let group = PermutationGroup::new(4, Vec::<Vec<usize>>::new()).unwrap();
    assert_eq!(group.degree(), 4);
    assert_eq!(group.order(), Some(1));
    assert!(group.contains(&identity(4)));
    assert!(!group.contains(&cycle(4, &[0, 1])));
    assert_eq!(group.orbit(2), Ok(vec![2]));
    let smallest = group.smallest_image(&[3, 1]).unwrap();
    assert_eq!(smallest.image, vec![1, 3]);
    assert_eq!(smallest.element, identity(4));

    let empty = PermutationGroup::new(0, Vec::<Vec<usize>>::new()).unwrap();
    assert_eq!(empty.order(), Some(1));
    assert!(empty.contains(&[]));
    assert_eq!(empty.smallest_image(&[]).unwrap().image, Vec::<usize>::new());
}

#[test]
fn test_identity_and_repeated_generators_do_not_change_the_group() {
    let rotation = cycle(6, &identity(6));
    let group = PermutationGroup::new(6, [identity(6), rotation.clone(), rotation]).unwrap();
    assert_eq!(group.order(), Some(6));
}

#[test]
fn test_products_of_generators_are_members() {
    let group = dihedral(7);
    let rotation = cycle(7, &identity(7));
    let reflection: Vec<usize> = (0..7).map(|point| (7 - point) % 7).collect();
    let mut product = identity(7);
    for step in [&rotation, &reflection, &rotation, &rotation, &reflection, &rotation] {
        product = compose(&product, step);
        assert!(group.contains(&product));
    }
}

#[test]
fn test_transposition_is_not_in_the_dihedral_group_of_a_pentagon() {
    let group = dihedral(5);
    assert!(!group.contains(&cycle(5, &[0, 1])));
    assert!(!group.contains(&cycle(5, &[0, 2, 1])));
    assert!(group.contains(&[0, 4, 3, 2, 1]));
}

#[test]
fn test_contains_is_false_for_slices_that_are_not_permutations_of_the_points() {
    let group = symmetric(4);
    assert!(!group.contains(&[0, 1, 2]));
    assert!(!group.contains(&[0, 1, 2, 3, 4]));
    assert!(!group.contains(&[0, 0, 2, 3]));
    assert!(!group.contains(&[0, 1, 2, 7]));
}

#[test]
fn test_orbits_follow_the_cycles_of_the_generators() {
    let group = PermutationGroup::new(7, [cycle(7, &[0, 3, 5]), cycle(7, &[1, 6])]).unwrap();
    assert_eq!(group.orbit(5), Ok(vec![0, 3, 5]));
    assert_eq!(group.orbit(1), Ok(vec![1, 6]));
    assert_eq!(group.orbit(2), Ok(vec![2]));
    assert_eq!(group.orbit(7), Err(PermutationGroupError::PointOutOfRange { point: 7, degree: 7 }));
}

#[test]
fn test_new_rejects_generators_that_are_not_permutations_of_the_points() {
    assert_eq!(
        PermutationGroup::new(3, [vec![1, 0, 2], vec![0, 1]]).unwrap_err(),
        PermutationGroupError::GeneratorLength { generator: 1, length: 2, degree: 3 }
    );
    assert_eq!(
        PermutationGroup::new(3, [vec![1, 1, 2]]).unwrap_err(),
        PermutationGroupError::NotAPermutation { generator: 0, degree: 3 }
    );
    assert_eq!(
        PermutationGroup::new(3, [vec![0, 1, 3]]).unwrap_err(),
        PermutationGroupError::NotAPermutation { generator: 0, degree: 3 }
    );
}

#[test]
fn test_smallest_image_rejects_points_outside_the_domain_and_repeats() {
    let group = symmetric(4);
    assert_eq!(
        group.smallest_image(&[0, 4]).unwrap_err(),
        PermutationGroupError::PointOutOfRange { point: 4, degree: 4 }
    );
    assert_eq!(
        group.smallest_image(&[2, 1, 2]).unwrap_err(),
        PermutationGroupError::DuplicatePoint { point: 2 }
    );
}

#[test]
fn test_smallest_image_picks_one_point_per_transposition() {
    for pairs in [1, 8, 32, 64] {
        let group = PermutationGroup::new(2 * pairs, disjoint_transpositions(pairs)).unwrap();
        let set: Vec<usize> = (0..pairs).map(|pair| 2 * pair + 1).collect();
        let smallest = group.smallest_image(&set).unwrap();
        let expected: Vec<usize> = (0..pairs).map(|pair| 2 * pair).collect();
        assert_eq!(smallest.image, expected);
        assert!(group.contains(&smallest.element));
        assert_eq!(sorted_image(&smallest.element, &set), expected);
    }
}

#[test]
fn test_smallest_image_under_the_symmetric_group_is_the_first_points() {
    let group = symmetric(40);
    let set: Vec<usize> = (0..40).filter(|point| point % 3 != 0).collect();
    let smallest = group.smallest_image(&set).unwrap();
    assert_eq!(smallest.image, identity(set.len()));
    assert!(group.contains(&smallest.element));
    assert_eq!(sorted_image(&smallest.element, &set), smallest.image);
}

#[test]
fn test_smallest_image_under_a_rotation_compares_whole_sets() {
    // Three rotations of {4, 5, 7} contain 0, namely {0, 1, 3}, {0, 2, 7} and
    // {0, 5, 6}, and only the rotation by four reaches {0, 1, 3}.
    let group = PermutationGroup::new(8, [cycle(8, &identity(8))]).unwrap();
    let smallest = group.smallest_image(&[4, 5, 7]).unwrap();
    assert_eq!(smallest.image, vec![0, 1, 3]);
    assert_eq!(smallest.element, (0..8).map(|point| (point + 4) % 8).collect::<Vec<_>>());
}

#[cfg(feature = "arbitrary")]
#[test]
fn test_random_small_groups_match_brute_force_closure() {
    use geometric_traits::test_utils::check_permutation_group_invariants;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    let mut rng = SmallRng::seed_from_u64(0x5EED_6A0B_0000_0001);
    for degree in 0..=7_usize {
        let sets: Vec<Vec<usize>> = (0..1_usize << degree)
            .map(|mask| (0..degree).filter(|point| mask & (1 << point) != 0).collect())
            .collect();
        for _ in 0..40 {
            let generator_count = rng.random_range(0..=3);
            let generators: Vec<Vec<usize>> = (0..generator_count)
                .map(|_| {
                    // Permuting a random subset of the points yields groups
                    // smaller than the full symmetric group.
                    let moved: Vec<usize> = (0..degree).filter(|_| rng.random_bool(0.6)).collect();
                    let mut images = moved.clone();
                    for index in (1..images.len()).rev() {
                        images.swap(index, rng.random_range(0..=index));
                    }
                    let mut generator = identity(degree);
                    for (&point, &image) in moved.iter().zip(&images) {
                        generator[point] = image;
                    }
                    generator
                })
                .collect();
            check_permutation_group_invariants(degree, &generators, &sets);
        }
    }
}
