//! Tests for the Leicht-Newman directed leading-eigenvector community detector.
#![cfg(feature = "std")]

use std::collections::{HashMap, HashSet};

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LeichtNewmanConfig, ModularityError},
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

fn build(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WeightedMatrix {
    let mut edges = edges;
    edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

/// Complete directed clique on `start..start + size` (all ordered pairs).
fn directed_clique(start: usize, size: usize, weight: f64) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for source in start..start + size {
        for destination in start..start + size {
            if source != destination {
                edges.push((source, destination, weight));
            }
        }
    }
    edges
}

fn partitions_are_equivalent(expected: &[usize], actual: &[usize]) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    let mut expected_to_actual: HashMap<usize, usize> = HashMap::new();
    let mut actual_to_expected: HashMap<usize, usize> = HashMap::new();
    for (e, a) in expected.iter().copied().zip(actual.iter().copied()) {
        if let Some(mapped) = expected_to_actual.get(&e) {
            if *mapped != a {
                return false;
            }
        } else {
            expected_to_actual.insert(e, a);
        }
        if let Some(mapped) = actual_to_expected.get(&a) {
            if *mapped != e {
                return false;
            }
        } else {
            actual_to_expected.insert(a, e);
        }
    }
    true
}

fn distinct_count(partition: &[usize]) -> usize {
    partition.iter().copied().collect::<HashSet<_>>().len()
}

#[test]
fn test_recovers_two_directed_cycles() {
    let graph = build(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
    let result =
        LeichtNewman::<usize>::leicht_newman(&graph, &LeichtNewmanConfig::default()).unwrap();
    let partition = result.partition();
    assert_eq!(partition.len(), 4);
    assert!(
        partitions_are_equivalent(&[0, 0, 1, 1], partition),
        "expected two communities, got {partition:?}"
    );
    assert_eq!(result.number_of_communities(), 2);
    assert!((result.modularity() - 0.5).abs() < 1e-9, "modularity {}", result.modularity());
}

#[test]
fn test_recovers_two_weakly_linked_triangles() {
    let mut edges =
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 3, 1.0)];
    edges.push((2, 3, 0.05));
    edges.push((5, 0, 0.05));
    let graph = build(6, edges);
    let result =
        LeichtNewman::<usize>::leicht_newman(&graph, &LeichtNewmanConfig::default()).unwrap();
    assert!(
        partitions_are_equivalent(&[0, 0, 0, 1, 1, 1], result.partition()),
        "expected two triangles, got {:?}",
        result.partition()
    );
}

#[test]
fn test_recovers_three_directed_cliques() {
    let mut edges = directed_clique(0, 4, 1.0);
    edges.extend(directed_clique(4, 4, 1.0));
    edges.extend(directed_clique(8, 4, 1.0));
    let graph = build(12, edges);
    let result =
        LeichtNewman::<usize>::leicht_newman(&graph, &LeichtNewmanConfig::default()).unwrap();
    assert_eq!(result.number_of_communities(), 3, "got {:?}", result.partition());
    assert!(partitions_are_equivalent(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], result.partition()));
}

#[test]
fn test_is_deterministic() {
    let mut edges = directed_clique(0, 5, 1.0);
    edges.extend(directed_clique(5, 5, 1.0));
    edges.push((4, 5, 0.1));
    let graph = build(10, edges);
    let config = LeichtNewmanConfig::default();
    let first = LeichtNewman::<usize>::leicht_newman(&graph, &config).unwrap();
    let second = LeichtNewman::<usize>::leicht_newman(&graph, &config).unwrap();
    assert_eq!(first.partition(), second.partition());
    assert!((first.modularity() - second.modularity()).abs() < 1e-12);
}

#[test]
fn test_invariants_partition_length_and_modularity_bounds() {
    let mut edges = directed_clique(0, 4, 2.0);
    edges.extend(directed_clique(4, 3, 2.0));
    edges.push((0, 4, 0.01));
    let graph = build(7, edges);
    let result =
        LeichtNewman::<usize>::leicht_newman(&graph, &LeichtNewmanConfig::default()).unwrap();
    assert_eq!(result.partition().len(), 7);
    assert_eq!(result.number_of_communities(), distinct_count(result.partition()));
    let modularity = result.modularity();
    assert!((-1.0..=1.0).contains(&modularity), "modularity {modularity} out of bounds");
    // The detected partition's directed modularity must equal the reported one.
    let recomputed =
        DirectedModularity::<usize>::directed_modularity(&graph, result.partition(), 1.0).unwrap();
    assert!((recomputed - modularity).abs() < 1e-9);
}

#[test]
fn test_refinement_does_not_decrease_modularity() {
    let mut edges = directed_clique(0, 6, 1.0);
    edges.extend(directed_clique(6, 6, 1.0));
    edges.push((5, 6, 0.2));
    edges.push((11, 0, 0.2));
    let graph = build(12, edges);

    let refined = LeichtNewman::<usize>::leicht_newman(
        &graph,
        &LeichtNewmanConfig { refine: true, ..LeichtNewmanConfig::default() },
    )
    .unwrap();
    let unrefined = LeichtNewman::<usize>::leicht_newman(
        &graph,
        &LeichtNewmanConfig { refine: false, ..LeichtNewmanConfig::default() },
    )
    .unwrap();
    assert!(
        refined.modularity() >= unrefined.modularity() - 1e-9,
        "refined {} < unrefined {}",
        refined.modularity(),
        unrefined.modularity()
    );
}

#[test]
fn test_single_node_and_single_edge() {
    let single_node: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(0)
        .expected_shape((1, 1))
        .edges(Vec::<(usize, usize, f64)>::new().into_iter())
        .build()
        .unwrap();
    let result =
        LeichtNewman::<usize>::leicht_newman(&single_node, &LeichtNewmanConfig::default()).unwrap();
    assert_eq!(result.partition().len(), 1);
    assert_eq!(result.number_of_communities(), 1);

    let single_edge = build(2, vec![(0, 1, 1.0)]);
    let result =
        LeichtNewman::<usize>::leicht_newman(&single_edge, &LeichtNewmanConfig::default()).unwrap();
    assert_eq!(result.partition().len(), 2);
}

#[test]
fn test_empty_graph() {
    let empty: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(0)
        .expected_shape((0, 0))
        .edges(Vec::<(usize, usize, f64)>::new().into_iter())
        .build()
        .unwrap();
    let result =
        LeichtNewman::<usize>::leicht_newman(&empty, &LeichtNewmanConfig::default()).unwrap();
    assert_eq!(result.partition().len(), 0);
    assert_eq!(result.number_of_communities(), 0);
}

#[test]
fn test_config_validation() {
    let graph = build(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);

    let bad_resolution = LeichtNewmanConfig { resolution: 0.0, ..LeichtNewmanConfig::default() };
    assert!(matches!(
        LeichtNewman::<usize>::leicht_newman(&graph, &bad_resolution),
        Err(ModularityError::InvalidResolution)
    ));

    let bad_iterations =
        LeichtNewmanConfig { max_power_iterations: 0, ..LeichtNewmanConfig::default() };
    assert!(matches!(
        LeichtNewman::<usize>::leicht_newman(&graph, &bad_iterations),
        Err(ModularityError::InvalidMaxPowerIterations)
    ));
}

#[test]
fn test_rejects_non_square_matrix() {
    let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(1)
        .expected_shape((2, 3))
        .edges(vec![(0, 1, 1.0)].into_iter())
        .build()
        .unwrap();
    assert!(matches!(
        LeichtNewman::<usize>::leicht_newman(&matrix, &LeichtNewmanConfig::default()),
        Err(ModularityError::NonSquareMatrix { .. })
    ));
}
